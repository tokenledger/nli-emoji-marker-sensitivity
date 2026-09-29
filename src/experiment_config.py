"""Load, validate, hash, and materialize frozen experiment configurations."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_v6.json"


def canonical_config_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: Mapping[str, Any]) -> str:
    payload = canonical_config_json(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def data_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that determine source rows and text transformations."""

    return {
        "schema_version": config["schema_version"],
        "dataset_protocol": deepcopy(config["dataset_protocol"]),
        "transformations": deepcopy(config["transformations"]),
    }


def data_contract_hash(config: Mapping[str, Any]) -> str:
    return config_hash(data_contract(config))


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "experiment_id", "frozen", "primary_hypotheses",
        "dataset_protocol", "transformations", "systems", "models", "training",
        "statistics",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Experiment config is missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError(f"Unsupported config schema: {config['schema_version']}")
    if config["frozen"] is not True:
        raise ValueError("Principal experiment configuration must be frozen")

    hypothesis_ids = [item["id"] for item in config["primary_hypotheses"]]
    if hypothesis_ids != ["H1", "H2", "H3", "H4"]:
        raise ValueError("Exactly four ordered primary hypotheses H1-H4 are required")

    transforms = config["transformations"]
    if not isinstance(transforms.get("generation_seed"), int):
        raise ValueError("Transform generation requires a frozen integer seed")
    multi_nli_protocol = config["dataset_protocol"]["multi_nli"]
    exclusion_ids = set()
    for dataset_key, protocol in config["dataset_protocol"].items():
        hub_name = protocol.get("huggingface_name")
        if not isinstance(hub_name, str) or "/" not in hub_name:
            raise ValueError(
                f"Dataset {dataset_key!r} requires a canonical namespaced Hub ID"
            )
        for rule in protocol.get("source_exclusions", []):
            required_rule_keys = {
                "id", "field", "match", "value", "expected_total_matches", "reason",
            }
            allowed_rule_keys = required_rule_keys | {"expected_matches_by_split"}
            if not required_rule_keys <= set(rule) or not set(rule) <= allowed_rule_keys:
                raise ValueError(
                    f"Dataset {dataset_key!r} has malformed source exclusion: {rule!r}"
                )
            if rule["id"] in exclusion_ids:
                raise ValueError(f"Duplicate source exclusion ID: {rule['id']}")
            exclusion_ids.add(rule["id"])
            if rule["field"] not in {"premise", "hypothesis"}:
                raise ValueError(f"Invalid source exclusion field: {rule['field']!r}")
            if rule["match"] not in {"exact", "prefix"}:
                raise ValueError(f"Invalid source exclusion match: {rule['match']!r}")
            if not isinstance(rule["value"], str) or not rule["value"]:
                raise ValueError("Source exclusion values must be non-empty strings")
            if not isinstance(rule["expected_total_matches"], int):
                raise ValueError("Source exclusion expected counts must be integers")
            if rule["expected_total_matches"] < 1:
                raise ValueError("Source exclusions must declare at least one expected match")
            if not isinstance(rule["reason"], str) or not rule["reason"]:
                raise ValueError("Source exclusions require a reason")
            expected_by_split = rule.get("expected_matches_by_split")
            if expected_by_split is not None:
                if (
                    not isinstance(expected_by_split, dict)
                    or not expected_by_split
                    or any(
                        not isinstance(key, str)
                        or not isinstance(value, int)
                        or value < 0
                        for key, value in expected_by_split.items()
                    )
                ):
                    raise ValueError(
                        "Source exclusion split counts must map names to nonnegative integers"
                    )
                if sum(expected_by_split.values()) != rule["expected_total_matches"]:
                    raise ValueError(
                        "Source exclusion split counts must sum to the expected total"
                    )
        exclusion_stage = protocol.get(
            "source_exclusion_stage", "before_train_development_selection"
        )
        if exclusion_stage not in {
            "before_train_development_selection",
            "after_train_development_selection",
        }:
            raise ValueError(
                f"Dataset {dataset_key!r} has invalid source exclusion stage: "
                f"{exclusion_stage!r}"
            )
    if "development_seed" not in multi_nli_protocol:
        raise ValueError("MultiNLI development split requires a frozen seed")
    if not isinstance(multi_nli_protocol["development_seed"], int):
        raise ValueError("MultiNLI development seed must be an integer")
    train_markers = set(transforms["train_markers"])
    held_out_markers = set(transforms["held_out_markers"])
    excluded = set(transforms["excluded_markers_pending_validation"])
    if train_markers & held_out_markers:
        raise ValueError("Training and held-out markers must be disjoint")
    if excluded & (train_markers | held_out_markers):
        raise ValueError("Excluded markers cannot enter principal marker pools")
    emoji_values = list(transforms["main_emoji_map"].values())
    if len(emoji_values) != len(set(emoji_values)):
        raise ValueError("Principal emoji map must be one-to-one")
    if not 1 <= transforms["max_emoji_replacements"] <= 2:
        raise ValueError("Principal examples must contain at most two emoji replacements")
    rare_controls = transforms.get("rare_string_controls")
    if not isinstance(rare_controls, list) or not rare_controls:
        raise ValueError("Emoji mechanism analysis requires frozen rare-string controls")
    if any(not isinstance(value, str) or not value for value in rare_controls):
        raise ValueError("Rare-string controls must be non-empty strings")
    psych = transforms.get("psych")
    expected_psych = {
        "marker": "psych",
        "placement": "hypothesis_suffix",
        "scope": "complete_hypothesis",
        "label_policy": "invert_entailment_contradiction",
        "evaluation_only": True,
    }
    if psych is not None and psych != expected_psych:
        raise ValueError(f"Psych evaluation contract mismatch: {psych!r}")
    if config["experiment_id"] in {
        "wnut2026-stage-matched-psych-v4",
        "wnut2026-stage-matched-psych-source-filter-v5",
        "wnut2026-stage-matched-psych-removal-only-v6",
    } and psych is None:
        raise ValueError("The v4/v5/v6 experiments require the psych evaluation contract")
    all_markers = train_markers | held_out_markers
    fold_ids = set()
    for fold in transforms["marker_folds"]:
        if fold["id"] in fold_ids:
            raise ValueError(f"Duplicate marker fold ID: {fold['id']}")
        fold_ids.add(fold["id"])
        fold_train = set(fold["train"])
        fold_test = set(fold["test"])
        if fold_train & fold_test:
            raise ValueError(f"Marker leakage within {fold['id']}")
        if fold_train | fold_test != all_markers:
            raise ValueError(f"Marker fold {fold['id']} does not partition all main markers")

    for name, model in config["models"].items():
        if len(model["seeds"]) < 3:
            raise ValueError(f"Model {name!r} requires at least three seeds")
        if model["learning_rate"] not in model["learning_rate_sweep"]:
            raise ValueError(f"Selected learning rate for {name!r} is absent from its sweep")
        if model.get("tokenizer_backend", "fast") not in {"fast", "slow"}:
            raise ValueError(f"Model {name!r} has an invalid tokenizer backend")
        if model.get("tokenizer_normalization", "default") not in {
            "default", "enabled", "disabled", "spm_byte_fallback",
        }:
            raise ValueError(f"Model {name!r} has invalid tokenizer normalization")
        if model.get("input_preprocessing", "none") not in {"none", "timelm"}:
            raise ValueError(f"Model {name!r} has invalid input preprocessing")
        if model.get("mixed_precision", "auto") not in {"auto", "no", "fp16", "bf16"}:
            raise ValueError(f"Model {name!r} has invalid mixed precision policy")

    expected_seed_mixer = "blake2b-64-v1(run_seed,epoch,index)"
    declared_seed_mixer = config["training"].get("augmentation_seed_mixer")
    if declared_seed_mixer not in {None, expected_seed_mixer}:
        raise ValueError("Training augmentation seed mixer does not match runtime policy")
    if (
        config["experiment_id"] == "wnut2026-stage-matched-psych-removal-only-v6"
        and declared_seed_mixer is None
    ):
        raise ValueError("The v6 experiment must freeze its augmentation seed mixer")

    if config["statistics"]["primary_correction"].lower() != "holm":
        raise ValueError("The frozen primary family must use Holm correction")

    # Guard against configuration/code drift in the registries used at runtime.
    from transforms import (
        EXCLUDED_MARKERS,
        FORMAL_MARKER_CONTROLS,
        HELD_OUT_MARKERS,
        LOSSY_EMOJI_MAP,
        MAIN_EMOJI_MAP,
        RANDOM_PHRASE_CONTROLS,
        PSYCH_LABEL_POLICY,
        PSYCH_MARKER,
        TRAIN_MARKERS,
    )
    expected = {
        "main_emoji_map": MAIN_EMOJI_MAP,
        "lossy_emoji_map_appendix_only": LOSSY_EMOJI_MAP,
        "train_markers": list(TRAIN_MARKERS),
        "held_out_markers": list(HELD_OUT_MARKERS),
        "excluded_markers_pending_validation": list(EXCLUDED_MARKERS),
        "formal_marker_controls": list(FORMAL_MARKER_CONTROLS),
        "random_phrase_controls": list(RANDOM_PHRASE_CONTROLS),
    }
    for key, runtime_value in expected.items():
        if transforms[key] != runtime_value:
            raise ValueError(f"Config/runtime registry mismatch for {key}")
    if psych is not None and (
        psych["marker"] != PSYCH_MARKER
        or psych["label_policy"] != PSYCH_LABEL_POLICY
    ):
        raise ValueError("Config/runtime psych contract mismatch")


def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def model_config(config: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    if model_key not in config["models"]:
        choices = ", ".join(sorted(config["models"]))
        raise ValueError(f"Unknown model {model_key!r}; choose one of: {choices}")
    resolved = deepcopy(config["models"][model_key])
    resolved["prefix"] = model_key
    resolved["model_name"] = resolved.pop("checkpoint")
    resolved["grad_accum"] = resolved.pop("gradient_accumulation_steps")
    resolved["lr"] = resolved.pop("learning_rate")
    return resolved


def write_run_manifest(
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy the frozen config and a provenance manifest into a run directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config_copy = destination / "experiment_config.json"
    config_copy.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash(config),
        "config_source": str(Path(source_path).resolve()) if source_path else str(DEFAULT_CONFIG_PATH),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_overrides": dict(runtime_overrides or {}),
        "python": sys.version,
        "platform": platform.platform(),
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
