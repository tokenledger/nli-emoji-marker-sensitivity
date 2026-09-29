"""Deterministic, provenance-rich publication analysis for the frozen v6 runs.

This module never trains a model.  It combines the frozen statistical reports,
the final data-release manifest, and (when supplied) per-example prediction
bundles.  Every planned analysis row is emitted even when an input is missing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_MODELS = ("electra", "roberta", "roberta_base", "timelm", "bertweet")
MODEL_LABELS = {
    "electra": "ELECTRA-small",
    "roberta": "RoBERTa-large",
    "roberta_base": "RoBERTa-base",
    "timelm": "TimeLM-21",
    "bertweet": "BERTweet",
}
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
FINAL_SPLITS = (
    ("snli", "test"),
    ("multi_nli", "validation_matched"),
    ("multi_nli", "validation_mismatched"),
)
FOLDS = ("fold_1", "fold_2", "fold_3")
PLACEMENTS = ("hypothesis_suffix", "hypothesis_prefix", "premise_suffix")
MARKER_STATUSES = ("seen", "unseen")
CONTROL_TYPES = ("formal", "random")
HYPOTHESES = (
    ("H1", "H1_emoji_vs_clean_baseline"),
    ("H2", "H2_marker_vs_clean_baseline"),
    ("H3", "H3_emoji_normalization_vs_augmentation"),
    ("H4", "H4_hybrid_vs_baseline_combined"),
)
MODAL_VOLUME = "wnut2026-nli-results-v2"


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _point(ci: Mapping[str, Any] | None) -> float | None:
    if not isinstance(ci, Mapping):
        return None
    value = ci.get("point_estimate", ci.get("mean"))
    return float(value) if value is not None else None


def _ci_value(
    report: Mapping[str, Any], approach: str, variant: str, key: str
) -> float | None:
    ci = (
        report.get("confidence_intervals", {})
        .get(approach, {})
        .get(variant)
    )
    if not isinstance(ci, Mapping):
        return None
    if key == "point":
        return _point(ci)
    value = ci.get(key)
    return float(value) if value is not None else None


def parse_statistics_filename(
    path: Path, models: Sequence[str] = DEFAULT_MODELS
) -> tuple[str, str, str, str]:
    """Return model, dataset, split, and fold from a statistics filename."""

    stem = path.stem
    match = re.search(r"_fold_([123])$", stem)
    if not match:
        raise ValueError("filename has no fold suffix")
    fold = f"fold_{match.group(1)}"
    prefix = stem[: match.start()]

    model = next(
        (candidate for candidate in sorted(models, key=len, reverse=True)
         if prefix == candidate or prefix.startswith(f"{candidate}_")),
        None,
    )
    if model is None:
        raise ValueError("filename has no configured model prefix")
    suffix = prefix[len(model):].lstrip("_")
    if suffix in ("snli", "snli_test"):
        return model, "snli", "test", fold
    if suffix == "multi_nli_validation_matched":
        return model, "multi_nli", "validation_matched", fold
    if suffix == "multi_nli_validation_mismatched":
        return model, "multi_nli", "validation_mismatched", fold
    raise ValueError(f"unrecognized dataset/split suffix {suffix!r}")


def discover_statistics(
    roots: Sequence[Path], models: Sequence[str]
) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    cells: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for root in sorted({path.resolve() for path in roots}, key=str):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                key = parse_statistics_filename(path, models)
            except ValueError:
                continue
            payload = load_json(path)
            if not isinstance(payload, Mapping) or not {
                "primary", "confidence_intervals"
            }.issubset(payload):
                continue
            digest = file_sha256(path)
            if key in cells:
                if cells[key]["sha256"] != digest:
                    raise ValueError(
                        f"Conflicting statistics artifacts for {key}: "
                        f"{cells[key]['path']} and {path}"
                    )
                continue
            cells[key] = {
                "path": str(path.resolve()),
                "sha256": digest,
                "payload": payload,
            }
            provenance.append({
                "artifact_type": "statistics",
                "model": key[0],
                "dataset": key[1],
                "split": key[2],
                "fold": key[3],
                "local_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "embedded_identity": False,
                "origin": "local_staged_artifact; remote source not supplied",
            })
    return cells, provenance


def prediction_filename(model: str, dataset: str, split: str) -> str:
    prefix = "roberta_large" if model == "roberta" else model
    split_label = (
        "snli" if dataset == "snli"
        else "mnli_matched" if split == "validation_matched"
        else "mnli_mismatched"
    )
    return f"{prefix}_{split_label}_predictions.json"


def modal_prediction_path(model: str, dataset: str, split: str) -> str:
    final_scope = "test" if dataset == "snli" else split
    if model in {"electra", "timelm", "bertweet"}:
        return (
            f"publication_results/{dataset}/seed_42/"
            f"{model}_baseline_final_{final_scope}/predictions.json"
        )
    if model == "roberta":
        root = (
            "_colab/publication_results_roberta42_snli"
            if dataset == "snli"
            else "_colab/publication_results_roberta42_mnli"
        )
        return (
            f"{root}/{dataset}/seed_42/"
            f"roberta_baseline_final_{final_scope}/predictions.json"
        )
    return (
        f"_colab/publication_results_roberta_base42/{dataset}/seed_42/"
        f"roberta_base_baseline_final_{final_scope}/predictions.json"
    )


class PredictionStore:
    """Memory-bounded access to flattened baseline prediction bundles."""

    def __init__(
        self, directory: Path | None, models: Sequence[str],
        missing: list[dict[str, Any]],
    ) -> None:
        self.directory = directory.resolve() if directory else None
        self.paths: dict[tuple[str, str, str], Path] = {}
        self.provenance: list[dict[str, Any]] = []
        for model in models:
            for dataset, split in FINAL_SPLITS:
                name = prediction_filename(model, dataset, split)
                path = self.directory / name if self.directory else Path(name)
                if self.directory and path.is_file():
                    self.paths[(model, dataset, split)] = path
                    self.provenance.append({
                        "artifact_type": "baseline_predictions",
                        "model": model,
                        "dataset": dataset,
                        "split": split,
                        "seed": 42,
                        "local_path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                        "modal_volume": MODAL_VOLUME,
                        "modal_path": modal_prediction_path(model, dataset, split),
                        "identity_manifest": "not supplied in flattened bundle",
                    })
                else:
                    missing.append({
                        "artifact_type": "baseline_predictions",
                        "model": model,
                        "dataset": dataset,
                        "split": split,
                        "fold": None,
                        "status": "missing_input",
                        "expected_path": str(path),
                        "reason": "paired prediction bundle was not supplied",
                    })

    def get(self, model: str, dataset: str, split: str) -> dict[str, Any] | None:
        path = self.paths.get((model, dataset, split))
        if path is None:
            return None
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Prediction bundle is not an object: {path}")
        return payload


class DataRelease:
    """Read-only index over a prepared data release."""

    def __init__(self, root: Path | None) -> None:
        self.root = root.resolve() if root else None
        self.eval_root = self.root / "eval_sets" if self.root else None
        self.release_manifest: dict[str, Any] | None = None
        self.dataset_manifest: dict[str, Any] | None = None
        self.audits: dict[tuple[str, str], dict[str, Any]] = {}
        if not self.root:
            return
        release_path = self.root / "release_manifest.json"
        dataset_path = self.eval_root / "dataset_manifest.json"
        if not release_path.is_file() or not dataset_path.is_file():
            raise FileNotFoundError(
                f"Data release lacks release/dataset manifest: {self.root}"
            )
        self.release_manifest = load_json(release_path)
        self.dataset_manifest = load_json(dataset_path)
        for dataset, split in FINAL_SPLITS:
            audit_path = self.eval_root / dataset / "final" / split / "audit_report.json"
            if audit_path.is_file():
                self.audits[(dataset, split)] = load_json(audit_path)

    @property
    def available(self) -> bool:
        return self.root is not None

    def split_record(self, dataset: str, split: str) -> Mapping[str, Any] | None:
        if self.dataset_manifest is None:
            return None
        return (
            self.dataset_manifest.get("datasets", {})
            .get(dataset, {})
            .get("splits", {})
            .get(f"final:{split}")
        )

    def examples(self, dataset: str, split: str, variant: str) -> int | None:
        record = self.split_record(dataset, split)
        if not isinstance(record, Mapping):
            return None
        if variant == "original":
            value = record.get("source_examples")
        else:
            value = record.get("variants", {}).get(variant, {}).get("examples")
        return int(value) if value is not None else None

    def variant_path(self, dataset: str, split: str, variant: str) -> Path | None:
        if self.eval_root is None:
            return None
        record = self.split_record(dataset, split)
        if not isinstance(record, Mapping):
            return None
        if variant == "original":
            relative = f"{record['base_path']}/original"
        else:
            relative = record.get("variants", {}).get(variant, {}).get("path")
        return self.eval_root / relative if relative else None

    def audit(self, dataset: str, split: str, variant: str) -> Mapping[str, Any] | None:
        value = self.audits.get((dataset, split), {}).get(variant)
        return value if isinstance(value, Mapping) else None

    def provenance(self) -> dict[str, Any] | None:
        if not self.root or self.release_manifest is None:
            return None
        release_path = self.root / "release_manifest.json"
        dataset_path = self.eval_root / "dataset_manifest.json"
        return {
            "artifact_type": "data_release",
            "local_path": str(self.root),
            "release_id": self.release_manifest.get("release_id"),
            "config_sha256": self.release_manifest.get("config_sha256"),
            "release_manifest_sha256": file_sha256(release_path),
            "dataset_manifest_sha256": file_sha256(dataset_path),
        }


def _validated_artifact_manifest(
    root: Path,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Validate every file declared by a generated artifact manifest."""

    payload = load_json(manifest_path)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("files"), list
    ):
        raise ValueError(f"Invalid artifact manifest: {manifest_path}")
    root = root.resolve()
    records: dict[str, dict[str, Any]] = {}
    for raw_record in payload["files"]:
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"Invalid file record in {manifest_path}")
        relative = Path(str(raw_record.get("path", "")))
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError(
                f"Unsafe artifact-manifest path {relative!s}"
            )
        key = relative.as_posix()
        if key in records:
            raise ValueError(f"Duplicate artifact-manifest path {key}")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Artifact-manifest path escapes its root: {key}"
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Artifact manifest references a missing file: {candidate}"
            )
        expected_size = raw_record.get("size_bytes")
        expected_sha256 = raw_record.get("sha256")
        if (
            not isinstance(expected_size, int)
            or expected_size != candidate.stat().st_size
        ):
            raise ValueError(f"Artifact size mismatch for {candidate}")
        observed_sha256 = file_sha256(candidate)
        if expected_sha256 != observed_sha256:
            raise ValueError(f"Artifact SHA-256 mismatch for {candidate}")
        records[key] = dict(raw_record)
    return records


def load_marker_token_associations(
    evidence_path: Path | None,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    models: Sequence[str],
    release_manifest_sha256: str | None,
    source_manifest_sha256: str | None,
    missing: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Load and verify frozen-training marker token/label association evidence."""

    if evidence_path is None:
        missing.append({
            "artifact_type": "marker_training_token_associations",
            "model": None,
            "dataset": None,
            "split": None,
            "fold": None,
            "status": "not_run",
            "expected_path": None,
            "reason": (
                "per-marker accuracy is computed, but frozen-training token "
                "frequency/PMI evidence was not supplied"
            ),
        })
        return [], None, None

    root = evidence_path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            "Marker-token association evidence must be the generated output "
            f"directory: {root}"
        )
    rows_path = root / "marker_token_associations.json"
    provenance_path = root / "provenance.json"
    manifest_path = root / "artifact_manifest.json"
    for required in (rows_path, provenance_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Marker-token association evidence is incomplete: {required}"
            )

    manifest_records = _validated_artifact_manifest(root, manifest_path)
    for required_name in (
        "marker_token_associations.json",
        "provenance.json",
    ):
        if required_name not in manifest_records:
            raise ValueError(
                f"Artifact manifest does not declare {required_name}"
            )

    evidence_provenance = load_json(provenance_path)
    if not isinstance(evidence_provenance, Mapping):
        raise ValueError(
            f"Marker-token provenance is not an object: {provenance_path}"
        )
    if evidence_provenance.get("config_sha256") != config_sha256:
        raise ValueError(
            "Marker-token association evidence/config hash mismatch"
        )
    if (
        release_manifest_sha256 is not None
        and evidence_provenance.get("release_manifest_sha256")
        != release_manifest_sha256
    ):
        raise ValueError(
            "Marker-token association evidence/release-manifest hash mismatch"
        )
    if (
        source_manifest_sha256 is not None
        and evidence_provenance.get("source_manifest_sha256")
        != source_manifest_sha256
    ):
        raise ValueError(
            "Marker-token association evidence/source-manifest hash mismatch"
        )

    requested_models = set(models)
    declared_models = {
        str(value) for value in evidence_provenance.get("models", [])
    }
    if not requested_models.issubset(declared_models):
        absent = sorted(requested_models - declared_models)
        raise ValueError(
            f"Marker-token association evidence lacks models: {absent}"
        )
    declared_datasets = {
        str(value) for value in evidence_provenance.get("datasets", [])
    }
    required_datasets = {"snli", "multi_nli"}
    if not required_datasets.issubset(declared_datasets):
        raise ValueError(
            "Marker-token association evidence lacks SNLI or MultiNLI"
        )

    payload = load_json(rows_path)
    if not isinstance(payload, list):
        raise ValueError(
            f"Marker-token association table is not an array: {rows_path}"
        )
    required_columns = {
        "model", "dataset", "marker", "marker_context", "token_position",
        "marker_token_count", "token_id", "token", "training_examples",
        "training_tokens", "subtoken_training_frequency",
        "subtoken_entailment_occurrences", "subtoken_entailment_pmi",
        "subtoken_neutral_occurrences", "subtoken_neutral_pmi",
        "subtoken_contradiction_occurrences",
        "subtoken_contradiction_pmi", "phrase_training_occurrences",
        "status",
    }
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(payload):
        if not isinstance(raw_row, Mapping):
            raise ValueError(
                f"Marker-token association row {index} is not an object"
            )
        model = str(raw_row.get("model"))
        if model not in requested_models:
            continue
        absent_columns = required_columns - set(raw_row)
        if absent_columns:
            raise ValueError(
                f"Marker-token association row {index} lacks "
                f"{sorted(absent_columns)}"
            )
        if raw_row.get("status") != "ok":
            raise ValueError(
                "Supplied marker-token association evidence is incomplete: "
                f"row {index} has status {raw_row.get('status')!r}"
            )
        rows.append(dict(raw_row))

    marker_folds = (
        config.get("transformations", {}).get("marker_folds", [])
    )
    expected_markers = {
        str(marker)
        for fold in marker_folds
        for group in ("train", "test")
        for marker in fold.get(group, [])
    }
    expected_groups = {
        (model, dataset, marker, context)
        for model in models
        for dataset in sorted(required_datasets)
        for marker in expected_markers
        for context in ("sentence_start", "continuation")
    }
    observed_groups = {
        (
            str(row["model"]),
            str(row["dataset"]),
            str(row["marker"]),
            str(row["marker_context"]),
        )
        for row in rows
    }
    if observed_groups != expected_groups:
        absent = sorted(expected_groups - observed_groups)
        extra = sorted(observed_groups - expected_groups)
        raise ValueError(
            "Marker-token association coverage mismatch: "
            f"missing={absent[:5]}, extra={extra[:5]}"
        )

    grouped_positions: dict[
        tuple[str, str, str, str], list[tuple[int, int]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            str(row["dataset"]),
            str(row["marker"]),
            str(row["marker_context"]),
        )
        grouped_positions[key].append((
            int(row["token_position"]),
            int(row["marker_token_count"]),
        ))
    for key, values in grouped_positions.items():
        positions = sorted(position for position, _ in values)
        declared_counts = {count for _, count in values}
        if len(declared_counts) != 1 or positions != list(range(len(values))):
            raise ValueError(
                f"Invalid marker token positions/count for {key}: {values}"
            )
        if declared_counts != {len(values)}:
            raise ValueError(
                f"Marker token count does not match rows for {key}"
            )

    model_order = {model: index for index, model in enumerate(models)}
    dataset_order = {"snli": 0, "multi_nli": 1}
    context_order = {"sentence_start": 0, "continuation": 1}
    rows.sort(key=lambda row: (
        model_order[str(row["model"])],
        dataset_order[str(row["dataset"])],
        str(row["marker"]),
        context_order[str(row["marker_context"])],
        int(row["token_position"]),
        int(row["token_id"]),
    ))

    association_metadata = dict(evidence_provenance)
    association_metadata["artifact_manifest_sha256"] = file_sha256(
        manifest_path
    )
    association_metadata["rows"] = len(rows)
    input_artifact = {
        "artifact_type": "marker_training_token_associations",
        "local_path": str(rows_path),
        "size_bytes": rows_path.stat().st_size,
        "sha256": file_sha256(rows_path),
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "evidence_provenance_sha256": file_sha256(provenance_path),
        "evidence_schema_version": evidence_provenance.get("schema_version"),
        "config_sha256": evidence_provenance.get("config_sha256"),
        "release_manifest_sha256": evidence_provenance.get(
            "release_manifest_sha256"
        ),
        "source_manifest_sha256": evidence_provenance.get(
            "source_manifest_sha256"
        ),
        "models": list(models),
        "datasets": sorted(required_datasets),
        "rows": len(rows),
        "manifest_verified": True,
    }
    return rows, input_artifact, association_metadata


def _combined_variant(fold: str) -> str:
    return "emoji_marker_combined" if fold == "fold_1" else f"emoji_marker_combined_{fold}"


def _variant_for_hypothesis(hypothesis: str, fold: str) -> str:
    if hypothesis in {"H1", "H3"}:
        return "emoji_raw"
    if hypothesis == "H2":
        return f"marker_unseen_{fold}_hypothesis_suffix"
    return _combined_variant(fold)


def build_primary_rows(
    statistics: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    data: DataRelease,
    models: Sequence[str],
    missing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for dataset, split in FINAL_SPLITS:
            for fold in FOLDS:
                cell = statistics.get((model, dataset, split, fold))
                if cell is None:
                    missing.append({
                        "artifact_type": "statistics",
                        "model": model,
                        "dataset": dataset,
                        "split": split,
                        "fold": fold,
                        "status": "missing_input",
                        "expected_path": None,
                        "reason": "no statistical report matched this planned cell",
                    })
                for hypothesis, test_name in HYPOTHESES:
                    base = {
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "dataset": dataset,
                        "split": split,
                        "fold": fold,
                        "seed": 42,
                        "hypothesis": hypothesis,
                        "test_name": test_name,
                        "family_id": f"primary:{model}:{dataset}:{split}:{fold}",
                        "family_size": 4,
                        "correction": "Holm",
                        "alpha": 0.05,
                        "status": "missing_input" if cell is None else "ok",
                    }
                    if cell is None:
                        rows.append(base)
                        continue
                    report = cell["payload"]
                    test = (
                        report.get("primary", {})
                        .get("tests", {})
                        .get(test_name)
                    )
                    if not isinstance(test, Mapping):
                        base["status"] = "missing_test"
                        base["reason"] = f"statistics report lacks {test_name}"
                        rows.append(base)
                        continue
                    variant = _variant_for_hypothesis(hypothesis, fold)
                    n = data.examples(dataset, split, variant)
                    if n is None:
                        base["status"] = "missing_example_count"
                        base["reason"] = "data-release manifest is unavailable"
                        rows.append(base)
                        continue
                    a_wins = int(test["n_a_wins"])
                    b_wins = int(test["n_b_wins"])
                    if hypothesis in {"H1", "H2"}:
                        sign = -1
                        treatment_system = "baseline"
                        reference_system = "baseline"
                        treatment_variant = variant
                        reference_variant = (
                            "paired_clean_emoji_raw" if hypothesis == "H1"
                            else "original"
                        )
                    elif hypothesis == "H3":
                        sign = 1
                        treatment_system = "preprocessing"
                        reference_system = "augmented"
                        treatment_variant = reference_variant = "emoji_raw"
                    else:
                        sign = 1
                        treatment_system = "hybrid"
                        reference_system = "baseline"
                        treatment_variant = reference_variant = variant
                    difference = sign * (a_wins - b_wins) / n
                    treatment_accuracy = _ci_value(
                        report, treatment_system, treatment_variant, "point"
                    )
                    if hypothesis == "H1":
                        reference_accuracy = (
                            treatment_accuracy - difference
                            if treatment_accuracy is not None else None
                        )
                        reference_source = "derived_exactly_from_paired_difference"
                        reference_lower = reference_upper = None
                    else:
                        reference_accuracy = _ci_value(
                            report, reference_system, reference_variant, "point"
                        )
                        reference_source = "statistics_confidence_intervals"
                        reference_lower = _ci_value(
                            report, reference_system, reference_variant, "lower"
                        )
                        reference_upper = _ci_value(
                            report, reference_system, reference_variant, "upper"
                        )
                    base.update({
                        "treatment_system": treatment_system,
                        "treatment_variant": treatment_variant,
                        "reference_system": reference_system,
                        "reference_variant": reference_variant,
                        "examples": n,
                        "treatment_accuracy": treatment_accuracy,
                        "treatment_ci_lower": _ci_value(
                            report, treatment_system, treatment_variant, "lower"
                        ),
                        "treatment_ci_upper": _ci_value(
                            report, treatment_system, treatment_variant, "upper"
                        ),
                        "reference_accuracy": reference_accuracy,
                        "reference_ci_lower": reference_lower,
                        "reference_ci_upper": reference_upper,
                        "reference_accuracy_source": reference_source,
                        "difference": difference,
                        "difference_pp": 100 * difference,
                        "treatment_only_correct": (
                            b_wins if hypothesis in {"H1", "H2"} else a_wins
                        ),
                        "reference_only_correct": (
                            a_wins if hypothesis in {"H1", "H2"} else b_wins
                        ),
                        "mcnemar_statistic": float(test["statistic"]),
                        "raw_pvalue": float(test["pvalue"]),
                        "holm_adjusted_pvalue": float(test["holm_adjusted_pvalue"]),
                        "significant_holm": bool(test["significant_holm"]),
                        "statistics_sha256": cell["sha256"],
                    })
                    rows.append(base)
    return rows


def build_primary_summary(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        for hypothesis, _ in HYPOTHESES:
            selected = [
                row for row in rows
                if row["model"] == model
                and row["hypothesis"] == hypothesis
                and row.get("status") == "ok"
                and (hypothesis != "H1" or row["fold"] == "fold_1")
            ]
            effects = [float(row["difference_pp"]) for row in selected]
            checkpoint = config.get("models", {}).get(model, {}).get("checkpoint")
            summaries.append({
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "checkpoint": checkpoint,
                "hypothesis": hypothesis,
                "seed": 42,
                "analysis_cells": len(selected),
                "expected_cells": 3 if hypothesis == "H1" else 9,
                "fold_handling": (
                    "fold-independent result deduplicated"
                    if hypothesis == "H1" else "all three marker folds"
                ),
                "mean_difference_pp": mean(effects) if effects else None,
                "min_difference_pp": min(effects) if effects else None,
                "max_difference_pp": max(effects) if effects else None,
                "positive_cells": sum(value > 0 for value in effects),
                "negative_cells": sum(value < 0 for value in effects),
                "holm_significant_cells": sum(
                    bool(row.get("significant_holm")) for row in selected
                ),
                "uncertainty_scope": (
                    "paired-example tests at seed 42; no training-seed variance"
                ),
                "status": "ok" if len(selected) == (3 if hypothesis == "H1" else 9)
                else "incomplete",
            })
    return summaries


def _exploratory_system_test(
    report: Mapping[str, Any],
    *,
    treatment_system: str,
    reference_system: str,
    variant: str,
    examples: int,
) -> dict[str, Any] | None:
    tests = report.get("exploratory", {}).get("tests", {})
    forward = f"{reference_system}_vs_{treatment_system}"
    reverse = f"{treatment_system}_vs_{reference_system}"
    if isinstance(tests.get(forward, {}).get(variant), Mapping):
        test = tests[forward][variant]
        treatment_wins = int(test["n_b_wins"])
        reference_wins = int(test["n_a_wins"])
    elif isinstance(tests.get(reverse, {}).get(variant), Mapping):
        test = tests[reverse][variant]
        treatment_wins = int(test["n_a_wins"])
        reference_wins = int(test["n_b_wins"])
    else:
        return None
    return {
        "treatment_only_correct": treatment_wins,
        "reference_only_correct": reference_wins,
        "paired_difference_pp": (
            100 * (treatment_wins - reference_wins) / examples
        ),
        "raw_pvalue": float(test["pvalue"]),
        "exploratory_broad_holm_adjusted_pvalue": float(
            test["holm_adjusted_pvalue"]
        ),
        "exploratory_broad_significant_holm": bool(
            test["significant_holm"]
        ),
    }


def build_augmentation_transfer_rows(
    statistics: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    data: DataRelease,
    models: Sequence[str],
) -> list[dict[str, Any]]:
    """Report marker-augmentation transfer, distinct from baseline sensitivity."""

    rows: list[dict[str, Any]] = []
    for model in models:
        for dataset, split in FINAL_SPLITS:
            for fold in FOLDS:
                cell = statistics.get((model, dataset, split, fold))
                planned = [
                    ("augmented", "seen"),
                    ("augmented", "unseen"),
                    ("hybrid", "unseen"),
                ]
                for treatment_system, marker_status in planned:
                    variant = _marker_variant(
                        fold, marker_status, "hypothesis_suffix"
                    )
                    row = {
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "dataset": dataset,
                        "split": split,
                        "fold": fold,
                        "seed": 42,
                        "marker_status": marker_status,
                        "placement": "hypothesis_suffix",
                        "variant": variant,
                        "treatment_system": treatment_system,
                        "reference_system": "baseline",
                        "contrast": f"{treatment_system}_minus_baseline",
                        "claim_status": "exploratory_not_preregistered",
                    }
                    if cell is None:
                        row.update({
                            "status": "missing_statistics",
                            "reason": "statistical report is unavailable",
                        })
                        rows.append(row)
                        continue
                    n = data.examples(dataset, split, variant)
                    if n is None:
                        row.update({
                            "status": "missing_example_count",
                            "reason": "data-release manifest is unavailable",
                        })
                        rows.append(row)
                        continue
                    report = cell["payload"]
                    treatment_accuracy = _ci_value(
                        report, treatment_system, variant, "point"
                    )
                    reference_accuracy = _ci_value(
                        report, "baseline", variant, "point"
                    )
                    paired = _exploratory_system_test(
                        report,
                        treatment_system=treatment_system,
                        reference_system="baseline",
                        variant=variant,
                        examples=n,
                    )
                    if treatment_accuracy is None or reference_accuracy is None:
                        row.update({
                            "status": "missing_point_estimate",
                            "reason": "confidence-interval point estimate is unavailable",
                        })
                        rows.append(row)
                        continue
                    point_difference_pp = 100 * (
                        treatment_accuracy - reference_accuracy
                    )
                    row.update({
                        "examples": n,
                        "treatment_accuracy": treatment_accuracy,
                        "reference_accuracy": reference_accuracy,
                        "difference_pp": point_difference_pp,
                        "treatment_ci_lower": _ci_value(
                            report, treatment_system, variant, "lower"
                        ),
                        "treatment_ci_upper": _ci_value(
                            report, treatment_system, variant, "upper"
                        ),
                        "reference_ci_lower": _ci_value(
                            report, "baseline", variant, "lower"
                        ),
                        "reference_ci_upper": _ci_value(
                            report, "baseline", variant, "upper"
                        ),
                        "paired_test_status": (
                            "available_existing_exploratory"
                            if paired else "not_available_point_contrast_descriptive"
                        ),
                        "exploratory_family_id": (
                            f"all_system_pairs_by_variant:{model}:{dataset}:"
                            f"{split}:{fold}"
                        ),
                        "exploratory_family_size": (
                            report.get("exploratory", {}).get("n_tests")
                        ),
                        "status": "ok",
                    })
                    if paired:
                        row.update(paired)
                        if abs(
                            float(paired["paired_difference_pp"])
                            - point_difference_pp
                        ) > 1e-8:
                            raise ValueError(
                                f"Paired/point transfer contrast mismatch for "
                                f"{model}/{dataset}/{split}/{fold}/{variant}"
                            )
                    rows.append(row)
    return rows


def build_augmentation_transfer_summary(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    groupings = (
        (
            "model_overall",
            lambda row: (
                str(row["model"]), str(row["treatment_system"]),
                str(row["marker_status"]),
            ),
            9,
        ),
        (
            "all_models",
            lambda row: (
                "all_models", str(row["treatment_system"]),
                str(row["marker_status"]),
            ),
            45,
        ),
    )
    for scope, key_function, expected in groupings:
        grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[key_function(row)].append(row)
        for (model, system, status), values in sorted(grouped.items()):
            complete = [row for row in values if row.get("status") == "ok"]
            effects = [float(row["difference_pp"]) for row in complete]
            summaries.append({
                "aggregation_scope": scope,
                "model": model,
                "model_label": (
                    "All five models" if model == "all_models"
                    else MODEL_LABELS.get(model, model)
                ),
                "treatment_system": system,
                "marker_status": status,
                "placement": "hypothesis_suffix",
                "cells": len(complete),
                "expected_cells": expected,
                "mean_treatment_minus_baseline_pp": (
                    mean(effects) if effects else None
                ),
                "min_treatment_minus_baseline_pp": min(effects) if effects else None,
                "max_treatment_minus_baseline_pp": max(effects) if effects else None,
                "positive_cells": sum(effect > 0 for effect in effects),
                "negative_cells": sum(effect < 0 for effect in effects),
                "broad_holm_significant_cells": sum(
                    bool(row.get("exploratory_broad_significant_holm"))
                    for row in complete
                ),
                "claim_status": "exploratory_not_preregistered",
                "status": "ok" if len(complete) == expected else "incomplete",
            })
    return summaries


def validate_prediction_record(
    record: Mapping[str, Any], *, name: str, expected_examples: int | None = None
) -> None:
    required = ("predictions", "labels", "source_indices")
    if any(not isinstance(record.get(key), list) for key in required):
        raise ValueError(f"Malformed prediction record {name!r}")
    lengths = {len(record[key]) for key in required}
    if len(lengths) != 1:
        raise ValueError(f"Prediction record {name!r} has unequal array lengths")
    size = next(iter(lengths))
    if expected_examples is not None and size != expected_examples:
        raise ValueError(
            f"Prediction record {name!r} has {size} examples; "
            f"manifest declares {expected_examples}"
        )
    source_ids = [int(value) for value in record["source_indices"]]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(f"Prediction record {name!r} has duplicate source IDs")
    if any(int(label) not in (0, 1, 2) for label in record["labels"]):
        raise ValueError(f"Prediction record {name!r} has an invalid NLI label")
    condition = record.get("condition_metadata", {}).get("condition")
    if condition is not None and condition != name:
        raise ValueError(
            f"Prediction record key/condition mismatch: {name!r} versus {condition!r}"
        )


def _aligned_vectors(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    source_ids: Sequence[int] | None = None,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Align two prediction records by source ID and validate their labels."""

    first_name = str(
        first.get("condition_metadata", {}).get("condition", "first")
    )
    second_name = str(
        second.get("condition_metadata", {}).get("condition", "second")
    )
    validate_prediction_record(first, name=first_name)
    validate_prediction_record(second, name=second_name)
    first_map = {
        int(source): (int(prediction), int(label))
        for source, prediction, label in zip(
            first["source_indices"], first["predictions"], first["labels"]
        )
    }
    second_map = {
        int(source): (int(prediction), int(label))
        for source, prediction, label in zip(
            second["source_indices"], second["predictions"], second["labels"]
        )
    }
    selected = (
        [int(value) for value in source_ids]
        if source_ids is not None
        else sorted(first_map)
    )
    if len(set(selected)) != len(selected):
        raise ValueError("Requested comparison source IDs contain duplicates")
    if source_ids is None and set(first_map) != set(second_map):
        raise ValueError("Paired prediction records have different source-ID sets")
    first_predictions: list[int] = []
    second_predictions: list[int] = []
    labels: list[int] = []
    for source in selected:
        if source not in first_map or source not in second_map:
            raise ValueError(f"Paired prediction record is missing source {source}")
        first_prediction, first_label = first_map[source]
        second_prediction, second_label = second_map[source]
        if first_label != second_label:
            raise ValueError(f"Paired labels disagree for source {source}")
        first_predictions.append(first_prediction)
        second_predictions.append(second_prediction)
        labels.append(first_label)
    return first_predictions, second_predictions, labels, selected


def paired_test(
    treatment: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    source_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Continuity-corrected paired McNemar test, treatment minus reference."""

    first, second, labels, selected = _aligned_vectors(
        treatment, reference, source_ids=source_ids
    )
    if not selected:
        raise ValueError("Cannot compare an empty paired subset")
    treatment_correct = [a == label for a, label in zip(first, labels)]
    reference_correct = [b == label for b, label in zip(second, labels)]
    treatment_wins = sum(a and not b for a, b in zip(
        treatment_correct, reference_correct
    ))
    reference_wins = sum(b and not a for a, b in zip(
        treatment_correct, reference_correct
    ))
    discordant = treatment_wins + reference_wins
    statistic = (
        ((abs(treatment_wins - reference_wins) - 1) ** 2) / discordant
        if discordant else 0.0
    )
    pvalue = math.erfc(math.sqrt(statistic / 2))
    n = len(selected)
    treatment_accuracy = sum(treatment_correct) / n
    reference_accuracy = sum(reference_correct) / n
    return {
        "examples": n,
        "treatment_accuracy": treatment_accuracy,
        "reference_accuracy": reference_accuracy,
        "difference": treatment_accuracy - reference_accuracy,
        "difference_pp": 100 * (treatment_accuracy - reference_accuracy),
        "treatment_only_correct": treatment_wins,
        "reference_only_correct": reference_wins,
        "mcnemar_statistic": statistic,
        "raw_pvalue": pvalue,
    }


def predicted_label_distribution(
    record: Mapping[str, Any], source_ids: Sequence[int]
) -> dict[str, Any]:
    name = str(record.get("condition_metadata", {}).get("condition", "record"))
    validate_prediction_record(record, name=name)
    prediction_by_source = {
        int(source): int(prediction)
        for source, prediction in zip(
            record["source_indices"], record["predictions"]
        )
    }
    selected = [int(value) for value in source_ids]
    try:
        predictions = [prediction_by_source[source] for source in selected]
    except KeyError as error:
        raise ValueError(
            f"Prediction record is missing per-marker source {error.args[0]}"
        ) from error
    counts = {
        label: sum(prediction == index for prediction in predictions)
        for index, label in LABEL_NAMES.items()
    }
    n = len(predictions)
    return {
        **{
            f"predicted_{label}_count": count
            for label, count in counts.items()
        },
        **{
            f"predicted_{label}_rate": count / n if n else None
            for label, count in counts.items()
        },
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    count = len(pvalues)
    order = sorted(range(count), key=lambda index: (pvalues[index], index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(pvalues[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _control_family_id(
    scope: str, model: str, dataset: str, split: str
) -> str:
    if scope == "global":
        return "marker_controls:all_models_and_splits"
    if scope == "model":
        return f"marker_controls:{model}"
    if scope == "model_split":
        return f"marker_controls:{model}:{dataset}:{split}"
    raise ValueError(f"Unknown control family scope: {scope}")


def _marker_variant(fold: str, status: str, placement: str) -> str:
    return f"marker_{status}_{fold}_{placement}"


def _planned_control_rows(
    model: str, dataset: str, split: str, family_scope: str
) -> Iterable[dict[str, Any]]:
    for fold in FOLDS:
        for status in MARKER_STATUSES:
            for placement in PLACEMENTS:
                condition = _marker_variant(fold, status, placement)
                for control_type in CONTROL_TYPES:
                    yield {
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "dataset": dataset,
                        "split": split,
                        "seed": 42,
                        "approach": "baseline",
                        "fold": fold,
                        "marker_status": status,
                        "placement": placement,
                        "control_type": control_type,
                        "treatment_variant": condition,
                        "reference_variant": f"{condition}_{control_type}_control",
                        "family_id": _control_family_id(
                            family_scope, model, dataset, split
                        ),
                        "correction": "Holm",
                        "alpha": 0.05,
                    }


def _expected_marker_names(
    config: Mapping[str, Any], fold: str, status: str
) -> list[str]:
    fold_record = next(
        (
            item for item in config.get("transformations", {}).get("marker_folds", [])
            if item.get("id") == fold
        ),
        None,
    )
    if not isinstance(fold_record, Mapping):
        return []
    return list(fold_record["train" if status == "seen" else "test"])


def apply_control_holm(rows: Sequence[dict[str, Any]]) -> None:
    """Adjust complete planned control families in place, fail-closed."""

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    for family_rows in by_family.values():
        family_size = len(family_rows)
        complete = all(row.get("status") == "ok" for row in family_rows)
        for row in family_rows:
            row["family_size"] = family_size
            row["family_status"] = "complete" if complete else "incomplete"
            row["holm_adjusted_pvalue"] = None
            row["significant_holm"] = None
        if complete:
            adjusted = holm_adjust(
                [float(row["raw_pvalue"]) for row in family_rows]
            )
            for row, pvalue in zip(family_rows, adjusted):
                row["holm_adjusted_pvalue"] = pvalue
                row["significant_holm"] = pvalue <= float(row["alpha"])


def load_marker_groups(
    data: DataRelease,
    config: Mapping[str, Any],
    missing: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, list[int]]]:
    """Load source IDs by inserted marker once for all per-marker reports."""

    groups: dict[tuple[str, str, str], dict[str, list[int]]] = {}
    if not data.available:
        missing.append({
            "artifact_type": "per_marker_dataset_rows",
            "model": None,
            "dataset": None,
            "split": None,
            "fold": None,
            "status": "missing_input",
            "expected_path": None,
            "reason": "data release was not supplied; per-marker accuracy is unavailable",
        })
        return groups
    try:
        from datasets import load_from_disk
    except ImportError:
        missing.append({
            "artifact_type": "python_dependency",
            "model": None,
            "dataset": None,
            "split": None,
            "fold": None,
            "status": "missing_dependency",
            "expected_path": None,
            "reason": "datasets is required to read per-marker Arrow rows",
        })
        return groups
    for dataset, split in FINAL_SPLITS:
        for fold in FOLDS:
            for status in MARKER_STATUSES:
                for placement in PLACEMENTS:
                    variant = _marker_variant(fold, status, placement)
                    path = data.variant_path(dataset, split, variant)
                    if path is None or not path.is_dir():
                        continue
                    dataset_rows = load_from_disk(str(path))
                    marker_groups: dict[str, list[int]] = defaultdict(list)
                    for source, metadata in zip(
                        dataset_rows["source_index"], dataset_rows["transform_metadata"]
                    ):
                        marker = metadata.get("marker") if isinstance(metadata, Mapping) else None
                        if not marker:
                            raise ValueError(f"{variant} row lacks inserted-marker metadata")
                        marker_groups[str(marker)].append(int(source))
                    expected = set(_expected_marker_names(config, fold, status))
                    if set(marker_groups) != expected:
                        raise ValueError(
                            f"{variant} marker set {sorted(marker_groups)} does not match "
                            f"config {sorted(expected)}"
                        )
                    audit = data.audit(dataset, split, variant)
                    if isinstance(audit, Mapping):
                        audited = {
                            str(marker): int(count)
                            for marker, count in audit.get("marker_frequencies", {}).items()
                        }
                        observed = {
                            marker: len(ids) for marker, ids in marker_groups.items()
                        }
                        if audited != observed:
                            raise ValueError(
                                f"{variant} Arrow/audit marker counts disagree"
                            )
                    groups[(dataset, split, variant)] = dict(marker_groups)
    return groups


def load_emoji_intensity_groups(
    data: DataRelease,
    missing: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[int, list[int]]]:
    groups: dict[tuple[str, str], dict[int, list[int]]] = {}
    if not data.available:
        return groups
    try:
        from datasets import load_from_disk
    except ImportError:
        return groups
    for dataset, split in FINAL_SPLITS:
        path = data.variant_path(dataset, split, "emoji_raw")
        if path is None or not path.is_dir():
            continue
        dataset_rows = load_from_disk(str(path))
        by_count: dict[int, list[int]] = defaultdict(list)
        for source, metadata in zip(
            dataset_rows["source_index"], dataset_rows["transform_metadata"]
        ):
            events = metadata.get("events", {}) if isinstance(metadata, Mapping) else {}
            replacement_count = sum(
                len(events.get(field, [])) for field in ("premise", "hypothesis")
            )
            by_count[replacement_count].append(int(source))
        audit = data.audit(dataset, split, "emoji_raw")
        if isinstance(audit, Mapping):
            expected = {
                int(count): int(examples)
                for count, examples in audit.get(
                    "emoji_replacements_per_example", {}
                ).items()
            }
            observed = {count: len(ids) for count, ids in by_count.items()}
            if expected != observed:
                raise ValueError(
                    f"{dataset}/{split} emoji intensity Arrow/audit counts disagree"
                )
        groups[(dataset, split)] = dict(by_count)
    if not groups:
        missing.append({
            "artifact_type": "emoji_intensity_rows",
            "model": None,
            "dataset": None,
            "split": None,
            "fold": None,
            "status": "missing_input",
            "expected_path": None,
            "reason": "no emoji Arrow rows were available for intensity analysis",
        })
    return groups


def build_prediction_diagnostics(
    store: PredictionStore,
    data: DataRelease,
    config: Mapping[str, Any],
    statistics: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    models: Sequence[str],
    *,
    family_scope: str,
    marker_groups: Mapping[tuple[str, str, str], Mapping[str, Sequence[int]]],
    intensity_groups: Mapping[tuple[str, str], Mapping[int, Sequence[int]]],
) -> dict[str, list[dict[str, Any]]]:
    control_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    generalization_rows: list[dict[str, Any]] = []
    per_marker_rows: list[dict[str, Any]] = []
    emoji_rows: list[dict[str, Any]] = []
    intensity_rows: list[dict[str, Any]] = []

    for model in models:
        for dataset, split in FINAL_SPLITS:
            payload = store.get(model, dataset, split)
            original = payload.get("original") if payload else None
            if isinstance(original, Mapping):
                validate_prediction_record(
                    original,
                    name="original",
                    expected_examples=data.examples(dataset, split, "original"),
                )

            # Planned condition-versus-control tests.
            for row in _planned_control_rows(
                model, dataset, split, family_scope
            ):
                treatment = payload.get(row["treatment_variant"]) if payload else None
                reference = payload.get(row["reference_variant"]) if payload else None
                if not isinstance(treatment, Mapping) or not isinstance(
                    reference, Mapping
                ):
                    row.update({
                        "status": "missing_input",
                        "reason": "prediction bundle or planned variant is unavailable",
                    })
                else:
                    expected = data.examples(
                        dataset, split, str(row["treatment_variant"])
                    )
                    validate_prediction_record(
                        treatment,
                        name=str(row["treatment_variant"]),
                        expected_examples=expected,
                    )
                    validate_prediction_record(
                        reference,
                        name=str(row["reference_variant"]),
                        expected_examples=expected,
                    )
                    row.update(paired_test(treatment, reference))
                    row["status"] = "ok"
                control_rows.append(row)

            # Marker placement effects relative to the paired clean input and
            # relative to the frozen hypothesis-suffix placement.
            for fold in FOLDS:
                for status in MARKER_STATUSES:
                    anchor_name = _marker_variant(
                        fold, status, "hypothesis_suffix"
                    )
                    anchor = payload.get(anchor_name) if payload else None
                    for placement in PLACEMENTS:
                        variant = _marker_variant(fold, status, placement)
                        record = payload.get(variant) if payload else None
                        row = {
                            "model": model,
                            "model_label": MODEL_LABELS.get(model, model),
                            "dataset": dataset,
                            "split": split,
                            "seed": 42,
                            "approach": "baseline",
                            "fold": fold,
                            "marker_status": status,
                            "placement": placement,
                            "variant": variant,
                            "clean_reference": "original",
                            "placement_reference": anchor_name,
                            "inference_status": "descriptive_unadjusted",
                        }
                        if not isinstance(record, Mapping) or not isinstance(
                            original, Mapping
                        ) or not isinstance(anchor, Mapping):
                            row.update({
                                "status": "missing_input",
                                "reason": "planned prediction variant is unavailable",
                            })
                        else:
                            expected = data.examples(dataset, split, variant)
                            validate_prediction_record(
                                record, name=variant, expected_examples=expected
                            )
                            clean_result = paired_test(record, original)
                            anchor_result = paired_test(record, anchor)
                            row.update({
                                "examples": clean_result["examples"],
                                "accuracy": clean_result["treatment_accuracy"],
                                "clean_accuracy": clean_result["reference_accuracy"],
                                "difference_from_clean_pp": clean_result["difference_pp"],
                                "clean_comparison_raw_pvalue": clean_result["raw_pvalue"],
                                "difference_from_hypothesis_suffix_pp": (
                                    anchor_result["difference_pp"]
                                ),
                                "placement_comparison_raw_pvalue": (
                                    anchor_result["raw_pvalue"]
                                ),
                                "status": "ok",
                            })
                        placement_rows.append(row)

            # Seen versus unseen generalization at matched fold and placement.
            for fold in FOLDS:
                for placement in PLACEMENTS:
                    seen_name = _marker_variant(fold, "seen", placement)
                    unseen_name = _marker_variant(fold, "unseen", placement)
                    seen = payload.get(seen_name) if payload else None
                    unseen = payload.get(unseen_name) if payload else None
                    row = {
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "dataset": dataset,
                        "split": split,
                        "seed": 42,
                        "approach": "baseline",
                        "fold": fold,
                        "placement": placement,
                        "treatment_variant": unseen_name,
                        "reference_variant": seen_name,
                        "contrast": "unseen_minus_seen",
                        "inference_status": "descriptive_unadjusted",
                    }
                    if not isinstance(seen, Mapping) or not isinstance(
                        unseen, Mapping
                    ):
                        row.update({
                            "status": "missing_input",
                            "reason": "seen or unseen prediction variant is unavailable",
                        })
                    else:
                        row.update(paired_test(unseen, seen))
                        row["unseen_accuracy"] = row.pop("treatment_accuracy")
                        row["seen_accuracy"] = row.pop("reference_accuracy")
                        row["unseen_minus_seen_pp"] = row.pop("difference_pp")
                        row.pop("difference")
                        row["status"] = "ok"
                    generalization_rows.append(row)

            # Marker-level accuracy and paired clean degradation.
            for fold in FOLDS:
                for status in MARKER_STATUSES:
                    expected_markers = _expected_marker_names(
                        config, fold, status
                    )
                    for placement in PLACEMENTS:
                        variant = _marker_variant(fold, status, placement)
                        record = payload.get(variant) if payload else None
                        groups = marker_groups.get((dataset, split, variant), {})
                        for marker in expected_markers:
                            source_ids = groups.get(marker)
                            row = {
                                "model": model,
                                "model_label": MODEL_LABELS.get(model, model),
                                "dataset": dataset,
                                "split": split,
                                "seed": 42,
                                "approach": "baseline",
                                "fold": fold,
                                "marker_status": status,
                                "placement": placement,
                                "variant": variant,
                                "marker": marker,
                                "inference_status": "descriptive_unadjusted",
                            }
                            if not isinstance(record, Mapping) or not isinstance(
                                original, Mapping
                            ):
                                row.update({
                                    "status": "missing_predictions",
                                    "reason": "baseline prediction bundle is unavailable",
                                })
                            elif source_ids is None:
                                audit = data.audit(dataset, split, variant)
                                audit_count = (
                                    audit.get("marker_frequencies", {}).get(marker)
                                    if isinstance(audit, Mapping) else None
                                )
                                row.update({
                                    "status": "missing_dataset_rows",
                                    "generation_examples": audit_count,
                                    "reason": (
                                        "Arrow transformation metadata is unavailable; "
                                        "generation count is audit-only"
                                    ),
                                })
                            else:
                                result = paired_test(
                                    record, original, source_ids=source_ids
                                )
                                row.update({
                                    "examples": result["examples"],
                                    "accuracy": result["treatment_accuracy"],
                                    "clean_accuracy": result["reference_accuracy"],
                                    "difference_from_clean_pp": result["difference_pp"],
                                    "raw_pvalue": result["raw_pvalue"],
                                    "correct": round(
                                        result["treatment_accuracy"]
                                        * result["examples"]
                                    ),
                                    "status": "ok",
                                })
                                row.update(
                                    predicted_label_distribution(record, source_ids)
                                )
                            per_marker_rows.append(row)

            # Emoji raw/gloss, normalization, and augmentation diagnostics.
            raw = payload.get("emoji_raw") if payload else None
            gloss = payload.get("emoji_gloss") if payload else None
            clean_raw = payload.get("clean_emoji_raw") if payload else None
            raw_clean_result = (
                paired_test(raw, clean_raw)
                if isinstance(raw, Mapping) and isinstance(clean_raw, Mapping)
                else None
            )
            gloss_raw_result = (
                paired_test(gloss, raw)
                if isinstance(gloss, Mapping) and isinstance(raw, Mapping)
                else None
            )
            for fold in FOLDS:
                stat_cell = statistics.get((model, dataset, split, fold))
                row = {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "dataset": dataset,
                    "split": split,
                    "fold": fold,
                    "seed": 42,
                    "inference_status": (
                        "H3 uses frozen primary Holm family; other contrasts "
                        "are descriptive"
                    ),
                }
                if (
                    raw_clean_result is None
                    or gloss_raw_result is None
                    or stat_cell is None
                ):
                    row.update({
                        "status": "missing_input",
                        "reason": "statistics or baseline emoji predictions unavailable",
                    })
                else:
                    stat_report = stat_cell["payload"]
                    combined = _combined_variant(fold)
                    h3 = (
                        stat_report.get("primary", {})
                        .get("tests", {})
                        .get("H3_emoji_normalization_vs_augmentation", {})
                    )
                    normalization_accuracy = _ci_value(
                        stat_report, "preprocessing", "emoji_raw", "point"
                    )
                    augmentation_accuracy = _ci_value(
                        stat_report, "augmented", "emoji_raw", "point"
                    )
                    row.update({
                        "examples": raw_clean_result["examples"],
                        "paired_clean_accuracy": raw_clean_result["reference_accuracy"],
                        "emoji_raw_accuracy": raw_clean_result["treatment_accuracy"],
                        "raw_minus_clean_pp": raw_clean_result["difference_pp"],
                        "raw_vs_clean_raw_pvalue": raw_clean_result["raw_pvalue"],
                        "emoji_gloss_accuracy": gloss_raw_result["treatment_accuracy"],
                        "gloss_minus_raw_pp": gloss_raw_result["difference_pp"],
                        "gloss_vs_raw_raw_pvalue": gloss_raw_result["raw_pvalue"],
                        "normalization_accuracy": normalization_accuracy,
                        "augmentation_accuracy": augmentation_accuracy,
                        "normalization_minus_augmentation_pp": (
                            100 * (normalization_accuracy - augmentation_accuracy)
                            if normalization_accuracy is not None
                            and augmentation_accuracy is not None else None
                        ),
                        "h3_holm_adjusted_pvalue": h3.get(
                            "holm_adjusted_pvalue"
                        ),
                        "h3_significant_holm": h3.get("significant_holm"),
                        "baseline_combined_accuracy": _ci_value(
                            stat_report, "baseline", combined, "point"
                        ),
                        "hybrid_combined_accuracy": _ci_value(
                            stat_report, "hybrid", combined, "point"
                        ),
                        "status": "ok",
                    })
                emoji_rows.append(row)

            # Accuracy by one versus two bijective emoji replacements.
            raw_groups = intensity_groups.get((dataset, split), {})
            for replacement_count in (1, 2):
                source_ids = raw_groups.get(replacement_count)
                row = {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "dataset": dataset,
                    "split": split,
                    "seed": 42,
                    "variant": "emoji_raw",
                    "intensity_measure": "emoji_replacement_count",
                    "replacement_count": replacement_count,
                    "inference_status": "descriptive_unadjusted",
                }
                if not isinstance(raw, Mapping) or not isinstance(
                    clean_raw, Mapping
                ):
                    row.update({
                        "status": "missing_predictions",
                        "reason": "emoji prediction records are unavailable",
                    })
                elif source_ids is None:
                    audit = data.audit(dataset, split, "emoji_raw")
                    audit_count = (
                        audit.get("emoji_replacements_per_example", {}).get(
                            str(replacement_count)
                        )
                        if isinstance(audit, Mapping) else None
                    )
                    row.update({
                        "status": "missing_dataset_rows",
                        "generation_examples": audit_count,
                        "reason": "Arrow event metadata is unavailable",
                    })
                else:
                    result = paired_test(
                        raw, clean_raw, source_ids=source_ids
                    )
                    row.update({
                        "examples": result["examples"],
                        "accuracy": result["treatment_accuracy"],
                        "clean_accuracy": result["reference_accuracy"],
                        "difference_from_clean_pp": result["difference_pp"],
                        "raw_pvalue": result["raw_pvalue"],
                        "status": "ok",
                    })
                intensity_rows.append(row)

    # Adjust only complete, predeclared families.  Never shrink a family around
    # missing rows.
    apply_control_holm(control_rows)

    return {
        "marker_control_tests": control_rows,
        "placement_diagnostics": placement_rows,
        "generalization_diagnostics": generalization_rows,
        "per_marker_diagnostics": per_marker_rows,
        "emoji_diagnostics": emoji_rows,
        "intensity_diagnostics": intensity_rows,
    }


def build_control_summary(
    rows: Sequence[Mapping[str, Any]],
    family_scope: str = "model_split",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["model"]), str(row["marker_status"]), str(row["control_type"]))
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (model, status, control_type), values in sorted(grouped.items()):
        complete = [row for row in values if row.get("status") == "ok"]
        effects = [float(row["difference_pp"]) for row in complete]
        summaries.append({
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "marker_status": status,
            "control_type": control_type,
            "tests": len(complete),
            "expected_tests": 27,
            "mean_marker_minus_control_pp": mean(effects) if effects else None,
            "min_marker_minus_control_pp": min(effects) if effects else None,
            "max_marker_minus_control_pp": max(effects) if effects else None,
            "positive_tests": sum(effect > 0 for effect in effects),
            "negative_tests": sum(effect < 0 for effect in effects),
            "holm_significant_tests": sum(
                bool(row.get("significant_holm")) for row in complete
            ),
            "holm_significant_positive": sum(
                bool(row.get("significant_holm"))
                and float(row["difference_pp"]) > 0
                for row in complete
            ),
            "holm_significant_negative": sum(
                bool(row.get("significant_holm"))
                and float(row["difference_pp"]) < 0
                for row in complete
            ),
            "family_definition": {
                "model_split": (
                    "36 marker-vs-control tests within each model x evaluation split"
                ),
                "model": "108 marker-vs-control tests within each model",
                "global": "540 marker-vs-control tests across all models and splits",
            }[family_scope],
            "status": "ok" if len(complete) == 27 else "incomplete",
        })
    return summaries


def build_primary_condition_control_rows(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Extract the H2-matched condition as a descriptive, non-primary contrast."""

    selected: list[dict[str, Any]] = []
    for source in rows:
        if (
            source.get("marker_status") != "unseen"
            or source.get("placement") != "hypothesis_suffix"
        ):
            continue
        row = {
            key: value for key, value in source.items()
            if key not in {
                "holm_adjusted_pvalue", "significant_holm", "family_id",
                "family_size", "family_status", "correction", "alpha",
            }
        }
        row.update({
            "contrast_scope": (
                "held-out marker; hypothesis suffix; three frozen marker folds"
            ),
            "relationship_to_primary": (
                "matches H2 input condition but control comparison is exploratory"
            ),
            "claim_status": "descriptive_not_preregistered",
            "pvalue_policy": (
                "raw paired McNemar p-value; no subset-specific significance claim"
            ),
            "broad_family_id": source.get("family_id"),
            "broad_family_holm_adjusted_pvalue": source.get(
                "holm_adjusted_pvalue"
            ),
            "broad_family_significant_holm": source.get("significant_holm"),
        })
        selected.append(row)
    return selected


def build_primary_condition_control_summary(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    grouping_specs = (
        (
            "model_split",
            lambda row: (
                str(row["model"]), str(row["dataset"]), str(row["split"]),
                str(row["control_type"]),
            ),
        ),
        (
            "model_overall",
            lambda row: (
                str(row["model"]), "all", "all", str(row["control_type"]),
            ),
        ),
        (
            "all_models",
            lambda row: (
                "all_models", "all", "all", str(row["control_type"]),
            ),
        ),
    )
    for aggregation_scope, key_function in grouping_specs:
        grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = (
            defaultdict(list)
        )
        for row in rows:
            grouped[key_function(row)].append(row)
        for (model, dataset, split, control_type), values in sorted(
            grouped.items()
        ):
            complete = [row for row in values if row.get("status") == "ok"]
            effects = [float(row["difference_pp"]) for row in complete]
            expected = {
                "model_split": 3,
                "model_overall": 9,
                "all_models": 45,
            }[aggregation_scope]
            summaries.append({
                "aggregation_scope": aggregation_scope,
                "model": model,
                "model_label": (
                    "All five models" if model == "all_models"
                    else MODEL_LABELS.get(model, model)
                ),
                "dataset": dataset,
                "split": split,
                "control_type": control_type,
                "cells": len(complete),
                "expected_cells": expected,
                "aggregation_method": (
                    "unweighted mean over completed model x dataset x fold cells"
                ),
                "mean_marker_minus_control_pp": mean(effects) if effects else None,
                "min_marker_minus_control_pp": min(effects) if effects else None,
                "max_marker_minus_control_pp": max(effects) if effects else None,
                "positive_cells": sum(effect > 0 for effect in effects),
                "negative_cells": sum(effect < 0 for effect in effects),
                "raw_p_below_05": sum(
                    float(row["raw_pvalue"]) <= 0.05 for row in complete
                ),
                "broad_family_holm_significant": sum(
                    bool(row.get("broad_family_significant_holm"))
                    for row in complete
                ),
                "claim_status": "descriptive_not_preregistered",
                "status": (
                    "ok"
                    if len(complete) == expected
                    else "incomplete"
                ),
            })
    return summaries


def _timelm_text(text: str) -> str:
    normalized: list[str] = []
    for token in text.split():
        if len(token) > 1:
            if token[0] == "@" and token.count("@") == 1:
                token = "@user"
            elif token.startswith("http"):
                token = "http"
        normalized.append(token)
    return " ".join(normalized)


def _local_tokenizer_snapshot(checkpoint: str) -> Path | None:
    cache_name = f"models--{checkpoint.replace('/', '--')}"
    cache_roots = [
        Path(os.environ.get("HF_HOME", "")).expanduser() / "hub"
        if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for cache_root in cache_roots:
        if cache_root is None:
            continue
        snapshots = cache_root / cache_name / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates = sorted(
            path for path in snapshots.iterdir()
            if path.is_dir() and (path / "tokenizer_config.json").is_file()
        )
        if candidates:
            return candidates[-1]
    return None


def _load_tokenizer(
    model_config: Mapping[str, Any], mode: str
) -> tuple[Any | None, dict[str, Any]]:
    checkpoint = str(model_config["checkpoint"])
    backend = str(model_config.get("tokenizer_backend", "fast"))
    normalization = str(model_config.get("tokenizer_normalization", "default"))
    metadata: dict[str, Any] = {
        "checkpoint": checkpoint,
        "backend": backend,
        "normalization": normalization,
        "mode": mode,
        "model_weights_required": False,
        "model_weights_loaded": False,
    }
    if mode == "off":
        metadata.update({"status": "not_run", "reason": "tokenizer mode is off"})
        return None, metadata
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        metadata.update({
            "status": "missing_dependency",
            "reason": f"transformers import failed: {error}",
        })
        return None, metadata
    kwargs: dict[str, Any] = {
        "use_fast": backend == "fast",
        "local_files_only": mode == "local",
    }
    if normalization != "default":
        kwargs["normalization"] = normalization == "enabled"
    attempts: list[tuple[str, str]] = [(checkpoint, "configured_checkpoint")]
    if mode == "local":
        snapshot = _local_tokenizer_snapshot(checkpoint)
        if snapshot is not None:
            attempts.append((str(snapshot), "local_snapshot"))
    errors: list[str] = []
    for source, source_kind in attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
            metadata.update({
                "status": "ok",
                "resolved_source": source,
                "resolved_source_kind": source_kind,
                "tokenizer_class": type(tokenizer).__name__,
            })
            return tokenizer, metadata
        except Exception as error:  # tokenizer libraries expose several error types
            errors.append(f"{source_kind}: {type(error).__name__}: {error}")
    metadata.update({
        "status": "missing_tokenizer",
        "reason": (
            "tokenizer is not present in the local Hugging Face cache; "
            "network access was disabled"
            if mode == "local" else " | ".join(errors)
        ),
    })
    return None, metadata


def _token_lengths(
    tokenizer: Any,
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    input_preprocessing: str,
    batch_size: int = 512,
) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(premises), batch_size):
        batch_premises = list(premises[start:start + batch_size])
        batch_hypotheses = list(hypotheses[start:start + batch_size])
        if input_preprocessing == "timelm":
            batch_premises = [_timelm_text(value) for value in batch_premises]
            batch_hypotheses = [_timelm_text(value) for value in batch_hypotheses]
        elif input_preprocessing != "none":
            raise ValueError(
                f"Unknown input preprocessing profile: {input_preprocessing}"
            )
        encoded = tokenizer(
            batch_premises,
            batch_hypotheses,
            truncation=False,
            padding=False,
            add_special_tokens=True,
            return_length=True,
        )
        batch_lengths = encoded.get("length")
        if batch_lengths is None:
            batch_lengths = [len(ids) for ids in encoded["input_ids"]]
        lengths.extend(int(value) for value in batch_lengths)
    return lengths


def _truncation_variants() -> list[str]:
    return [
        "emoji_raw",
        *[
            _marker_variant(fold, "unseen", placement)
            for fold in FOLDS for placement in PLACEMENTS
        ],
    ]


def _missing_truncation_rows(
    model: str,
    *,
    status: str,
    reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, split in FINAL_SPLITS:
        for variant in _truncation_variants():
            for limit in (128, 256):
                rows.append({
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "dataset": dataset,
                    "split": split,
                    "variant": variant,
                    "max_length": limit,
                    "status": status,
                    "reason": reason,
                })
    return rows


def build_truncation_rows(
    data: DataRelease,
    store: PredictionStore,
    config: Mapping[str, Any],
    models: Sequence[str],
    *,
    tokenizer_mode: str,
    missing: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute model-token truncation diagnostics without loading model weights."""

    rows: list[dict[str, Any]] = []
    tokenizer_provenance: list[dict[str, Any]] = []
    if not data.available:
        for model in models:
            rows.extend(_missing_truncation_rows(
                model,
                status="missing_input",
                reason="data release was not supplied",
            ))
        return rows, tokenizer_provenance
    try:
        from datasets import load_from_disk
    except ImportError as error:
        for model in models:
            rows.extend(_missing_truncation_rows(
                model,
                status="missing_dependency",
                reason=f"datasets import failed: {error}",
            ))
        return rows, tokenizer_provenance

    for model in models:
        model_config = config.get("models", {}).get(model)
        if not isinstance(model_config, Mapping):
            rows.extend(_missing_truncation_rows(
                model,
                status="missing_model_config",
                reason="model is absent from experiment config",
            ))
            continue
        tokenizer, tokenizer_record = _load_tokenizer(
            model_config, tokenizer_mode
        )
        tokenizer_record = {"artifact_type": "tokenizer", "model": model, **tokenizer_record}
        tokenizer_provenance.append(tokenizer_record)
        if tokenizer is None:
            status = str(tokenizer_record["status"])
            reason = str(tokenizer_record["reason"])
            rows.extend(_missing_truncation_rows(
                model, status=status, reason=reason
            ))
            missing.append({
                "artifact_type": "tokenizer",
                "model": model,
                "dataset": None,
                "split": None,
                "fold": None,
                "status": status,
                "expected_path": model_config.get("checkpoint"),
                "reason": reason,
            })
            continue
        input_preprocessing = str(
            model_config.get("input_preprocessing", "none")
        )
        for dataset, split in FINAL_SPLITS:
            payload = store.get(model, dataset, split)
            original_path = data.variant_path(dataset, split, "original")
            if original_path is None or not original_path.is_dir():
                for variant in _truncation_variants():
                    for limit in (128, 256):
                        rows.append({
                            "model": model,
                            "model_label": MODEL_LABELS.get(model, model),
                            "dataset": dataset,
                            "split": split,
                            "variant": variant,
                            "max_length": limit,
                            "status": "missing_dataset_rows",
                            "reason": "original Arrow dataset is unavailable",
                        })
                continue
            original = load_from_disk(str(original_path))
            # Frozen originals intentionally preserve the upstream schema and
            # therefore use row position as source_index; transformed variants
            # carry that position explicitly.
            original_sources = (
                [int(value) for value in original["source_index"]]
                if "source_index" in original.column_names
                else list(range(len(original)))
            )
            original_lengths = _token_lengths(
                tokenizer,
                original["premise"],
                original["hypothesis"],
                input_preprocessing=input_preprocessing,
            )
            clean_length_by_source = dict(zip(original_sources, original_lengths))
            for variant in _truncation_variants():
                variant_path = data.variant_path(dataset, split, variant)
                if variant_path is None or not variant_path.is_dir():
                    for limit in (128, 256):
                        rows.append({
                            "model": model,
                            "model_label": MODEL_LABELS.get(model, model),
                            "dataset": dataset,
                            "split": split,
                            "variant": variant,
                            "max_length": limit,
                            "status": "missing_dataset_rows",
                            "reason": "transformed Arrow dataset is unavailable",
                        })
                    continue
                transformed = load_from_disk(str(variant_path))
                sources = [int(value) for value in transformed["source_index"]]
                transformed_lengths = _token_lengths(
                    tokenizer,
                    transformed["premise"],
                    transformed["hypothesis"],
                    input_preprocessing=input_preprocessing,
                )
                try:
                    clean_lengths = [
                        clean_length_by_source[source] for source in sources
                    ]
                except KeyError as error:
                    raise ValueError(
                        f"{variant} references unknown original source {error.args[0]}"
                    ) from error
                record = payload.get(variant) if payload else None
                clean_record = (
                    payload.get(f"clean_{variant}") if payload else None
                )
                if not isinstance(clean_record, Mapping) and payload:
                    clean_record = payload.get("original")
                correct_by_source: dict[int, bool] = {}
                if isinstance(record, Mapping):
                    validate_prediction_record(
                        record,
                        name=variant,
                        expected_examples=len(sources),
                    )
                    correct_by_source = {
                        int(source): int(prediction) == int(label)
                        for source, prediction, label in zip(
                            record["source_indices"],
                            record["predictions"],
                            record["labels"],
                        )
                    }
                for limit in (128, 256):
                    truncated_sources = [
                        source for source, length in zip(
                            sources, transformed_lengths
                        ) if length > limit
                    ]
                    nontruncated_sources = [
                        source for source, length in zip(
                            sources, transformed_lengths
                        ) if length <= limit
                    ]
                    truncated_correct = [
                        correct_by_source[source] for source in truncated_sources
                        if source in correct_by_source
                    ]
                    nontruncated_correct = [
                        correct_by_source[source] for source in nontruncated_sources
                        if source in correct_by_source
                    ]
                    jointly_nontruncated_sources = [
                        source
                        for source, clean_length, transformed_length in zip(
                            sources, clean_lengths, transformed_lengths
                        )
                        if clean_length <= limit and transformed_length <= limit
                    ]
                    paired_nontruncated = (
                        paired_test(
                            record,
                            clean_record,
                            source_ids=jointly_nontruncated_sources,
                        )
                        if jointly_nontruncated_sources
                        and isinstance(record, Mapping)
                        and isinstance(clean_record, Mapping)
                        else None
                    )
                    n = len(sources)
                    rows.append({
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "dataset": dataset,
                        "split": split,
                        "seed": 42,
                        "variant": variant,
                        "checkpoint": model_config.get("checkpoint"),
                        "input_preprocessing": input_preprocessing,
                        "max_length": limit,
                        "examples": n,
                        "clean_exceed_rate": sum(
                            value > limit for value in clean_lengths
                        ) / n,
                        "transformed_exceed_rate": len(truncated_sources) / n,
                        "newly_exceeds_rate": sum(
                            clean <= limit < transformed
                            for clean, transformed in zip(
                                clean_lengths, transformed_lengths
                            )
                        ) / n,
                        "mean_clean_tokens": mean(clean_lengths),
                        "mean_transformed_tokens": mean(transformed_lengths),
                        "mean_token_increase": mean(
                            transformed - clean
                            for clean, transformed in zip(
                                clean_lengths, transformed_lengths
                            )
                        ),
                        "mean_transformed_tokens_removed": mean(
                            max(0, value - limit)
                            for value in transformed_lengths
                        ),
                        "truncated_examples": len(truncated_sources),
                        "truncated_accuracy": (
                            sum(truncated_correct) / len(truncated_correct)
                            if truncated_correct else None
                        ),
                        "nontruncated_examples": len(nontruncated_sources),
                        "nontruncated_accuracy": (
                            sum(nontruncated_correct) / len(nontruncated_correct)
                            if nontruncated_correct else None
                        ),
                        "jointly_nontruncated_examples": len(
                            jointly_nontruncated_sources
                        ),
                        "jointly_nontruncated_transformed_accuracy": (
                            paired_nontruncated["treatment_accuracy"]
                            if paired_nontruncated else None
                        ),
                        "jointly_nontruncated_clean_accuracy": (
                            paired_nontruncated["reference_accuracy"]
                            if paired_nontruncated else None
                        ),
                        "jointly_nontruncated_difference_pp": (
                            paired_nontruncated["difference_pp"]
                            if paired_nontruncated else None
                        ),
                        "jointly_nontruncated_raw_pvalue": (
                            paired_nontruncated["raw_pvalue"]
                            if paired_nontruncated else None
                        ),
                        "jointly_nontruncated_inference_status": (
                            "descriptive_unadjusted"
                            if paired_nontruncated else "missing_predictions"
                        ),
                        "prediction_accuracy_status": (
                            "ok" if correct_by_source else "missing_predictions"
                        ),
                        "status": "ok",
                    })
    return rows, tokenizer_provenance


TABLE_COLUMNS: dict[str, list[str]] = {
    "primary_results": [
        "model", "dataset", "split", "fold", "hypothesis", "treatment_system",
        "treatment_variant", "reference_system", "reference_variant", "examples",
        "treatment_accuracy", "reference_accuracy", "difference_pp",
        "raw_pvalue", "holm_adjusted_pvalue", "significant_holm", "status",
    ],
    "primary_summary": [
        "model", "checkpoint", "hypothesis", "analysis_cells",
        "mean_difference_pp", "min_difference_pp", "max_difference_pp",
        "positive_cells", "negative_cells", "holm_significant_cells", "status",
    ],
    "marker_control_tests": [
        "model", "dataset", "split", "fold", "marker_status", "placement",
        "control_type", "examples", "treatment_accuracy", "reference_accuracy",
        "difference_pp", "raw_pvalue", "holm_adjusted_pvalue",
        "significant_holm", "family_id", "family_size", "family_status", "status",
    ],
    "marker_control_summary": [
        "model", "marker_status", "control_type", "tests",
        "mean_marker_minus_control_pp", "min_marker_minus_control_pp",
        "max_marker_minus_control_pp", "holm_significant_positive",
        "holm_significant_negative", "status",
    ],
    "primary_condition_control_tests": [
        "model", "dataset", "split", "fold", "control_type", "examples",
        "treatment_accuracy", "reference_accuracy", "difference_pp", "raw_pvalue",
        "claim_status", "broad_family_holm_adjusted_pvalue",
        "broad_family_significant_holm", "status",
    ],
    "primary_condition_control_summary": [
        "aggregation_scope", "model", "dataset", "split", "control_type",
        "cells", "mean_marker_minus_control_pp", "min_marker_minus_control_pp",
        "max_marker_minus_control_pp", "positive_cells", "negative_cells",
        "claim_status", "status",
    ],
    "augmentation_transfer_diagnostics": [
        "model", "dataset", "split", "fold", "treatment_system",
        "marker_status", "placement", "examples", "treatment_accuracy",
        "reference_accuracy", "difference_pp", "raw_pvalue",
        "exploratory_broad_holm_adjusted_pvalue",
        "exploratory_broad_significant_holm", "claim_status", "status",
    ],
    "augmentation_transfer_summary": [
        "aggregation_scope", "model", "treatment_system", "marker_status",
        "placement", "cells", "mean_treatment_minus_baseline_pp",
        "min_treatment_minus_baseline_pp", "max_treatment_minus_baseline_pp",
        "positive_cells", "negative_cells", "broad_holm_significant_cells",
        "claim_status", "status",
    ],
    "placement_diagnostics": [
        "model", "dataset", "split", "fold", "marker_status", "placement",
        "examples", "accuracy", "clean_accuracy", "difference_from_clean_pp",
        "difference_from_hypothesis_suffix_pp", "status",
    ],
    "generalization_diagnostics": [
        "model", "dataset", "split", "fold", "placement", "examples",
        "unseen_accuracy", "seen_accuracy", "unseen_minus_seen_pp", "raw_pvalue",
        "status",
    ],
    "per_marker_diagnostics": [
        "model", "dataset", "split", "fold", "marker_status", "placement",
        "marker", "examples", "accuracy", "clean_accuracy",
        "difference_from_clean_pp", "predicted_entailment_count",
        "predicted_entailment_rate", "predicted_neutral_count",
        "predicted_neutral_rate", "predicted_contradiction_count",
        "predicted_contradiction_rate", "raw_pvalue", "status",
    ],
    "marker_token_associations": [
        "model", "model_label", "checkpoint", "dataset", "marker",
        "marker_context", "token_position", "marker_token_count", "token_id",
        "token", "training_examples", "training_tokens",
        "subtoken_training_frequency", "subtoken_entailment_occurrences",
        "subtoken_entailment_pmi", "subtoken_neutral_occurrences",
        "subtoken_neutral_pmi", "subtoken_contradiction_occurrences",
        "subtoken_contradiction_pmi", "phrase_training_occurrences",
        "phrase_entailment_occurrences", "phrase_neutral_occurrences",
        "phrase_contradiction_occurrences", "seen_in_folds",
        "held_out_in_folds", "tokenizer_backend",
        "tokenizer_normalization", "input_preprocessing", "status",
    ],
    "emoji_diagnostics": [
        "model", "dataset", "split", "fold", "examples",
        "paired_clean_accuracy", "emoji_raw_accuracy", "raw_minus_clean_pp",
        "emoji_gloss_accuracy", "gloss_minus_raw_pp", "normalization_accuracy",
        "augmentation_accuracy", "normalization_minus_augmentation_pp",
        "h3_holm_adjusted_pvalue", "h3_significant_holm", "status",
    ],
    "intensity_diagnostics": [
        "model", "dataset", "split", "replacement_count", "examples",
        "accuracy", "clean_accuracy", "difference_from_clean_pp", "raw_pvalue",
        "status",
    ],
    "truncation_diagnostics": [
        "model", "dataset", "split", "variant", "max_length", "examples",
        "clean_exceed_rate", "transformed_exceed_rate", "newly_exceeds_rate",
        "mean_token_increase", "truncated_examples", "truncated_accuracy",
        "nontruncated_accuracy", "jointly_nontruncated_examples",
        "jointly_nontruncated_clean_accuracy",
        "jointly_nontruncated_transformed_accuracy",
        "jointly_nontruncated_difference_pp",
        "jointly_nontruncated_raw_pvalue", "status",
    ],
    "missing_artifacts": [
        "artifact_type", "model", "dataset", "split", "fold", "status", "reason",
    ],
}


def _all_columns(
    rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] | None = None
) -> list[str]:
    observed = {key for row in rows for key in row}
    if not rows and preferred:
        return list(preferred)
    columns = [key for key in (preferred or ()) if key in observed]
    columns.extend(sorted(observed - set(columns)))
    return columns


def _cell_text(value: Any, *, null: str = "") -> str:
    if value is None:
        return null
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Publication tables cannot contain NaN or infinity")
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _display_cell_text(column: str, value: Any, *, null: str = "NA") -> str:
    if value is None:
        return null
    if isinstance(value, bool):
        return "yes" if value else "no"
    if not isinstance(value, float):
        return _cell_text(value, null=null)
    if math.isnan(value) or math.isinf(value):
        raise ValueError("Publication tables cannot contain NaN or infinity")
    if "pvalue" in column:
        return f"{value:.3g}"
    if column.endswith("_pp") or "difference_pp" in column:
        return f"{value:.2f}"
    if any(
        token in column
        for token in ("accuracy", "_rate", "point_estimate", "ci_lower", "ci_upper")
    ):
        return f"{value:.3f}"
    if "tokens" in column or "statistic" in column:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _markdown_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _column_label(column: str) -> str:
    return column.replace("_pp", " (pp)").replace("_", " ").title()


def write_table_bundle(
    output_dir: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    display_columns: Sequence[str] | None = None,
) -> None:
    csv_columns = _all_columns(rows, TABLE_COLUMNS.get(name))
    selected = [
        column for column in (display_columns or TABLE_COLUMNS.get(name, csv_columns))
        if column in csv_columns
    ]
    if not selected:
        selected = csv_columns

    with (output_dir / f"{name}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=csv_columns, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                column: _cell_text(row.get(column)) for column in csv_columns
            })
    (output_dir / f"{name}.json").write_text(
        json.dumps(list(rows), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    headers = [_column_label(column) for column in selected]
    markdown = [
        "| " + " | ".join(_markdown_escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in selected) + " |",
    ]
    for row in rows:
        markdown.append(
            "| " + " | ".join(
                _markdown_escape(_display_cell_text(column, row.get(column)))
                for column in selected
            ) + " |"
        )
    (output_dir / f"{name}.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex = [
        rf"\begin{{tabular}}{{{'l' * len(selected)}}}",
        r"\toprule",
        " & ".join(_latex_escape(value) for value in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        latex.append(
            " & ".join(
                _latex_escape(_display_cell_text(column, row.get(column)))
                for column in selected
            ) + r" \\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / f"{name}.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def _primary_overall_summary(
    primary_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for hypothesis, _ in HYPOTHESES:
        rows = [
            row for row in primary_rows
            if row["hypothesis"] == hypothesis
            and row.get("status") == "ok"
            and (hypothesis != "H1" or row["fold"] == "fold_1")
        ]
        effects = [float(row["difference_pp"]) for row in rows]
        result.append({
            "hypothesis": hypothesis,
            "cells": len(rows),
            "mean_difference_pp": mean(effects) if effects else None,
            "min_difference_pp": min(effects) if effects else None,
            "max_difference_pp": max(effects) if effects else None,
            "holm_significant": sum(
                bool(row.get("significant_holm")) for row in rows
            ),
        })
    return result


def build_report(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    models: Sequence[str],
    family_scope: str,
    seed: int,
    limitations: Sequence[str],
) -> str:
    primary = _primary_overall_summary(tables["primary_results"])
    controls = tables["marker_control_tests"]
    complete_controls = [
        row for row in controls if row.get("status") == "ok"
    ]
    adjusted_controls = [
        row for row in complete_controls
        if row.get("holm_adjusted_pvalue") is not None
    ]
    positive_controls = sum(
        bool(row.get("significant_holm")) and float(row["difference_pp"]) > 0
        for row in adjusted_controls
    )
    negative_controls = sum(
        bool(row.get("significant_holm")) and float(row["difference_pp"]) < 0
        for row in adjusted_controls
    )
    ledger_rows = [
        row for row in tables["primary_condition_control_summary"]
        if row.get("aggregation_scope") == "all_models"
    ]
    ledger_text = " ".join(
        (
            f"Across {row['cells']} cells, marker minus {row['control_type']} "
            f"averaged "
            f"{_display_cell_text('mean_marker_minus_control_pp', row['mean_marker_minus_control_pp'])} "
            "pp (range "
            f"{_display_cell_text('min_marker_minus_control_pp', row['min_marker_minus_control_pp'])} "
            "to "
            f"{_display_cell_text('max_marker_minus_control_pp', row['max_marker_minus_control_pp'])} "
            "pp)."
        )
        for row in ledger_rows
    )
    association_rows = tables["marker_token_associations"]
    complete_associations = [
        row for row in association_rows if row.get("status") == "ok"
    ]
    if association_rows:
        association_text = (
            f"{len(complete_associations)}/{len(association_rows)} rows were "
            "verified from the frozen SNLI/MultiNLI training releases. The "
            "[marker-token association table](marker_token_associations.md) "
            "reports complete-phrase counts, tokenizer components, component "
            "frequency, label-conditioned counts, and token/label PMI. These "
            "statistics are descriptive corpus evidence—not causal effects or "
            "model attributions."
        )
    else:
        association_text = (
            "Frozen-training marker-token association evidence was not supplied; "
            "see [missing artifacts](missing_artifacts.md)."
        )
    missing_rows = tables["missing_artifacts"]
    missing_text = (
        "No planned analysis input or diagnostic is currently marked missing "
        "or not run."
        if not missing_rows
        else (
            f"{len(missing_rows)} planned artifact or diagnostic entries remain "
            "missing/not-run; see [missing artifacts](missing_artifacts.md)."
        )
    )
    diagnostic_lines = []
    for name in (
        "placement_diagnostics", "generalization_diagnostics",
        "augmentation_transfer_diagnostics", "per_marker_diagnostics",
        "emoji_diagnostics",
        "intensity_diagnostics", "truncation_diagnostics",
    ):
        values = tables[name]
        ok = sum(row.get("status") == "ok" for row in values)
        diagnostic_lines.append(
            f"- `{name}`: {ok}/{len(values)} planned rows computed."
        )

    primary_lines = [
        "| Hypothesis | Cells | Mean difference (pp) | Range (pp) | Holm significant |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in primary:
        mean_text = _display_cell_text(
            "mean_difference_pp", row["mean_difference_pp"]
        )
        range_text = (
            f"{_display_cell_text('min_difference_pp', row['min_difference_pp'])} to "
            f"{_display_cell_text('max_difference_pp', row['max_difference_pp'])}"
        )
        primary_lines.append(
            f"| {row['hypothesis']} | {row['cells']} | {mean_text} | "
            f"{range_text} | {row['holm_significant']} |"
        )

    scope_text = {
        "model_split": (
            "one family for each model × evaluation split (36 tests: "
            "3 folds × 2 seen statuses × 3 placements × 2 control types)"
        ),
        "model": "one family for each model (108 tests)",
        "global": "one family spanning every model and evaluation split (540 tests)",
    }[family_scope]
    limitation_lines = "\n".join(f"- {value}" for value in limitations)
    return f"""# Frozen five-model publication analysis

This report was generated entirely from completed seed-{seed} artifacts; it did
not train or submit any model. Models: {", ".join(models)}.

## Primary results

Positive differences favor the treatment named in `primary_results`; negative
differences favor its paired reference. H1 is fold-independent and is
deduplicated in this summary, while the full table preserves every frozen
four-test family.

{os.linesep.join(primary_lines)}

The primary reports retain their original four-hypothesis Holm family within
each model × evaluation split × fold.

## Marker controls

The planned control correction scope is {scope_text}. Holm adjustment is
performed only when every planned row in a family is present; the script never
shrinks an incomplete family. {len(complete_controls)}/{len(controls)} paired
tests were computed. After Holm correction, {positive_controls} significant
tests favored the informal marker and {negative_controls} favored its matched
control.

See [marker-control summary](marker_control_summary.md) and the complete
[paired tests](marker_control_tests.md).

The [primary-condition descriptive contrast](primary_condition_control_summary.md)
restricts the ledger to the held-out-marker hypothesis-suffix condition used by
H2, across all three folds. Its marker-versus-control tests are exploratory and
were not preregistered as a primary hypothesis; raw paired p-values are
descriptive, while the table retains the broader-family Holm result only as
clearly named context.

{ledger_text}

The [augmentation-transfer report](augmentation_transfer_summary.md) is
separate from baseline unseen-minus-seen sensitivity. It reports augmented
minus baseline for seen and held-out hypothesis-suffix markers, plus hybrid
minus baseline on held-out markers. These are exploratory mitigation
contrasts, not preregistered primary hypotheses.

## Frozen-training marker-token associations

{association_text}

## Diagnostics

Diagnostic p-values are paired McNemar p-values but are explicitly descriptive
and unadjusted unless a column names a frozen Holm result.
The truncation report includes the subset where both paired clean and
transformed inputs fit the 128/256-token limit, with clean versus transformed
accuracy, difference, and paired p-value as the reviewer-requested confound
check.

{os.linesep.join(diagnostic_lines)}

## Reproducibility and limitations

Input hashes, the frozen release/config identity, and exact Modal volume source
patterns are in `provenance.json` and `portable_provenance.json`. All tables
also have CSV, JSON, Markdown, and booktabs-compatible LaTeX versions.

{missing_text}

{limitation_lines}
"""


def _portable_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value for key, value in provenance.items()
        if key not in {"input_artifacts"}
    }
    portable_inputs: list[dict[str, Any]] = []
    for item in provenance.get("input_artifacts", []):
        portable = {
            key: value for key, value in item.items()
            if key not in {"local_path"}
        }
        if item.get("local_path"):
            portable["local_filename"] = Path(str(item["local_path"])).name
        portable_inputs.append(portable)
    result["input_artifacts"] = portable_inputs
    portable_tokenizers: list[dict[str, Any]] = []
    for source in provenance.get("tokenizers", []):
        item = dict(source)
        if (
            item.get("resolved_source_kind") == "local_snapshot"
            and item.get("resolved_source")
        ):
            item["local_snapshot_id"] = Path(str(item["resolved_source"])).name
            item.pop("resolved_source", None)
        portable_tokenizers.append(item)
    result["tokenizers"] = portable_tokenizers
    return result


def write_small_snapshot(
    destination: Path,
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    portable_provenance: Mapping[str, Any],
    report: str,
    force: bool,
) -> None:
    """Write a compact, path-safe result snapshot suitable for version control."""

    selected_names = (
        "primary_results",
        "primary_summary",
        "primary_condition_control_tests",
        "primary_condition_control_summary",
        "augmentation_transfer_diagnostics",
        "augmentation_transfer_summary",
        "marker_control_summary",
        "emoji_diagnostics",
        "intensity_diagnostics",
        "placement_diagnostics",
        "generalization_diagnostics",
        "per_marker_diagnostics",
        "marker_token_associations",
        "truncation_diagnostics",
        "missing_artifacts",
    )
    selected = {name: list(tables[name]) for name in selected_names}
    snapshot_report = report.replace(
        "See [marker-control summary](marker_control_summary.md) and the complete\n"
        "[paired tests](marker_control_tests.md).",
        "See [marker-control summary](marker_control_summary.md). The compact "
        "snapshot omits the 540-row broad test table; regenerate the full output "
        "for that table.",
    ).replace(
        "`provenance.json` and `portable_provenance.json`",
        "`provenance.json`",
    )
    truncation_rows = selected["truncation_diagnostics"]
    computed_tokenizers = sorted({
        str(row.get("model_label") or row.get("model"))
        for row in truncation_rows
        if row.get("status") == "ok"
    })
    missing_tokenizers = sorted({
        str(row.get("model_label") or row.get("model"))
        for row in truncation_rows
        if row.get("status") != "ok"
    })
    tokenizer_status = (
        "This snapshot computed truncation diagnostics locally for "
        + ", ".join(computed_tokenizers)
        + "."
        if computed_tokenizers
        else "This snapshot contains no computed tokenizer diagnostics."
    )
    if missing_tokenizers:
        tokenizer_status += (
            " Missing tokenizer diagnostics are recorded for "
            + ", ".join(missing_tokenizers)
            + "."
        )
    missing_status = (
        "No planned artifacts are marked missing or not run."
        if not selected["missing_artifacts"]
        else (
            f"{len(selected['missing_artifacts'])} planned artifact entries are "
            "explicitly marked missing/not-run."
        )
    )
    with _atomic_output_directory(destination.resolve(), force) as staging:
        for name, rows in selected.items():
            write_table_bundle(staging, name, rows)
        (staging / "REPORT.md").write_text(
            snapshot_report, encoding="utf-8"
        )
        (staging / "provenance.json").write_text(
            json.dumps(
                portable_provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        (staging / "snapshot.json").write_text(
            json.dumps({
                "schema_version": SCHEMA_VERSION,
                "tables": selected,
            }, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            f"""# Seed-42 publication analysis snapshot

This compact snapshot is path-safe and contains no prediction arrays,
checkpoint weights, local absolute paths, or newly trained outputs. `provenance.json`
records SHA-256 identities and exact Modal volume paths for the 15 baseline
prediction bundles.

It includes all primary model/split/fold results and summaries; the explicitly
non-preregistered H2-condition marker-control contrast; broad control summaries;
emoji and intensity
diagnostics; placement, baseline sensitivity, augmentation-transfer, and
per-marker reports;
frozen-training marker token frequency and label-PMI evidence;
128/256-token truncation and jointly non-truncated paired-subset checks; and
an explicit missing/not-run ledger.

Regenerate it with:

```bash
python scripts/publication_analysis.py \\
  --statistics-dir <three-model-statistics-directory> \\
  --statistics-dir <roberta-statistics-directory> \\
  --config configs/experiment_v6.json \\
  --data-release <nli-data-release-v10> \\
  --predictions-dir <flattened-baseline-prediction-bundles> \\
  --marker-token-associations audit_evidence/marker_token_associations \\
  --output-dir <full-output-directory> \\
  --snapshot-dir publication_analysis_seed42_snapshot \\
  --tokenizer-mode local \\
  --force
```

`--tokenizer-mode local` never downloads a tokenizer. {tokenizer_status}
{missing_status}
""",
            encoding="utf-8",
        )
        generated = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(staging.iterdir()) if path.is_file()
        ]
        (staging / "artifact_manifest.json").write_text(
            json.dumps({
                "schema_version": SCHEMA_VERSION,
                "files": generated,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _atomic_output_directory(destination: Path, force: bool):
    class _Context:
        def __enter__(self) -> Path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.staging = Path(tempfile.mkdtemp(
                prefix=f".{destination.name}.", dir=destination.parent
            ))
            return self.staging

        def __exit__(self, exc_type, exc, traceback) -> None:
            if exc_type is not None:
                shutil.rmtree(self.staging, ignore_errors=True)
                return
            if destination.exists():
                if not force:
                    shutil.rmtree(self.staging, ignore_errors=True)
                    raise FileExistsError(
                        f"Output directory already exists: {destination}; use --force"
                    )
                if not destination.is_dir():
                    shutil.rmtree(self.staging, ignore_errors=True)
                    raise ValueError(
                        f"Refusing to replace non-directory output: {destination}"
                    )
                shutil.rmtree(destination)
            os.replace(self.staging, destination)

    return _Context()


def run_publication_analysis(
    *,
    statistics_dirs: Sequence[Path],
    config_path: Path,
    data_release_path: Path | None,
    predictions_dir: Path | None,
    marker_token_associations_path: Path | None = None,
    output_dir: Path,
    models: Sequence[str] = DEFAULT_MODELS,
    seed: int = 42,
    control_family_scope: str = "model_split",
    tokenizer_mode: str = "off",
    include_local_provenance: bool = False,
    snapshot_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if seed != 42:
        raise ValueError(
            "Only completed primary-seed 42 artifacts are supported by this workstream"
        )
    if control_family_scope not in {"model_split", "model", "global"}:
        raise ValueError("Invalid marker-control family scope")
    if tokenizer_mode not in {"off", "local", "download"}:
        raise ValueError("tokenizer_mode must be off, local, or download")
    config_path = config_path.resolve()
    config = load_json(config_path)
    unknown = sorted(set(models) - set(config.get("models", {})))
    if unknown:
        raise ValueError(f"Models absent from config: {unknown}")
    config_sha256 = canonical_json_hash(config)
    frozen_seed = config.get("training", {}).get("primary_seed")
    if int(frozen_seed) != seed:
        raise ValueError(
            f"Config primary seed {frozen_seed} does not match requested seed {seed}"
        )

    missing: list[dict[str, Any]] = []
    data = DataRelease(data_release_path)
    if data.release_manifest is not None:
        release_config = data.release_manifest.get("config_sha256")
        if release_config != config_sha256:
            raise ValueError(
                f"Config/data-release hash mismatch: {config_sha256} versus "
                f"{release_config}"
            )
    else:
        missing.append({
            "artifact_type": "data_release",
            "model": None,
            "dataset": None,
            "split": None,
            "fold": None,
            "status": "missing_input",
            "expected_path": str(data_release_path) if data_release_path else None,
            "reason": "data release was not supplied",
        })
    release_provenance = data.provenance()
    release_manifest_sha256 = (
        str(release_provenance["release_manifest_sha256"])
        if release_provenance is not None
        else None
    )
    source_manifest_path = (
        data.root / "source_manifest.json" if data.root is not None else None
    )
    source_manifest_sha256 = (
        file_sha256(source_manifest_path)
        if source_manifest_path is not None and source_manifest_path.is_file()
        else None
    )
    (
        marker_token_association_rows,
        marker_token_association_artifact,
        marker_token_association_provenance,
    ) = load_marker_token_associations(
        marker_token_associations_path,
        config=config,
        config_sha256=config_sha256,
        models=models,
        release_manifest_sha256=release_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        missing=missing,
    )
    statistics, statistics_provenance = discover_statistics(
        statistics_dirs, models
    )
    store = PredictionStore(predictions_dir, models, missing)
    marker_groups = load_marker_groups(data, config, missing)
    intensity_groups = load_emoji_intensity_groups(data, missing)

    primary_rows = build_primary_rows(
        statistics, data, models, missing
    )
    primary_summary = build_primary_summary(primary_rows, config)
    diagnostics = build_prediction_diagnostics(
        store,
        data,
        config,
        statistics,
        models,
        family_scope=control_family_scope,
        marker_groups=marker_groups,
        intensity_groups=intensity_groups,
    )
    control_summary = build_control_summary(
        diagnostics["marker_control_tests"],
        family_scope=control_family_scope,
    )
    truncation_rows, tokenizer_provenance = build_truncation_rows(
        data,
        store,
        config,
        models,
        tokenizer_mode=tokenizer_mode,
        missing=missing,
    )
    diagnostics["truncation_diagnostics"] = truncation_rows

    missing.sort(key=lambda row: tuple(
        "" if row.get(key) is None else str(row.get(key))
        for key in ("artifact_type", "model", "dataset", "split", "fold", "status")
    ))

    primary_condition_rows = build_primary_condition_control_rows(
        diagnostics["marker_control_tests"]
    )
    primary_condition_summary = build_primary_condition_control_summary(
        primary_condition_rows
    )
    augmentation_transfer_rows = build_augmentation_transfer_rows(
        statistics, data, models
    )
    augmentation_transfer_summary = build_augmentation_transfer_summary(
        augmentation_transfer_rows
    )
    tables: dict[str, list[dict[str, Any]]] = {
        "primary_results": primary_rows,
        "primary_summary": primary_summary,
        "marker_control_tests": diagnostics.pop("marker_control_tests"),
        "marker_control_summary": control_summary,
        "primary_condition_control_tests": primary_condition_rows,
        "primary_condition_control_summary": primary_condition_summary,
        "augmentation_transfer_diagnostics": augmentation_transfer_rows,
        "augmentation_transfer_summary": augmentation_transfer_summary,
        "marker_token_associations": marker_token_association_rows,
        **diagnostics,
        "missing_artifacts": missing,
    }
    input_artifacts = [
        {
            "artifact_type": "experiment_config",
            "local_path": str(config_path),
            "size_bytes": config_path.stat().st_size,
            "file_sha256": file_sha256(config_path),
            "canonical_config_sha256": config_sha256,
        },
        *statistics_provenance,
        *store.provenance,
    ]
    if marker_token_association_artifact is not None:
        input_artifacts.append(marker_token_association_artifact)
    if release_provenance is not None:
        input_artifacts.append(release_provenance)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "analysis_seed": seed,
        "models": list(models),
        "control_family_scope": control_family_scope,
        "tokenizer_mode": tokenizer_mode,
        "modal_volume": MODAL_VOLUME,
        "input_artifacts": input_artifacts,
        "tokenizers": tokenizer_provenance,
        "marker_token_associations": marker_token_association_provenance,
        "identity_notes": [
            (
                "Flattened prediction bundles did not include detached prediction "
                "identity manifests; array shape, unique source IDs, labels, "
                "condition names, release counts, Arrow marker metadata, local "
                "SHA-256, and Modal source paths were validated instead."
            ),
            (
                "Statistical JSON files do not embed model/dataset identity; "
                "coordinates are parsed from filenames and every report is hashed."
            ),
        ],
    }
    limitations = [
        (
            "Only seed 42 has a complete five-model matrix in these inputs. "
            "Paired example-level uncertainty is valid, but these outputs do "
            "not estimate variation across training seeds."
        ),
        *provenance["identity_notes"],
    ]
    portable_provenance = _portable_provenance(provenance)
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "seed": seed,
            "models": list(models),
            "control_family_scope": control_family_scope,
            "control_family_definition": (
                "baseline marker versus matching formal/random control; "
                "3 folds x 2 seen statuses x 3 placements x 2 controls"
            ),
            "control_correction": "Holm",
            "control_familywise_alpha": 0.05,
            "tokenizer_mode": tokenizer_mode,
        },
        "provenance": portable_provenance,
        "tables": tables,
        "limitations": limitations,
    }

    output_dir = output_dir.resolve()
    if snapshot_dir is not None and snapshot_dir.resolve() == output_dir:
        raise ValueError("--snapshot-dir must differ from --output-dir")
    report = build_report(
        tables,
        models=models,
        family_scope=control_family_scope,
        seed=seed,
        limitations=limitations,
    )
    with _atomic_output_directory(output_dir, force) as staging:
        for name, rows in tables.items():
            write_table_bundle(staging, name, rows)
        (staging / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging / "provenance.json").write_text(
            json.dumps(
                portable_provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        (staging / "portable_provenance.json").write_text(
            json.dumps(
                portable_provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        if include_local_provenance:
            (staging / "local_provenance.json").write_text(
                json.dumps(
                    provenance,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
        (staging / "REPORT.md").write_text(
            report,
            encoding="utf-8",
        )
        generated = []
        for path in sorted(staging.iterdir()):
            if path.is_file():
                generated.append({
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                })
        (staging / "artifact_manifest.json").write_text(
            json.dumps({
                "schema_version": SCHEMA_VERSION,
                "files": generated,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if snapshot_dir is not None:
        write_small_snapshot(
            snapshot_dir,
            tables=tables,
            portable_provenance=portable_provenance,
            report=report,
            force=force,
        )
    return {
        "output_dir": str(output_dir),
        "snapshot_dir": str(snapshot_dir.resolve()) if snapshot_dir else None,
        "statistics_cells": len(statistics),
        "prediction_bundles": len(store.paths),
        "table_rows": {name: len(rows) for name, rows in tables.items()},
        "missing_artifacts": len(missing),
    }
