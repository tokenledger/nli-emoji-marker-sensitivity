"""Seed-aware reporting for the frozen additional-seed sensitivity design."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping, Sequence

from aggregate_statistics import noninferiority_test


SCHEMA_VERSION = 1
HYPOTHESIS_ORDER = ("H1", "H2", "H3", "H4")
HYPOTHESIS_DIRECTION = {
    "H1": -1,
    "H2": -1,
    "H3": 1,
    "H4": 1,
}
HYPOTHESIS_LABEL = {
    "H1": "emoji raw minus paired clean",
    "H2": "held-out marker minus paired clean",
    "H3": "emoji normalization minus marker augmentation",
    "H4": "hybrid minus baseline on combined inputs",
}
MODEL_ORDER = ("electra", "roberta", "roberta_base", "timelm", "bertweet")
MODEL_LABEL = {
    "electra": "ELECTRA-small",
    "roberta": "RoBERTa-large",
    "roberta_base": "RoBERTa-base",
    "timelm": "TimeLM",
    "bertweet": "BERTweet",
}
SCOPE_BY_MODEL = {
    "electra": "full_five_seed_matrix",
    "roberta": "partial_three_seed_fold1",
    "roberta_base": "partial_three_seed_fold1",
    "timelm": "partial_three_seed_fold1",
    "bertweet": "partial_three_seed_fold1",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _student_t_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("A training-seed interval requires at least two seeds")
    center = mean(values)
    spread = stdev(values)
    if spread == 0:
        return float(center), float(center)
    try:
        from scipy.stats import t
    except ImportError as error:
        raise RuntimeError("scipy is required for seed-aware t intervals") from error
    critical = float(t.ppf((1 + confidence) / 2, df=len(values) - 1))
    half_width = critical * spread / math.sqrt(len(values))
    return float(center - half_width), float(center + half_width)


def _supports_direction(lower: float, upper: float, direction: int) -> bool:
    return upper < 0 if direction < 0 else lower > 0


def _in_expected_direction(value: float, direction: int) -> bool:
    return value < 0 if direction < 0 else value > 0


def _group(
    rows: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(grouped)


def validate_compact_inputs(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported compact seed-sensitivity input schema")
    effects = payload.get("effects")
    clean = payload.get("clean_accuracies")
    artifacts = payload.get("artifacts")
    if not isinstance(effects, list) or not effects:
        raise ValueError("Compact inputs contain no seed effects")
    if not isinstance(clean, list) or not clean:
        raise ValueError("Compact inputs contain no clean accuracies")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Compact inputs contain no source artifacts")
    expected = payload.get("expected_counts", {})
    for key, rows in (
        ("effects", effects),
        ("clean_accuracies", clean),
        ("artifacts", artifacts),
    ):
        if key in expected and int(expected[key]) != len(rows):
            raise ValueError(
                f"Compact input count mismatch for {key}: "
                f"{len(rows)} versus {expected[key]}"
            )

    effect_ids: set[tuple[Any, ...]] = set()
    for row in effects:
        required = {
            "model",
            "dataset",
            "split",
            "fold",
            "hypothesis",
            "seed",
            "treatment_accuracy",
            "reference_accuracy",
            "effect_pp",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"Seed effect row lacks fields: {sorted(missing)}")
        hypothesis = str(row["hypothesis"])
        if hypothesis not in HYPOTHESIS_DIRECTION:
            raise ValueError(f"Unknown hypothesis: {hypothesis}")
        observed = (
            float(row["treatment_accuracy"])
            - float(row["reference_accuracy"])
        ) * 100
        if not math.isclose(
            observed,
            float(row["effect_pp"]),
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("Seed effect does not match its paired accuracies")
        identity = tuple(
            row[key]
            for key in ("model", "dataset", "split", "fold", "hypothesis", "seed")
        )
        if identity in effect_ids:
            raise ValueError(f"Duplicate seed effect: {identity}")
        effect_ids.add(identity)

    clean_ids: set[tuple[Any, ...]] = set()
    for row in clean:
        identity = tuple(
            row[key]
            for key in ("model", "dataset", "split", "fold", "seed", "system")
        )
        if identity in clean_ids:
            raise ValueError(f"Duplicate clean-accuracy row: {identity}")
        clean_ids.add(identity)
        accuracy = float(row["accuracy"])
        if not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid clean accuracy: {accuracy}")

    result_artifacts = [
        row for row in artifacts if row.get("result_sha256") is not None
    ]
    if result_artifacts:
        for field in (
            "config_sha256",
            "data_release_id",
            "source_manifest_sha256",
        ):
            identities = {row.get(field) for row in result_artifacts}
            if None in identities or "" in identities or len(identities) != 1:
                raise ValueError(
                    f"Result artifacts disagree on {field}: "
                    f"{sorted(str(value) for value in identities)}"
                )
        for row in result_artifacts:
            runtime = float(row["runtime_seconds"])
            if not math.isfinite(runtime) or runtime <= 0:
                raise ValueError(
                    f"Invalid observed runtime for {row.get('remote_dir')}"
                )


def build_cell_seed_summary(
    effects: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
) -> list[dict[str, Any]]:
    keys = ("model", "dataset", "split", "fold", "hypothesis")
    rows: list[dict[str, Any]] = []
    for identity, group in _group(effects, keys).items():
        model, dataset, split, fold, hypothesis = identity
        ordered = sorted(group, key=lambda item: int(item["seed"]))
        seeds = [int(item["seed"]) for item in ordered]
        values = [float(item["effect_pp"]) for item in ordered]
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError(f"Invalid seed set for cell {identity}: {seeds}")
        lower, upper = _student_t_interval(values, confidence=confidence)
        direction = HYPOTHESIS_DIRECTION[str(hypothesis)]
        holm_available = [
            item for item in ordered
            if item.get("significant_holm") is not None
        ]
        seed42 = next(
            (float(item["effect_pp"]) for item in ordered if int(item["seed"]) == 42),
            None,
        )
        rows.append({
            "model": model,
            "model_label": MODEL_LABEL[str(model)],
            "dataset": dataset,
            "split": split,
            "fold": fold,
            "hypothesis": hypothesis,
            "contrast": HYPOTHESIS_LABEL[str(hypothesis)],
            "evidence_scope": SCOPE_BY_MODEL[str(model)],
            "seeds": seeds,
            "n_seeds": len(seeds),
            "mean_effect_pp": float(mean(values)),
            "standard_deviation_pp": float(stdev(values)),
            "median_effect_pp": float(median(values)),
            "minimum_effect_pp": float(min(values)),
            "maximum_effect_pp": float(max(values)),
            "ci95_lower_pp": lower,
            "ci95_upper_pp": upper,
            "expected_direction": "negative" if direction < 0 else "positive",
            "expected_direction_seed_count": sum(
                _in_expected_direction(value, direction) for value in values
            ),
            "all_seeds_expected_direction": all(
                _in_expected_direction(value, direction) for value in values
            ),
            "ci95_supports_expected_direction": _supports_direction(
                lower, upper, direction
            ),
            "seed42_effect_pp": seed42,
            "example_holm_tests_available": len(holm_available),
            "example_holm_significant_count": sum(
                bool(item["significant_holm"]) for item in holm_available
            ),
        })
    return sorted(
        rows,
        key=lambda row: (
            MODEL_ORDER.index(str(row["model"])),
            str(row["dataset"]),
            str(row["split"]),
            HYPOTHESIS_ORDER.index(str(row["hypothesis"])),
            str(row["fold"]),
        ),
    )


def build_model_hypothesis_summary(
    cell_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, hypothesis), group in _group(
        cell_rows, ("model", "hypothesis")
    ).items():
        means = [float(item["mean_effect_pp"]) for item in group]
        rows.append({
            "model": model,
            "model_label": MODEL_LABEL[str(model)],
            "hypothesis": hypothesis,
            "contrast": HYPOTHESIS_LABEL[str(hypothesis)],
            "evidence_scope": SCOPE_BY_MODEL[str(model)],
            "cells": len(group),
            "seed_effects": sum(int(item["n_seeds"]) for item in group),
            "mean_of_cell_means_pp": float(mean(means)),
            "median_cell_mean_pp": float(median(means)),
            "minimum_cell_mean_pp": float(min(means)),
            "maximum_cell_mean_pp": float(max(means)),
            "cells_all_seeds_expected_direction": sum(
                bool(item["all_seeds_expected_direction"]) for item in group
            ),
            "cells_ci95_support_expected_direction": sum(
                bool(item["ci95_supports_expected_direction"]) for item in group
            ),
            "example_holm_tests_available": sum(
                int(item["example_holm_tests_available"]) for item in group
            ),
            "example_holm_significant_count": sum(
                int(item["example_holm_significant_count"]) for item in group
            ),
        })
    return sorted(
        rows,
        key=lambda row: (
            MODEL_ORDER.index(str(row["model"])),
            HYPOTHESIS_ORDER.index(str(row["hypothesis"])),
        ),
    )


def build_fold1_cross_model_summary(
    cell_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row for row in cell_rows
        if row["fold"] in {"shared", "fold_1"}
    ]
    rows: list[dict[str, Any]] = []
    for (hypothesis,), group in _group(selected, ("hypothesis",)).items():
        means = [float(item["mean_effect_pp"]) for item in group]
        rows.append({
            "hypothesis": hypothesis,
            "contrast": HYPOTHESIS_LABEL[str(hypothesis)],
            "models": len({str(item["model"]) for item in group}),
            "cells": len(group),
            "mean_of_cell_means_pp": float(mean(means)),
            "median_cell_mean_pp": float(median(means)),
            "minimum_cell_mean_pp": float(min(means)),
            "maximum_cell_mean_pp": float(max(means)),
            "cells_all_seeds_expected_direction": sum(
                bool(item["all_seeds_expected_direction"]) for item in group
            ),
            "cells_ci95_support_expected_direction": sum(
                bool(item["ci95_supports_expected_direction"]) for item in group
            ),
        })
    return sorted(
        rows,
        key=lambda row: HYPOTHESIS_ORDER.index(str(row["hypothesis"])),
    )


def build_seed42_comparison(
    effects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = ("model", "dataset", "split", "fold", "hypothesis")
    rows: list[dict[str, Any]] = []
    for identity, group in _group(effects, keys).items():
        model, dataset, split, fold, hypothesis = identity
        seed42_rows = [row for row in group if int(row["seed"]) == 42]
        new_rows = [row for row in group if int(row["seed"]) != 42]
        if len(seed42_rows) != 1 or not new_rows:
            raise ValueError(f"Cell lacks seed-42/new-seed comparison: {identity}")
        seed42 = float(seed42_rows[0]["effect_pp"])
        new_values = [float(row["effect_pp"]) for row in new_rows]
        new_mean = float(mean(new_values))
        direction = HYPOTHESIS_DIRECTION[str(hypothesis)]
        rows.append({
            "model": model,
            "model_label": MODEL_LABEL[str(model)],
            "dataset": dataset,
            "split": split,
            "fold": fold,
            "hypothesis": hypothesis,
            "seed42_effect_pp": seed42,
            "new_seed_mean_effect_pp": new_mean,
            "seed42_minus_new_mean_pp": seed42 - new_mean,
            "new_seed_minimum_pp": float(min(new_values)),
            "new_seed_maximum_pp": float(max(new_values)),
            "seed42_within_new_seed_range": min(new_values) <= seed42 <= max(new_values),
            "seed42_matches_new_mean_direction": (
                _in_expected_direction(seed42, direction)
                == _in_expected_direction(new_mean, direction)
            ),
        })
    return sorted(
        rows,
        key=lambda row: (
            MODEL_ORDER.index(str(row["model"])),
            str(row["dataset"]),
            str(row["split"]),
            HYPOTHESIS_ORDER.index(str(row["hypothesis"])),
            str(row["fold"]),
        ),
    )


def build_clean_performance_summary(
    clean_rows: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
) -> list[dict[str, Any]]:
    keys = ("model", "dataset", "split", "fold")
    rows: list[dict[str, Any]] = []
    for identity, group in _group(clean_rows, keys).items():
        model, dataset, split, fold = identity
        by_system_seed = {
            (str(item["system"]), int(item["seed"])): float(item["accuracy"])
            for item in group
        }
        seeds = sorted({
            int(item["seed"]) for item in group if item["system"] == "baseline"
        })
        for system in ("augmented", "hybrid", "clean_control"):
            if not all((system, seed) in by_system_seed for seed in seeds):
                continue
            values = [
                (
                    by_system_seed[(system, seed)]
                    - by_system_seed[("baseline", seed)]
                ) * 100
                for seed in seeds
            ]
            lower, upper = _student_t_interval(values, confidence=confidence)
            rows.append({
                "model": model,
                "model_label": MODEL_LABEL[str(model)],
                "dataset": dataset,
                "split": split,
                "fold": fold,
                "system": system,
                "reference_system": "baseline",
                "evidence_scope": SCOPE_BY_MODEL[str(model)],
                "seeds": seeds,
                "n_seeds": len(seeds),
                "mean_difference_pp": float(mean(values)),
                "standard_deviation_pp": float(stdev(values)),
                "ci95_lower_pp": lower,
                "ci95_upper_pp": upper,
                "minimum_difference_pp": float(min(values)),
                "maximum_difference_pp": float(max(values)),
            })
    system_order = {"augmented": 0, "hybrid": 1, "clean_control": 2}
    return sorted(
        rows,
        key=lambda row: (
            MODEL_ORDER.index(str(row["model"])),
            str(row["dataset"]),
            str(row["split"]),
            str(row["fold"]),
            system_order[str(row["system"])],
        ),
    )


def build_electra_clean_noninferiority(
    clean_rows: Sequence[Mapping[str, Any]],
    *,
    margin_pp: float,
    confidence: float,
) -> list[dict[str, Any]]:
    selected = [row for row in clean_rows if row["model"] == "electra"]
    rows: list[dict[str, Any]] = []
    for identity, group in _group(
        selected, ("dataset", "split", "fold")
    ).items():
        dataset, split, fold = identity
        baseline = {
            int(row["seed"]): float(row["accuracy"])
            for row in group if row["system"] == "baseline"
        }
        hybrid = {
            int(row["seed"]): float(row["accuracy"])
            for row in group if row["system"] == "hybrid"
        }
        if set(baseline) != set(hybrid):
            raise ValueError(f"Clean non-inferiority seed mismatch: {identity}")
        result = noninferiority_test(
            hybrid,
            baseline,
            margin_pp=margin_pp,
            confidence=confidence,
        )
        rows.append({
            "model": "electra",
            "model_label": MODEL_LABEL["electra"],
            "dataset": dataset,
            "split": split,
            "fold": fold,
            "system": "hybrid",
            "reference_system": "baseline",
            **result,
        })
    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]), str(row["split"]), str(row["fold"])
        ),
    )


def build_example_test_summary(
    effects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row for row in effects
        if row["model"] == "electra"
        and row.get("significant_holm") is not None
    ]
    rows: list[dict[str, Any]] = []
    for (hypothesis,), group in _group(selected, ("hypothesis",)).items():
        direction = HYPOTHESIS_DIRECTION[str(hypothesis)]
        rows.append({
            "model": "electra",
            "hypothesis": hypothesis,
            "tests": len(group),
            "holm_significant": sum(
                bool(row["significant_holm"]) for row in group
            ),
            "effects_in_expected_direction": sum(
                _in_expected_direction(float(row["effect_pp"]), direction)
                for row in group
            ),
            "holm_significant_and_expected_direction": sum(
                bool(row["significant_holm"])
                and _in_expected_direction(float(row["effect_pp"]), direction)
                for row in group
            ),
        })
    return sorted(
        rows,
        key=lambda row: HYPOTHESIS_ORDER.index(str(row["hypothesis"])),
    )


def build_additional_runtime_summary(
    artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize observed command intervals for additional result-producing calls.

    The Colab status records do not expose provider billing or notebook-idle time.
    Stats-only jobs also lack start/finish intervals, so this table deliberately
    reports command runtime rather than claiming billed GPU-hours.
    """
    selected = [
        row for row in artifacts
        if row.get("source_group") == "additional_seed"
        and row.get("runtime_seconds") is not None
    ]
    rows: list[dict[str, Any]] = []
    for (profile, model), group in _group(
        selected, ("profile", "model")
    ).items():
        values = [float(item["runtime_seconds"]) for item in group]
        rows.append({
            "profile": profile,
            "model": model,
            "model_label": MODEL_LABEL[str(model)],
            "result_producing_calls": len(values),
            "observed_command_hours": float(sum(values) / 3600),
            "mean_minutes_per_call": float(mean(values) / 60),
            "median_minutes_per_call": float(median(values) / 60),
            "minimum_minutes_per_call": float(min(values) / 60),
            "maximum_minutes_per_call": float(max(values) / 60),
        })
    return sorted(
        rows,
        key=lambda row: MODEL_ORDER.index(str(row["model"])),
    )


def build_analysis(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_compact_inputs(payload)
    confidence = float(payload["analysis_contract"]["confidence_level"])
    margin = float(
        payload["analysis_contract"][
            "clean_noninferiority_margin_percentage_points"
        ]
    )
    effects = list(payload["effects"])
    clean = list(payload["clean_accuracies"])
    cell_summary = build_cell_seed_summary(effects, confidence=confidence)
    model_summary = build_model_hypothesis_summary(cell_summary)
    cross_model = build_fold1_cross_model_summary(cell_summary)
    seed42 = build_seed42_comparison(effects)
    clean_summary = build_clean_performance_summary(
        clean, confidence=confidence
    )
    noninferiority = build_electra_clean_noninferiority(
        clean,
        margin_pp=margin,
        confidence=confidence,
    )
    example_tests = build_example_test_summary(effects)
    runtime = build_additional_runtime_summary(payload["artifacts"])
    tables = {
        "seed_effects": effects,
        "cell_seed_summary": cell_summary,
        "model_hypothesis_summary": model_summary,
        "fold1_cross_model_summary": cross_model,
        "seed42_comparison": seed42,
        "clean_performance_summary": clean_summary,
        "electra_clean_noninferiority": noninferiority,
        "electra_example_test_summary": example_tests,
        "additional_runtime_summary": runtime,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_contract": dict(payload["analysis_contract"]),
        "source_provenance": dict(payload["provenance"]),
        "tables": tables,
        "limitations": [
            (
                "Only ELECTRA has five seeds, both datasets, all three marker "
                "folds, and all six systems."
            ),
            (
                "RoBERTa-large, RoBERTa-base, TimeLM, and BERTweet have three "
                "seeds on fold 1 for baseline, augmentation, preprocessing, "
                "and hybrid only; their intervals and direction counts are "
                "sensitivity evidence, not full-matrix inference."
            ),
            (
                "Training-seed t intervals contain only five or three seeds "
                "and should be interpreted with the reported raw seed values."
            ),
            (
                "Formal clean non-inferiority is evaluated only for the "
                "registered ELECTRA hybrid-minus-baseline comparison with the "
                "prespecified 0.5-point margin."
            ),
            (
                "Observed command hours cover result-producing calls only. "
                "They exclude 36 stats-only calls, notebook startup and idle "
                "time, and provider billing, so they are not billed GPU-hours."
            ),
        ],
    }


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    fields = _fieldnames(rows)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        cells = [
            _format_value(row.get(field)).replace("|", "\\|").replace("\n", " ")
            for field in fields
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_table_bundle(
    destination: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    serialized = [dict(row) for row in rows]
    (destination / f"{name}.json").write_text(
        json.dumps(serialized, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = _fieldnames(serialized)
    with (destination / f"{name}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in serialized:
            writer.writerow({
                field: _format_value(row.get(field)) for field in fields
            })
    (destination / f"{name}.md").write_text(
        _markdown_table(serialized),
        encoding="utf-8",
    )


def _pp(value: Any) -> str:
    return f"{float(value):.2f}"


def render_report(analysis: Mapping[str, Any]) -> str:
    tables = analysis["tables"]
    model_rows = tables["model_hypothesis_summary"]
    cell_rows = tables["cell_seed_summary"]
    electra = [row for row in model_rows if row["model"] == "electra"]
    partial = [row for row in model_rows if row["model"] != "electra"]
    seed42 = tables["seed42_comparison"]
    ni = tables["electra_clean_noninferiority"]
    example = tables["electra_example_test_summary"]
    runtime = tables["additional_runtime_summary"]

    direction_cells = sum(
        int(row["cells_all_seeds_expected_direction"]) for row in model_rows
    )
    total_cells = sum(int(row["cells"]) for row in model_rows)
    ci_direction_cells = sum(
        bool(row["ci95_supports_expected_direction"]) for row in cell_rows
    )
    ci_failures = [
        row for row in cell_rows
        if not bool(row["ci95_supports_expected_direction"])
    ]
    ci_failure_lines = "\n".join(
        (
            f"- {row['model_label']} {row['dataset']}/{row['split']} "
            f"{row['hypothesis']} {row['fold']}"
        )
        for row in ci_failures
    )
    seed42_direction_mismatches = sum(
        not bool(row["seed42_matches_new_mean_direction"]) for row in seed42
    )
    seed42_outside = sum(
        not bool(row["seed42_within_new_seed_range"]) for row in seed42
    )
    electra_seed42 = [row for row in seed42 if row["model"] == "electra"]
    partial_seed42 = [row for row in seed42 if row["model"] != "electra"]
    electra_seed42_outside = sum(
        not bool(row["seed42_within_new_seed_range"]) for row in electra_seed42
    )
    partial_seed42_outside = sum(
        not bool(row["seed42_within_new_seed_range"]) for row in partial_seed42
    )
    max_shift = max(
        seed42,
        key=lambda row: abs(float(row["seed42_minus_new_mean_pp"])),
    )
    ni_passes = sum(bool(row["noninferior"]) for row in ni)

    electra_lines = [
        (
            f"| {row['hypothesis']} | {row['cells']} | "
            f"{_pp(row['mean_of_cell_means_pp'])} | "
            f"{_pp(row['minimum_cell_mean_pp'])} to "
            f"{_pp(row['maximum_cell_mean_pp'])} | "
            f"{row['cells_all_seeds_expected_direction']}/{row['cells']} | "
            f"{row['cells_ci95_support_expected_direction']}/{row['cells']} |"
        )
        for row in electra
    ]
    partial_lines = [
        (
            f"| {row['model_label']} | {row['hypothesis']} | "
            f"{_pp(row['mean_of_cell_means_pp'])} | "
            f"{_pp(row['minimum_cell_mean_pp'])} to "
            f"{_pp(row['maximum_cell_mean_pp'])} | "
            f"{row['cells_all_seeds_expected_direction']}/{row['cells']} | "
            f"{row['cells_ci95_support_expected_direction']}/{row['cells']} |"
        )
        for row in partial
    ]
    ni_lines = [
        (
            f"| {row['dataset']} | {row['split']} | {row['fold']} | "
            f"{_pp(row['mean_difference_percentage_points'])} | "
            f"{_pp(row['lower_one_sided_bound_percentage_points'])} | "
            f"{'yes' if row['noninferior'] else 'no'} |"
        )
        for row in ni
    ]
    example_lines = [
        (
            f"| {row['hypothesis']} | {row['holm_significant']}/"
            f"{row['tests']} | "
            f"{row['effects_in_expected_direction']}/{row['tests']} |"
        )
        for row in example
    ]
    runtime_lines = [
        (
            f"| {row['model_label']} | {row['result_producing_calls']} | "
            f"{float(row['observed_command_hours']):.2f} | "
            f"{float(row['median_minutes_per_call']):.1f} |"
        )
        for row in runtime
    ]
    runtime_hours = sum(
        float(row["observed_command_hours"]) for row in runtime
    )
    runtime_calls = sum(
        int(row["result_producing_calls"]) for row in runtime
    )

    limitations = "\n".join(
        f"- {item}" for item in analysis["limitations"]
    )
    return f"""# Additional-seed sensitivity analysis

All 252 frozen additional calls were validated before this analysis. The
compact evidence combines those runs with the verified seed-42 reference.

## Bottom line

The prespecified effect direction holds for **{direction_cells}/{total_cells}**
model × dataset/split × fold cells across every completed seed in each cell.
Seed 42 disagrees in direction with the mean of the newly added seeds in
**{seed42_direction_mismatches}/{len(seed42)}** cells. It falls outside the
new-seed range in **{seed42_outside}/{len(seed42)}** cells
({electra_seed42_outside}/{len(electra_seed42)} full ELECTRA cells and
{partial_seed42_outside}/{len(partial_seed42)} partial-profile cells). This is
a magnitude result—not a direction reversal—and the partial comparison has
only two new seeds. The largest
seed-42 versus new-seed-mean shift is
**{abs(float(max_shift['seed42_minus_new_mean_pp'])):.2f} points**
({max_shift['model_label']}, {max_shift['dataset']}/{max_shift['split']},
{max_shift['hypothesis']}, {max_shift['fold']}).

This supports training-seed robustness for the direction of the paper's H1–H4
effects. Magnitudes remain heterogeneous across datasets, folds, and models.
The two-sided 95% training-seed interval supports the expected direction in
**{ci_direction_cells}/{total_cells}** cells. All 30 full ELECTRA cells pass;
the four intervals crossing zero are limited to three-seed partial cells:

{ci_failure_lines}

## Full five-seed ELECTRA matrix

Means below average the cell-level seed means, so no large SNLI cell is allowed
to dominate the smaller MultiNLI challenge subsets. Cell intervals are
two-sided 95% Student-t intervals across the five training seeds.

| Hypothesis | Cells | Mean effect (pp) | Cell-mean range (pp) | All seeds expected direction | Seed CI supports direction |
| --- | ---: | ---: | ---: | ---: | ---: |
{os.linesep.join(electra_lines)}

The original paired-example Holm family was also available for every ELECTRA
seed and full-matrix cell:

| Hypothesis | Holm significant | Expected direction |
| --- | ---: | ---: |
{os.linesep.join(example_lines)}

## Three-seed fold-1 directional slices

These rows average three dataset/split cell means per model and hypothesis.
They do not establish cross-fold stability or full cross-model seed inference.

| Model | Hypothesis | Mean effect (pp) | Cell-mean range (pp) | All three seeds expected direction | Seed CI supports direction |
| --- | --- | ---: | ---: | ---: | ---: |
{os.linesep.join(partial_lines)}

## Clean-performance non-inferiority

This is the repository's registered hybrid-minus-baseline test: a one-sided
95% Student-t lower bound across the five ELECTRA seeds must exceed the
prespecified **−0.5 percentage-point** margin.

| Dataset | Split | Fold | Mean hybrid−baseline (pp) | Lower bound (pp) | Noninferior |
| --- | --- | --- | ---: | ---: | --- |
{os.linesep.join(ni_lines)}

Non-inferiority is established in **{ni_passes}/{len(ni)}** ELECTRA cells.
All registered ELECTRA cells pass, but this is neither an equivalence result
nor a global five-model claim. The other four profiles omit the
compute-matched clean-control system and are descriptive only for clean
performance.

## Observed additional-run runtime

The completed result-producing commands account for
**{runtime_hours:.2f} observed command-hours** across **{runtime_calls} calls**:

| Model | Result-producing calls | Command-hours | Median minutes/call |
| --- | ---: | ---: | ---: |
{os.linesep.join(runtime_lines)}

These intervals come from the per-command status records. They exclude 36
stats-only calls, Colab notebook startup/idle time, and provider accounting;
they therefore must not be reported as billed GPU-hours.

## Interpretation for the paper

- Replace the “only seed 42” limitation with a five-seed full ELECTRA result
  plus explicitly partial three-seed, fold-1 sensitivity for the other models.
- The direction of H1–H4 is robust across the frozen added seeds; retain the
  strong magnitude-heterogeneity language.
- Report the clean non-inferiority result cell by cell with its exact lower
  bound. Do not generalize the nine-cell ELECTRA result as equivalence across
  models.
- Keep the seed-42 paired-example tests distinct from training-seed intervals.

## Reproducibility and limits

`compact_inputs.json` contains the exact remote result paths, hashes, run
identities, extracted accuracies, and source-bundle identity. The analysis
ignores the stale copied ELECTRA directories present under the TimeLM and
BERTweet destinations by resolving only frozen job identities.

{limitations}
"""


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
                        f"Output directory exists: {destination}; use --force"
                    )
                if not destination.is_dir():
                    shutil.rmtree(self.staging, ignore_errors=True)
                    raise ValueError(
                        f"Refusing to replace non-directory: {destination}"
                    )
                shutil.rmtree(destination)
            os.replace(self.staging, destination)

    return _Context()


def write_analysis(
    payload: Mapping[str, Any],
    destination: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    analysis = build_analysis(payload)
    destination = Path(destination).resolve()
    report = render_report(analysis)
    with _atomic_output_directory(destination, force) as staging:
        (staging / "compact_inputs.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging / "provenance.json").write_text(
            json.dumps(
                analysis["source_provenance"],
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        (staging / "REPORT.md").write_text(report, encoding="utf-8")
        (staging / "README.md").write_text(
            """# Frozen additional-seed sensitivity evidence

This directory contains compact, hash-addressed evidence for the completed
`seed_sensitivity_v1` design. It contains no checkpoints, logits, or prediction
arrays. See `REPORT.md` for findings and `compact_inputs.json` for exact source
paths and extracted accuracies.
""",
            encoding="utf-8",
        )
        for name, rows in analysis["tables"].items():
            write_table_bundle(staging, name, rows)
        generated = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        ]
        (staging / "artifact_manifest.json").write_text(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "files": generated},
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    return {
        "output_dir": str(destination),
        "table_rows": {
            name: len(rows) for name, rows in analysis["tables"].items()
        },
        "noninferiority_passes": sum(
            bool(row["noninferior"])
            for row in analysis["tables"]["electra_clean_noninferiority"]
        ),
        "noninferiority_cells": len(
            analysis["tables"]["electra_clean_noninferiority"]
        ),
    }
