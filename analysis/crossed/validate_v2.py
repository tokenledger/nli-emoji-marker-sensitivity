#!/usr/bin/env python3
"""Independent validator for tokenizer-corrected counterfactual analysis.

This program deliberately does not import :mod:`analyze_v2` and does not use
``validation_v2.json`` as evidence.  It rereads the frozen evaluation matrices,
the 12 validated v1 non-BERTweet prediction cells, and exactly three corrected
BERTweet cells.  It then independently reconstructs the analysis endpoints and
checks the published CSVs, multiplicity adjustments, report headlines, and
output hashes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import binomtest, hypergeom


MODELS = ("electra", "roberta", "roberta_base", "timelm", "bertweet")
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
NON_BERTWEET_MODELS = MODELS[:-1]
SUITES = (
    ("snli", "test"),
    ("multi_nli", "validation_matched"),
    ("multi_nli", "validation_mismatched"),
)
COMPONENT_PAIRS = (
    ("c04_informal_real_talk", "c10_component_real"),
    ("c04_informal_real_talk", "c11_component_talk"),
    ("c05_informal_on_god", "c12_component_on"),
    ("c05_informal_on_god", "c13_component_god"),
    ("c09_informal_no_cap", "c14_component_no"),
    ("c09_informal_no_cap", "c15_component_cap"),
)
ON_GOD_COMPONENTS = ("c12_component_on", "c13_component_god")
FULL_PHRASES = (
    "c04_informal_real_talk",
    "c05_informal_on_god",
    "c09_informal_no_cap",
)
TWO_TOKEN_CONTROLS = (
    "c18_formal_in_fact",
    "c21_random_in_the",
    "c22_random_with_it",
    "c23_random_at_this",
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260726
PRIMARY_FAMILY_SIZE = 132
FIXED_MARGIN_FAMILY_SIZE = 72
TOKENIZER_REVISION = "b349c1243407b0dcffeabb2337497477286e27ab"
TOKENIZER_REPOSITORY = "vinai/bertweet-base"
TOKENIZER_PACKAGE_SHA256 = (
    "934aa09df86b70e9ca86aace92f7dc58d276fc906ae9c6373584fdf49038094f"
)
TOKENIZER_FILE_SHA256 = {
    "bpe.codes": "77712739cd1a7f638e6694b0dd832494e4f66e3d05c709fc6a6a2f988ff9e589",
    "config.json": "7926dbeefbaabac88352b291f86c2363b5d54164b8e67437ea0edae7010257a6",
    "tokenizer.json": "48a8972b321c93163b78d98f40bb410d898d8869d8439b6abb5f8283f545b85d",
    "vocab.txt": "d3f3d56ed440cdb39bd60a76884b67e1061abca462146fdcc751f1ee40ae9ed3",
}
MNLI_WEIGHT_SHA256 = (
    "d5715e15a66ebbfac5724a66d73680186d82d24f7f15ce3063b45878638edeba"
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repository = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-workspace",
        type=Path,
        default=repository / "new_runs" / "phrase_component_counterfactuals",
        help="Frozen v1 workspace holding manifests, matrices, and 12 valid cells.",
    )
    parser.add_argument(
        "--corrected-predictions",
        type=Path,
        default=here / "predictions",
        help="Tree containing exactly three corrected BERTweet artifacts.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=(
            repository
            / "modal_backup_2026-07-26"
            / "volumes"
            / "wnut2026-nli-results-v2"
        ),
        help="Archived seed-42 clean-prediction root used for replay checks.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=here,
        help="Directory containing analyzer v2 CSV, report, and validation outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "independent_validation_v2.json",
        help="Independent validation JSON destination.",
    )
    parser.add_argument(
        "--min-clean-replay-agreement",
        type=float,
        default=0.995,
        help="Independent minimum corrected/archive clean agreement.",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_json_checksum(path: Path) -> str:
    return hashlib.sha256(
        canonical_json(json.loads(path.read_text(encoding="utf-8"))).encode("utf-8")
    ).hexdigest()


def logical_object_checksum(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def check_json_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".logical.sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"Missing logical checksum sidecar: {sidecar}")
    observed = logical_json_checksum(path)
    if sidecar.read_text(encoding="ascii").strip() != observed:
        raise RuntimeError(f"Logical checksum mismatch: {path}")
    return observed


def write_json_with_sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".logical.sha256").write_text(
        logical_json_checksum(path) + "\n",
        encoding="ascii",
    )


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    left = int(np.sum(np.asarray(a, dtype=bool) & ~np.asarray(b, dtype=bool)))
    right = int(np.sum(~np.asarray(a, dtype=bool) & np.asarray(b, dtype=bool)))
    discordant = left + right
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(left, n=discordant, p=0.5, alternative="two-sided").pvalue)
    )
    return p_value, left, right


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: float(p_values[index]))
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def fixed_margin_intersection_tail(
    population: int,
    margins: Sequence[int],
    observed: int,
) -> float:
    """Upper tail of a multiway intersection under fixed uniform margins."""

    ordered = sorted(int(value) for value in margins)
    if not ordered:
        return 1.0
    if any(value < 0 or value > population for value in ordered):
        raise ValueError((population, ordered, observed))
    distribution = np.zeros(population + 1, dtype=np.float64)
    distribution[ordered[0]] = 1.0
    current_maximum = ordered[0]
    for margin in ordered[1:]:
        updated = np.zeros(population + 1, dtype=np.float64)
        for current in range(current_maximum + 1):
            weight = distribution[current]
            if weight == 0.0:
                continue
            lower = max(0, margin - (population - current))
            upper = min(current, margin)
            support = np.arange(lower, upper + 1, dtype=np.int32)
            updated[support] += weight * hypergeom.pmf(
                support,
                population,
                current,
                margin,
            )
        distribution = updated
        current_maximum = min(current_maximum, margin)
    if not math.isclose(float(distribution.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError("Independent fixed-margin distribution failed normalization")
    return float(np.clip(distribution[observed:].sum(), 0.0, 1.0))


def bootstrap_rng(key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}|{key}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def bootstrap_mean_ci(values: np.ndarray, key: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    support, counts = np.unique(values, return_counts=True)
    draws = bootstrap_rng(key).multinomial(
        values.size,
        counts / values.size,
        size=BOOTSTRAP_REPLICATES,
    )
    estimates = (draws @ support) / values.size
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def bootstrap_joint_metrics(
    failure_matrix: np.ndarray,
    key: str,
) -> dict[str, tuple[float, float]]:
    patterns, counts = np.unique(
        np.asarray(failure_matrix, dtype=np.int8).T,
        axis=0,
        return_counts=True,
    )
    population = int(counts.sum())
    draws = bootstrap_rng(key).multinomial(
        population,
        counts / population,
        size=BOOTSTRAP_REPLICATES,
    )
    margins = (draws @ patterns) / population
    joint = (draws @ np.all(patterns == 1, axis=1).astype(float)) / population
    expected = np.prod(margins, axis=1)
    ratio = np.divide(
        joint,
        expected,
        out=np.full_like(joint, np.nan),
        where=expected > 0,
    )
    vectors = {
        "joint_rate": joint,
        "expected_count": population * expected,
        "excess_count": population * (joint - expected),
        "ratio": ratio,
    }
    output: dict[str, tuple[float, float]] = {}
    for name, vector in vectors.items():
        finite = vector[np.isfinite(vector)]
        output[name] = (
            tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))
            if finite.size
            else (math.nan, math.nan)
        )
    return output


def bootstrap_pair_metrics(
    left: np.ndarray,
    right: np.ndarray,
    key: str,
) -> dict[str, tuple[float, float]]:
    patterns, counts = np.unique(
        np.column_stack((left, right)).astype(np.int8),
        axis=0,
        return_counts=True,
    )
    population = int(counts.sum())
    if population == 0:
        return {
            "excess_risk": (math.nan, math.nan),
            "odds_ratio": (math.nan, math.nan),
        }
    draws = bootstrap_rng(key).multinomial(
        population,
        counts / population,
        size=BOOTSTRAP_REPLICATES,
    )
    masks = {
        (a, b): np.all(patterns == (a, b), axis=1).astype(float)
        for a in (0, 1)
        for b in (0, 1)
    }
    n00 = draws @ masks[(0, 0)]
    n01 = draws @ masks[(0, 1)]
    n10 = draws @ masks[(1, 0)]
    n11 = draws @ masks[(1, 1)]
    marginal_left = (n10 + n11) / population
    marginal_right = (n01 + n11) / population
    excess = n11 / population - marginal_left * marginal_right
    zero = (n00 == 0) | (n01 == 0) | (n10 == 0) | (n11 == 0)
    odds = ((n11 + 0.5 * zero) * (n00 + 0.5 * zero)) / (
        (n10 + 0.5 * zero) * (n01 + 0.5 * zero)
    )
    return {
        "excess_risk": tuple(
            float(value) for value in np.quantile(excess, [0.025, 0.975])
        ),
        "odds_ratio": tuple(
            float(value) for value in np.quantile(odds, [0.025, 0.975])
        ),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Required analyzer output is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expect_float(
    observed: str,
    expected: float,
    context: str,
    *,
    tolerance: float = 1e-10,
) -> None:
    try:
        parsed = float(observed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Expected numeric value for {context}: {observed!r}") from exc
    if math.isnan(expected):
        if not math.isnan(parsed):
            raise RuntimeError(f"Expected NaN for {context}, observed {observed!r}")
    elif not math.isclose(parsed, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"Float mismatch for {context}: {parsed!r} != {expected!r}")


def expect_int(observed: str, expected: int, context: str) -> None:
    try:
        parsed = int(observed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Expected integer for {context}: {observed!r}") from exc
    if parsed != expected:
        raise RuntimeError(f"Integer mismatch for {context}: {parsed} != {expected}")


def archive_prediction_path(root: Path, dataset: str, split: str) -> Path:
    return (
        root
        / "publication_results"
        / dataset
        / "seed_42"
        / f"bertweet_baseline_final_{split}"
        / "predictions.json"
    )


def load_registry(source: Path, run: dict[str, Any]) -> list[dict[str, str]]:
    path = source / "CONDITION_REGISTRY.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_prefixes = [f"c{index:02d}_" for index in range(24)]
    if len(rows) != 24 or any(
        not row.get("condition_id", "").startswith(prefix)
        for row, prefix in zip(rows, expected_prefixes)
    ):
        raise RuntimeError("Frozen condition registry is not ordered c00--c23")
    if rows[0]["condition_id"] != "c00_clean":
        raise RuntimeError("Frozen condition registry does not start with c00_clean")
    if sha256_file(path) != run.get("condition_registry_sha256"):
        raise RuntimeError("Condition registry physical hash differs from run manifest")
    return rows


def load_eval_support(
    source: Path,
    registry: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_path = source / "eval_manifest.json"
    check_json_sidecar(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry_hash = sha256_file(source / "CONDITION_REGISTRY.csv")
    if manifest.get("condition_registry_sha256") != registry_hash:
        raise RuntimeError("Evaluation manifest is not bound to the condition registry")
    condition_ids = [row["condition_id"] for row in registry]
    if manifest.get("condition_ids") != condition_ids:
        raise RuntimeError("Evaluation manifest condition order differs from registry")
    suites = {(row["dataset"], row["split"]): row for row in manifest.get("suites", [])}
    if set(suites) != set(SUITES):
        raise RuntimeError("Evaluation manifest suite support differs from frozen suites")
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, split in SUITES:
        suite = suites[(dataset, split)]
        artifact = source / suite["artifact_path"]
        if sha256_file(artifact) != suite.get("artifact_sha256"):
            raise RuntimeError(f"Evaluation physical hash mismatch: {dataset}:{split}")
        source_pairs = int(suite["source_examples"])
        expected_rows = source_pairs * len(condition_ids)
        if int(suite["matrix_rows"]) != expected_rows:
            raise RuntimeError(f"Evaluation manifest row arithmetic failed: {dataset}:{split}")
        labels: list[int] = []
        logical = hashlib.sha256()
        counts: Counter[str] = Counter()
        row_count = 0
        with gzip.open(artifact, "rt", encoding="utf-8") as handle:
            for zero_index, line in enumerate(handle):
                row_count = zero_index + 1
                row = json.loads(line)
                condition_index, source_index = divmod(zero_index, source_pairs)
                if condition_index >= len(condition_ids):
                    raise RuntimeError(f"Extra evaluation row: {dataset}:{split}:{row_count}")
                condition_id = condition_ids[condition_index]
                expected = {
                    "schema_version": 1,
                    "dataset": dataset,
                    "split": split,
                    "condition_id": condition_id,
                    "condition_index": condition_index,
                    "source_index": source_index,
                }
                if any(row.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(
                        f"Evaluation condition-major crossing mismatch: "
                        f"{dataset}:{split}:row{row_count}"
                    )
                label = int(row.get("label", -1))
                if label not in LABEL_NAMES:
                    raise RuntimeError(f"Invalid evaluation label: {dataset}:{split}:row{row_count}")
                if condition_index == 0:
                    labels.append(label)
                elif labels[source_index] != label:
                    raise RuntimeError(
                        f"Evaluation label changed across conditions: {dataset}:{split}:{source_index}"
                    )
                counts[condition_id] += 1
                logical.update(canonical_json(row).encode("utf-8") + b"\n")
        if row_count != expected_rows or any(
            counts[condition] != source_pairs for condition in condition_ids
        ):
            raise RuntimeError(f"Incomplete evaluation crossing: {dataset}:{split}")
        if logical.hexdigest() != suite.get("logical_checksum_sha256"):
            raise RuntimeError(f"Evaluation logical hash mismatch: {dataset}:{split}")
        output[(dataset, split)] = {
            "labels": np.asarray(labels, dtype=np.int8),
            "source_pairs": source_pairs,
            "rows": row_count,
            "physical_sha256": sha256_file(artifact),
            "logical_sha256": logical.hexdigest(),
        }
    return output


def discover_prediction(root: Path, cell_id: str, *, recursive: bool) -> Path:
    pattern = f"{cell_id}__*.jsonl.gz"
    candidates = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one prediction artifact for {cell_id}; "
            f"found {len(candidates)} under {root}"
        )
    return candidates[0]


def validate_legacy_completion(
    completion: dict[str, Any],
    artifact: Path,
    cell: dict[str, Any],
    run: dict[str, Any],
) -> None:
    checks = {
        "status": completion.get("status") == "complete",
        "physical_prediction_hash": completion.get("prediction_sha256")
        == sha256_file(artifact),
        "cell_identity": completion.get("cell_identity_sha256")
        == cell["cell_identity_sha256"],
        "checkpoint": completion.get("checkpoint_package_sha256")
        == cell["trained_checkpoint_package_sha256"],
        "matrix": completion.get("evaluation_logical_checksum_sha256")
        == cell["condition_matrix_checksum_sha256"],
        "inference_code": completion.get("inference_code_sha256")
        == run.get("inference_code_sha256"),
        "runtime": completion.get("runtime_profile") == cell.get("runtime_profile"),
        "manifest": completion.get("run_manifest_logical_sha256")
        == run.get("pre_execution_run_manifest_logical_sha256"),
        "rows": int(completion.get("prediction_rows", -1))
        == int(cell["evaluation_rows"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Legacy completion mismatch for {cell['cell_id']}: {failed}")


def corrected_runtime_is_valid(runtime: Any) -> bool:
    device = runtime.get("device") if isinstance(runtime, dict) else None
    device_valid = device == "cpu" or (
        device == "cuda"
        and isinstance(runtime.get("accelerator_name"), str)
        and "H100" in runtime["accelerator_name"].upper()
        and isinstance(runtime.get("accelerator_count"), int)
        and int(runtime["accelerator_count"]) >= 1
        and isinstance(runtime.get("cuda_version"), str)
        and bool(runtime["cuda_version"])
        and runtime.get("allow_tf32") is False
    )
    return (
        isinstance(runtime, dict)
        and device_valid
        and isinstance(runtime.get("threads"), int)
        and int(runtime["threads"]) > 0
        and int(runtime.get("interop_threads", -1)) == 1
        and isinstance(runtime.get("batch_size"), int)
        and int(runtime["batch_size"]) > 0
        and int(runtime.get("max_length", -1)) == 128
        and runtime.get("input_preprocessing") == "none"
        and runtime.get("tokenizer_backend") == "slow"
        and runtime.get("tokenizer_normalization") == "disabled"
        and runtime.get("logit_dtype") == "float32"
        and int(runtime.get("seed", -1)) == 42
        and runtime.get("deterministic_algorithms") is True
        and isinstance(runtime.get("torch_version"), str)
        and bool(runtime["torch_version"])
    )


def validate_corrected_completion(
    completion: dict[str, Any],
    artifact: Path,
    cell: dict[str, Any],
    run: dict[str, Any],
    registry_hash: str,
    correction_code_hash: str,
) -> None:
    tokenizer = completion.get("tokenizer_identity")
    checkpoint = completion.get("checkpoint_identity")
    matrix = completion.get("matrix_validation")
    runtime = completion.get("runtime_profile")
    source_manifest_identity = completion.get("source_manifest_identity")
    expected_weight_sha = MNLI_WEIGHT_SHA256 if cell["dataset"] == "multi_nli" else None
    correction_payload = {
        "schema_version": 2,
        "cell_id": cell["cell_id"],
        "source_cell_identity_sha256": cell["cell_identity_sha256"],
        "checkpoint_package_sha256": cell["trained_checkpoint_package_sha256"],
        "checkpoint_weight_sha256": expected_weight_sha,
        "evaluation_logical_checksum_sha256": cell["condition_matrix_checksum_sha256"],
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_package_sha256": TOKENIZER_PACKAGE_SHA256,
        "inference_code_sha256": correction_code_hash,
        "runtime_profile": runtime,
    }
    expected_correction_identity = hashlib.sha256(
        canonical_json(correction_payload).encode("utf-8")
    ).hexdigest()
    checks = {
        "schema": int(completion.get("schema_version", -1)) == 2,
        "status": completion.get("status") == "complete",
        "correction": completion.get("correction")
        == "bertweet_base_tokenizer_roundtrip_fix",
        "cell": completion.get("cell_id") == cell["cell_id"],
        "physical_prediction_hash": completion.get("prediction_sha256")
        == sha256_file(artifact),
        "source_cell_identity": completion.get("source_cell_identity_sha256")
        == cell["cell_identity_sha256"],
        "checkpoint": completion.get("checkpoint_package_sha256")
        == cell["trained_checkpoint_package_sha256"],
        "checkpoint_nested": isinstance(checkpoint, dict)
        and checkpoint.get("package_sha256") == cell["trained_checkpoint_package_sha256"],
        "checkpoint_weight": isinstance(checkpoint, dict)
        and is_sha256(checkpoint.get("weight_sha256"))
        and (
            (
                cell["dataset"] == "multi_nli"
                and checkpoint.get("weight_source") == "verified_remote_override"
                and checkpoint.get("weight_sha256") == MNLI_WEIGHT_SHA256
            )
            or (
                cell["dataset"] == "snli"
                and checkpoint.get("weight_source") == "checkpoint_package"
                and checkpoint.get("weight_artifact") == "model.safetensors"
            )
        ),
        "matrix": completion.get("evaluation_logical_checksum_sha256")
        == cell["condition_matrix_checksum_sha256"],
        "registry": completion.get("condition_registry_sha256") == registry_hash,
        "inference_code": completion.get("inference_code_sha256") == correction_code_hash,
        "manifest": completion.get("source_run_manifest_logical_sha256")
        == run.get("pre_execution_run_manifest_logical_sha256"),
        "source_manifest_identity": isinstance(source_manifest_identity, dict)
        and source_manifest_identity.get("approved_pre_execution_logical_sha256")
        == run.get("pre_execution_run_manifest_logical_sha256")
        and source_manifest_identity.get("observed_post_execution_logical_sha256")
        == logical_object_checksum(run),
        "tokenizer": isinstance(tokenizer, dict)
        and tokenizer.get("repository") == TOKENIZER_REPOSITORY
        and tokenizer.get("revision") == TOKENIZER_REVISION
        and tokenizer.get("package_sha256") == TOKENIZER_PACKAGE_SHA256
        and tokenizer.get("file_sha256") == TOKENIZER_FILE_SHA256
        and tokenizer.get("backend") == "slow"
        and tokenizer.get("normalization") is False,
        "runtime": corrected_runtime_is_valid(runtime),
        "matrix_validation": isinstance(matrix, dict)
        and matrix.get("passed") is True
        and int(matrix.get("condition_count", -1)) == 24
        and int(matrix.get("source_pairs", -1)) == int(cell["source_pairs"])
        and int(matrix.get("matrix_rows", -1)) == int(cell["evaluation_rows"]),
        "rows": int(completion.get("prediction_rows", -1))
        == int(cell["evaluation_rows"]),
        "logical_prediction_hash_shape": is_sha256(
            completion.get("prediction_logical_checksum_sha256")
        ),
        "correction_identity": completion.get("correction_identity_sha256")
        == expected_correction_identity,
        "prediction_filename": artifact.name
        == f"{cell['cell_id']}__tokenizerfix__{expected_correction_identity[:16]}.jsonl.gz"
        and completion.get("prediction_artifact") == artifact.name,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Corrected completion mismatch for {cell['cell_id']}: {failed}")


def validate_clean_replay(
    loaded: dict[str, Any],
    completion: dict[str, Any],
    archive_root: Path,
    cell: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    archive = archive_prediction_path(archive_root, cell["dataset"], cell["split"])
    payload = json.loads(archive.read_text(encoding="utf-8"))
    original = payload.get("original")
    if not isinstance(original, dict):
        raise RuntimeError(f"Archive lacks original block: {archive}")
    archived_predictions = np.asarray(original.get("predictions"), dtype=np.int8)
    archived_labels = np.asarray(original.get("labels"), dtype=np.int8)
    archived_sources = np.asarray(original.get("source_indices"), dtype=np.int64)
    labels = loaded["labels"]
    clean = loaded["predictions"]["c00_clean"]
    if (
        archived_predictions.shape != clean.shape
        or not np.array_equal(archived_labels, labels)
        or not np.array_equal(archived_sources, np.arange(len(labels)))
    ):
        raise RuntimeError(f"Archive support mismatch: {cell['cell_id']}")
    matched = int(np.sum(archived_predictions == clean))
    agreement = matched / len(labels)
    replay_accuracy = float(np.mean(clean == labels))
    archive_accuracy = float(np.mean(archived_predictions == labels))
    physical = sha256_file(archive)
    logical = logical_object_checksum(original)
    claim = completion.get("clean_replay")
    if not isinstance(claim, dict):
        raise RuntimeError(f"Corrected completion lacks clean replay: {cell['cell_id']}")
    claim_checks = {
        "physical_archive_hash": claim.get("reference_prediction_sha256") == physical,
        "logical_archive_hash": claim.get("reference_original_logical_sha256") == logical,
        "matched": int(claim.get("matched", -1)) == matched,
        "total": int(claim.get("total", -1)) == len(labels),
        "agreement": math.isclose(
            float(claim.get("agreement", -1.0)), agreement, abs_tol=1e-15
        ),
        "replay_accuracy": math.isclose(
            float(claim.get("replay_accuracy", -1.0)), replay_accuracy, abs_tol=1e-15
        ),
        "reference_accuracy": math.isclose(
            float(claim.get("reference_accuracy", -1.0)), archive_accuracy, abs_tol=1e-15
        ),
        "passed": claim.get("passed") is True,
    }
    failed = [name for name, passed in claim_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Clean replay sidecar mismatch for {cell['cell_id']}: {failed}")
    if agreement < threshold:
        raise RuntimeError(
            f"Independent clean replay failed for {cell['cell_id']}: "
            f"{agreement:.8f} < {threshold:.8f}"
        )
    return {
        "cell_id": cell["cell_id"],
        "archive_prediction_sha256": physical,
        "archive_original_logical_sha256": logical,
        "matched": matched,
        "total": len(labels),
        "agreement": agreement,
        "required_agreement": threshold,
        "replay_accuracy": replay_accuracy,
        "archive_accuracy": archive_accuracy,
        "accuracy_absolute_deviation": abs(replay_accuracy - archive_accuracy),
        "passed": True,
    }


def load_prediction(
    artifact: Path,
    completion: dict[str, Any],
    cell: dict[str, Any],
    registry: list[dict[str, str]],
    eval_support: dict[str, Any],
) -> dict[str, Any]:
    condition_ids = [row["condition_id"] for row in registry]
    registry_by_id = {row["condition_id"]: row for row in registry}
    source_pairs = int(cell["source_pairs"])
    expected_rows = source_pairs * len(condition_ids)
    if source_pairs != int(eval_support["source_pairs"]):
        raise RuntimeError(f"Prediction/evaluation source support mismatch: {cell['cell_id']}")
    predictions = {
        condition: np.empty(source_pairs, dtype=np.int8) for condition in condition_ids
    }
    logical = hashlib.sha256()
    row_count = 0
    with gzip.open(artifact, "rt", encoding="utf-8") as handle:
        for zero_index, line in enumerate(handle):
            row_count = zero_index + 1
            condition_index, source_index = divmod(zero_index, source_pairs)
            if condition_index >= len(condition_ids):
                raise RuntimeError(f"Extra prediction row: {cell['cell_id']}:{row_count}")
            condition = condition_ids[condition_index]
            row = json.loads(line)
            logits = row.get("logits")
            expected = {
                "schema_version": 1,
                "cell_id": cell["cell_id"],
                "model": cell["model"],
                "dataset": cell["dataset"],
                "split": cell["split"],
                "source_index": source_index,
                "condition_id": condition,
                "condition_type": registry_by_id[condition]["condition_type"],
                "checkpoint_package_sha256": cell["trained_checkpoint_package_sha256"],
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise RuntimeError(
                    f"Prediction condition-major/identity mismatch: "
                    f"{cell['cell_id']}:row{row_count}"
                )
            label = int(row.get("label", -1))
            prediction = int(row.get("prediction", -1))
            if (
                label != int(eval_support["labels"][source_index])
                or prediction not in LABEL_NAMES
                or not isinstance(logits, list)
                or len(logits) != 3
                or not all(math.isfinite(float(value)) for value in logits)
                or max(range(3), key=lambda index: float(logits[index])) != prediction
            ):
                raise RuntimeError(f"Prediction schema/value mismatch: {cell['cell_id']}:row{row_count}")
            predictions[condition][source_index] = prediction
            logical.update(canonical_json(row).encode("utf-8") + b"\n")
    if row_count != expected_rows or row_count != int(cell["evaluation_rows"]):
        raise RuntimeError(f"Prediction row-count mismatch: {cell['cell_id']}")
    if logical.hexdigest() != completion.get("prediction_logical_checksum_sha256"):
        raise RuntimeError(f"Prediction logical checksum mismatch: {cell['cell_id']}")
    return {
        "labels": eval_support["labels"].copy(),
        "predictions": predictions,
        "rows": row_count,
        "physical_sha256": sha256_file(artifact),
        "logical_sha256": logical.hexdigest(),
    }


def load_inputs(args: argparse.Namespace) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    list[dict[str, str]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    source = args.source_workspace.resolve()
    run_path = source / "run_manifest.json"
    check_json_sidecar(run_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    approved = run.get("pre_execution_run_manifest_logical_sha256")
    if not is_sha256(approved) or run.get("approval_granted") is not True:
        raise RuntimeError("Frozen run manifest is not bound to approved pre-execution state")
    registry = load_registry(source, run)
    eval_support = load_eval_support(source, registry)
    cells = run.get("cells")
    if not isinstance(cells, list) or len(cells) != 15:
        raise RuntimeError("Frozen run manifest must contain exactly 15 cells")
    expected_keys = {
        (model, dataset, split)
        for model in MODELS
        for dataset, split in SUITES
    }
    if {
        (cell.get("model"), cell.get("dataset"), cell.get("split")) for cell in cells
    } != expected_keys:
        raise RuntimeError("Frozen run manifest is not the required 5x3 cell grid")
    correction_code = Path(__file__).resolve().with_name("run_bertweet_cpu.py")
    if not correction_code.is_file():
        raise RuntimeError(f"Correction runner is missing: {correction_code}")
    correction_code_hash = sha256_file(correction_code)
    registry_hash = sha256_file(source / "CONDITION_REGISTRY.csv")
    data: dict[tuple[str, str, str], dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    for cell in cells:
        corrected = cell["model"] == "bertweet"
        root = (
            args.corrected_predictions.resolve()
            if corrected
            else source / "predictions"
        )
        artifact = discover_prediction(root, cell["cell_id"], recursive=corrected)
        completion_path = artifact.with_suffix(artifact.suffix + ".complete.json")
        check_json_sidecar(completion_path)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if corrected:
            validate_corrected_completion(
                completion,
                artifact,
                cell,
                run,
                registry_hash,
                correction_code_hash,
            )
        else:
            validate_legacy_completion(completion, artifact, cell, run)
        loaded = load_prediction(
            artifact,
            completion,
            cell,
            registry,
            eval_support[(cell["dataset"], cell["split"])],
        )
        if corrected:
            replay.append(
                validate_clean_replay(
                    loaded,
                    completion,
                    args.archive_root.resolve(),
                    cell,
                    args.min_clean_replay_agreement,
                )
            )
        key = (cell["model"], cell["dataset"], cell["split"])
        data[key] = loaded
        audit.append(
            {
                "cell_id": cell["cell_id"],
                "source": "corrected_v2" if corrected else "validated_v1",
                "prediction_artifact": artifact.name,
                "prediction_sha256": loaded["physical_sha256"],
                "prediction_logical_sha256": loaded["logical_sha256"],
                "completion_sha256": sha256_file(completion_path),
                "completion_logical_sha256": logical_json_checksum(completion_path),
                "prediction_rows": loaded["rows"],
            }
        )
    if len(data) != 15 or len(replay) != 3:
        raise RuntimeError("Independent input loader did not obtain 12 v1 + 3 corrected cells")
    for dataset, split in SUITES:
        labels = data[(MODELS[0], dataset, split)]["labels"]
        if any(
            not np.array_equal(labels, data[(model, dataset, split)]["labels"])
            for model in MODELS[1:]
        ):
            raise RuntimeError(f"Five-model label support mismatch: {dataset}:{split}")
    return data, registry, run, audit, eval_support, replay


def validate_condition_effects(
    analysis_dir: Path,
    data: dict[tuple[str, str, str], dict[str, Any]],
    registry: list[dict[str, str]],
) -> dict[str, Any]:
    path = analysis_dir / "condition_effects_v2.csv"
    rows = read_csv(path)
    expected_keys = [
        (model, dataset, split, condition["condition_id"])
        for model in MODELS
        for dataset, split in SUITES
        for condition in registry
    ]
    observed_keys = [
        (row["model"], row["dataset"], row["split"], row["condition_id"])
        for row in rows
    ]
    if observed_keys != expected_keys:
        raise RuntimeError("Condition-effect CSV row support/order is not the frozen 5x3x24 grid")
    registry_by_id = {row["condition_id"]: row for row in registry}
    on_god_changes: list[float] = []
    for row in rows:
        context = ":".join(
            (row["model"], row["dataset"], row["split"], row["condition_id"])
        )
        cell = data[(row["model"], row["dataset"], row["split"])]
        labels = cell["labels"]
        predictions = cell["predictions"]
        clean = predictions["c00_clean"]
        transformed = predictions[row["condition_id"]]
        clean_correct = clean == labels
        correct = transformed == labels
        delta = (correct.astype(np.int8) - clean_correct.astype(np.int8)) * 100.0
        failures = clean_correct & ~correct
        clean_count = int(clean_correct.sum())
        failure_count = int(failures.sum())
        destination_counts = [
            int(np.sum(failures & (transformed == destination)))
            for destination in range(3)
        ]
        bootstrap_identity = "|".join(
            (row["model"], row["dataset"], row["split"], row["condition_id"])
        )
        delta_low, delta_high = bootstrap_mean_ci(
            delta,
            f"effect|{bootstrap_identity}",
        )
        failure_low, failure_high = bootstrap_mean_ci(
            failures[clean_correct].astype(float),
            f"failure_rate|{bootstrap_identity}",
        )
        if (
            row["status"] != "complete"
            or row["condition_type"]
            != registry_by_id[row["condition_id"]]["condition_type"]
        ):
            raise RuntimeError(f"Condition-effect identity/status mismatch: {context}")
        integers = {
            "denominator": len(labels),
            "clean_correct_denominator": clean_count,
            "failures": failure_count,
            "destination_entailment_count": destination_counts[0],
            "destination_neutral_count": destination_counts[1],
            "destination_contradiction_count": destination_counts[2],
        }
        for field, expected in integers.items():
            expect_int(row[field], expected, f"{context}:{field}")
        floats = {
            "accuracy": float(correct.mean()),
            "clean_accuracy": float(clean_correct.mean()),
            "transformed_minus_clean_pp": float(delta.mean()),
            "bootstrap_ci_low_pp": delta_low,
            "bootstrap_ci_high_pp": delta_high,
            "failure_rate": failure_count / clean_count if clean_count else math.nan,
            "failure_rate_ci_low": failure_low,
            "failure_rate_ci_high": failure_high,
            "destination_entailment_rate_clean_correct": (
                destination_counts[0] / clean_count if clean_count else math.nan
            ),
            "destination_neutral_rate_clean_correct": (
                destination_counts[1] / clean_count if clean_count else math.nan
            ),
            "destination_contradiction_rate_clean_correct": (
                destination_counts[2] / clean_count if clean_count else math.nan
            ),
            "destination_entailment_rate_failures": (
                destination_counts[0] / failure_count if failure_count else math.nan
            ),
            "destination_neutral_rate_failures": (
                destination_counts[1] / failure_count if failure_count else math.nan
            ),
            "destination_contradiction_rate_failures": (
                destination_counts[2] / failure_count if failure_count else math.nan
            ),
        }
        for field, expected in floats.items():
            expect_float(row[field], expected, f"{context}:{field}")
        if row["condition_id"] == "c05_informal_on_god":
            on_god_changes.append(float(delta.mean()))
    return {
        "path": path,
        "rows": len(rows),
        "sha256": sha256_file(path),
        "on_god_accuracy_change_min_pp": min(on_god_changes),
        "on_god_accuracy_change_max_pp": max(on_god_changes),
    }


def contrast_event_arrays(
    row: dict[str, str],
    data: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = data[(row["model"], row["dataset"], row["split"])]
    labels = cell["labels"]
    predictions = cell["predictions"]
    clean_correct_entailment = (predictions["c00_clean"] == labels) & (labels == 0)
    if row["endpoint"] == "clean_correct_entailment_failure":
        eligible = clean_correct_entailment
        full = predictions[row["full_condition"]] != labels
        comparator = predictions[row["comparator_condition"]] != labels
    elif row["endpoint"] == "clean_correct_entailment_to_neutral":
        eligible = clean_correct_entailment
        full = predictions[row["full_condition"]] == 1
        comparator = predictions[row["comparator_condition"]] == 1
    elif row["endpoint"] == "unconditional_accuracy":
        eligible = np.ones(len(labels), dtype=bool)
        full = predictions[row["full_condition"]] == labels
        comparator = predictions[row["comparator_condition"]] == labels
    else:
        raise RuntimeError(f"Unexpected paired-contrast endpoint: {row['endpoint']}")
    return eligible, full[eligible], comparator[eligible]


def validate_primary_contrasts(
    analysis_dir: Path,
    data: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = analysis_dir / "phrase_component_contrasts_v2.csv"
    rows = read_csv(path)
    expected_keys: list[tuple[str, ...]] = []
    for model in MODELS:
        for dataset, split in SUITES:
            for full, comparator in COMPONENT_PAIRS:
                expected_keys.append(
                    (
                        "primary_phrase_component_failure",
                        "true",
                        "clean_correct_entailment_failure",
                        model,
                        dataset,
                        split,
                        full,
                        comparator,
                    )
                )
                expected_keys.append(
                    (
                        "secondary_phrase_component_accuracy",
                        "false",
                        "unconditional_accuracy",
                        model,
                        dataset,
                        split,
                        full,
                        comparator,
                    )
                )
            expected_keys.extend(
                (
                    "primary_on_god_neutral_destination",
                    "true",
                    "clean_correct_entailment_to_neutral",
                    model,
                    dataset,
                    split,
                    "c05_informal_on_god",
                    comparator,
                )
                for comparator in ON_GOD_COMPONENTS
            )
            for full in FULL_PHRASES:
                for comparator in TWO_TOKEN_CONTROLS:
                    for endpoint in (
                        "clean_correct_entailment_failure",
                        "unconditional_accuracy",
                    ):
                        expected_keys.append(
                            (
                                "secondary_full_phrase_same_length_control",
                                "false",
                                endpoint,
                                model,
                                dataset,
                                split,
                                full,
                                comparator,
                            )
                        )
    observed_keys = [
        (
            row["family"],
            row["primary"],
            row["endpoint"],
            row["model"],
            row["dataset"],
            row["split"],
            row["full_condition"],
            row["comparator_condition"],
        )
        for row in rows
    ]
    if observed_keys != expected_keys or len(rows) != 570:
        raise RuntimeError("Paired-contrast CSV support/order mismatch")
    records: list[dict[str, Any]] = []
    secondary_count = 0
    for row in rows:
        context = "|".join(
            (
                row["family"],
                row["endpoint"],
                row["model"],
                row["dataset"],
                row["split"],
                row["full_condition"],
                row["comparator_condition"],
            )
        )
        if row["status"] != "complete":
            raise RuntimeError(f"Paired-contrast status mismatch: {context}")
        eligible, full, comparator = contrast_event_arrays(row, data)
        differences = (full.astype(np.int8) - comparator.astype(np.int8)) * 100.0
        p_value, full_only, comparator_only = exact_mcnemar(full, comparator)
        low, high = bootstrap_mean_ci(differences, context)
        integer_expectations = {
            "eligible_denominator": int(eligible.sum()),
            "full_events": int(full.sum()),
            "comparator_events": int(comparator.sum()),
            "discordant_full_only": full_only,
            "discordant_comparator_only": comparator_only,
        }
        for field, expected in integer_expectations.items():
            expect_int(row[field], expected, f"{context}:{field}")
        float_expectations = {
            "effect_pp": float(differences.mean()),
            "bootstrap_ci_low_pp": low,
            "bootstrap_ci_high_pp": high,
            "mcnemar_p_raw": p_value,
        }
        for field, expected in float_expectations.items():
            expect_float(row[field], expected, f"{context}:{field}")
        if row["primary"] == "true":
            records.append(
                {
                    "row": row,
                    "p_raw": p_value,
                    "effect_pp": float(differences.mean()),
                    "family": row["family"],
                }
            )
        else:
            secondary_count += 1
            if row["holm_p"] != "":
                raise RuntimeError(f"Secondary paired contrast has Holm value: {context}")
    if len(records) != 120 or secondary_count != 450:
        raise RuntimeError(
            f"Paired-contrast primary/secondary count mismatch: "
            f"{len(records)}/{secondary_count}"
        )
    return {
        "path": path,
        "rows": len(rows),
        "sha256": sha256_file(path),
        "secondary_tests": secondary_count,
        "phrase_component_tests": sum(
            record["family"] == "primary_phrase_component_failure" for record in records
        ),
        "on_god_destination_tests": sum(
            record["family"] == "primary_on_god_neutral_destination" for record in records
        ),
    }, records


def validate_destination_rows(
    analysis_dir: Path,
    data: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    path = analysis_dir / "on_god_destination_v2.csv"
    rows = read_csv(path)
    expected_keys = [
        (model, dataset, split)
        for model in MODELS
        for dataset, split in SUITES
    ]
    if [(row["model"], row["dataset"], row["split"]) for row in rows] != expected_keys:
        raise RuntimeError("On-god destination CSV support/order mismatch")
    for row in rows:
        context = f"{row['model']}:{row['dataset']}:{row['split']}"
        cell = data[(row["model"], row["dataset"], row["split"])]
        labels = cell["labels"]
        predictions = cell["predictions"]
        eligible = (predictions["c00_clean"] == labels) & (labels == 0)
        transformed = predictions["c05_informal_on_god"]
        failures = eligible & (transformed != labels)
        neutral = failures & (transformed == 1)
        contradiction = failures & (transformed == 2)
        denominator = int(eligible.sum())
        failure_count = int(failures.sum())
        neutral_count = int(neutral.sum())
        contradiction_count = int(contradiction.sum())
        if (
            row["status"] != "complete"
            or row["primary"] != "false"
            or row["condition_id"] != "c05_informal_on_god"
            or row["scope"] != "model_clean_correct_entailment"
        ):
            raise RuntimeError(f"On-god destination identity/status mismatch: {context}")
        for field, expected in {
            "eligible_denominator": denominator,
            "failures": failure_count,
            "neutral_failures": neutral_count,
            "contradiction_failures": contradiction_count,
        }.items():
            expect_int(row[field], expected, f"{context}:{field}")
        for field, expected in {
            "failure_rate_eligible": failure_count / denominator,
            "neutral_rate_eligible": neutral_count / denominator,
            "contradiction_rate_eligible": contradiction_count / denominator,
            "neutral_rate_failures": (
                neutral_count / failure_count if failure_count else math.nan
            ),
            "contradiction_rate_failures": (
                contradiction_count / failure_count if failure_count else math.nan
            ),
        }.items():
            expect_float(row[field], expected, f"{context}:{field}")
    return {"path": path, "rows": len(rows), "sha256": sha256_file(path)}


def validate_cofailure_rows(
    analysis_dir: Path,
    data: dict[tuple[str, str, str], dict[str, Any]],
    registry: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    path = analysis_dir / "cofailure_results_v2.csv"
    rows = read_csv(path)
    summary_rows = [row for row in rows if row["record_type"] == "condition_summary"]
    pair_rows = [row for row in rows if row["record_type"] == "pairwise"]
    control_rows = [
        row
        for row in rows
        if row["record_type"] == "primary_all_wrong_control_contrast"
    ]
    expected_summary_keys = [
        (dataset, split, scope, condition["condition_id"])
        for dataset, split in SUITES
        for scope in (
            "five_model_common_clean_correct_all_labels",
            "five_model_common_clean_correct_entailment",
        )
        for condition in registry
    ]
    if [
        (row["dataset"], row["split"], row["scope"], row["condition_id"])
        for row in summary_rows
    ] != expected_summary_keys:
        raise RuntimeError("Cofailure condition-summary support/order mismatch")
    expected_pair_keys = [
        (dataset, split, MODELS[left], MODELS[right], scope, condition["condition_id"])
        for dataset, split in SUITES
        for left in range(len(MODELS))
        for right in range(left + 1, len(MODELS))
        for scope in (
            "pairwise_common_clean_correct_all_labels",
            "pairwise_common_clean_correct_entailment",
        )
        for condition in registry
    ]
    if [
        (
            row["dataset"],
            row["split"],
            row["model_a"],
            row["model_b"],
            row["scope"],
            row["condition_id"],
        )
        for row in pair_rows
    ] != expected_pair_keys:
        raise RuntimeError("Cofailure pairwise support/order mismatch")
    expected_control_keys = [
        (dataset, split, comparator)
        for dataset, split in SUITES
        for comparator in TWO_TOKEN_CONTROLS
    ]
    if [
        (row["dataset"], row["split"], row["comparator_condition"])
        for row in control_rows
    ] != expected_control_keys:
        raise RuntimeError("Cofailure frozen control support/order mismatch")
    if len(rows) != 1596 or len(summary_rows) != 144 or len(pair_rows) != 1440:
        raise RuntimeError(f"Unexpected cofailure row count: {len(rows)}")

    fixed_records: list[dict[str, Any]] = []
    headline: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        dataset, split, scope, condition_id = (
            row["dataset"],
            row["split"],
            row["scope"],
            row["condition_id"],
        )
        context = f"{dataset}:{split}:{scope}:{condition_id}"
        cells = [data[(model, dataset, split)] for model in MODELS]
        labels = cells[0]["labels"]
        eligible = np.logical_and.reduce(
            [cell["predictions"]["c00_clean"] == labels for cell in cells]
        )
        if scope == "five_model_common_clean_correct_entailment":
            eligible &= labels == 0
        population = int(eligible.sum())
        prediction_matrix = np.stack(
            [cell["predictions"][condition_id] for cell in cells]
        )[:, eligible]
        failure_matrix = prediction_matrix != labels[eligible][None, :]
        all_wrong = np.all(failure_matrix, axis=0)
        same_destination = all_wrong & np.all(
            prediction_matrix == prediction_matrix[0:1, :], axis=0
        )
        margins = [int(value) for value in failure_matrix.sum(axis=1)]
        rates = [value / population for value in margins]
        expected_count = population * float(np.prod(rates))
        observed = int(all_wrong.sum())
        ratio = observed / expected_count if expected_count > 0 else math.nan
        same_counts = [
            int(np.sum(same_destination & (prediction_matrix[0] == destination)))
            for destination in range(3)
        ]
        key = (
            f"cofailure|{scope}|{dataset}|{split}|{condition_id}"
        )
        bootstrap = bootstrap_joint_metrics(failure_matrix, key)
        same_low, same_high = bootstrap_mean_ci(
            same_destination.astype(float),
            key + "|same_destination",
        )
        fixed_p = (
            fixed_margin_intersection_tail(population, margins, observed)
            if scope == "five_model_common_clean_correct_entailment"
            else None
        )
        if (
            row["status"] != "complete"
            or row["primary"] != "false"
            or row["comparator_condition"] != ""
        ):
            raise RuntimeError(f"Cofailure summary identity/status mismatch: {context}")
        for field, expected in {
            "eligible_denominator": population,
            "observed_joint_failures": observed,
            "synchronized_destination_failures": int(same_destination.sum()),
            "synchronized_entailment_count": same_counts[0],
            "synchronized_neutral_count": same_counts[1],
            "synchronized_contradiction_count": same_counts[2],
        }.items():
            expect_int(row[field], expected, f"{context}:{field}")
        try:
            recorded_margins = json.loads(row["model_failure_margins_json"])
            recorded_rates = json.loads(row["model_failure_rates_json"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed cofailure model JSON: {context}") from exc
        if recorded_margins != dict(zip(MODELS, margins)):
            raise RuntimeError(f"Cofailure model-margin mismatch: {context}")
        for model, expected in zip(MODELS, rates):
            expect_float(
                str(recorded_rates.get(model)),
                expected,
                f"{context}:failure_rate:{model}",
            )
        float_expectations = {
            "observed_joint_rate": observed / population,
            "observed_joint_rate_ci_low": bootstrap["joint_rate"][0],
            "observed_joint_rate_ci_high": bootstrap["joint_rate"][1],
            "independence_expected_failures": expected_count,
            "independence_expected_failures_ci_low": bootstrap["expected_count"][0],
            "independence_expected_failures_ci_high": bootstrap["expected_count"][1],
            "observed_minus_independence_expected_failures": observed - expected_count,
            "observed_minus_independence_expected_failures_ci_low": bootstrap["excess_count"][0],
            "observed_minus_independence_expected_failures_ci_high": bootstrap["excess_count"][1],
            "observed_expected_ratio": ratio,
            "observed_expected_ratio_ci_low": bootstrap["ratio"][0],
            "observed_expected_ratio_ci_high": bootstrap["ratio"][1],
            "synchronized_destination_rate": float(same_destination.mean()),
            "synchronized_destination_rate_ci_low": same_low,
            "synchronized_destination_rate_ci_high": same_high,
        }
        for field, expected in float_expectations.items():
            expect_float(row[field], expected, f"{context}:{field}")
        expected_status = "defined" if expected_count > 0 else "undefined_zero_expected"
        if fixed_p is not None:
            expect_float(
                row["fixed_margin_intersection_p_upper"],
                fixed_p,
                f"{context}:fixed_margin_intersection_p_upper",
            )
            expected_fixed_status = (
                "underflow_or_below_reporting_floor_p_lt_1e-300"
                if fixed_p < 1e-300
                else "finite"
            )
            expected_fixed_null = (
                "independent_uniform_failure_subsets_conditional_on_model_margins"
            )
        else:
            if row["fixed_margin_intersection_p_upper"] != "":
                raise RuntimeError(f"All-label summary unexpectedly has fixed p: {context}")
            expected_fixed_status = "not_targeted_all_label_scope"
            expected_fixed_null = ""
        if (
            row["observed_expected_status"] != expected_status
            or row["fixed_margin_p_status"] != expected_fixed_status
            or row["fixed_margin_null"] != expected_fixed_null
        ):
            raise RuntimeError(f"Cofailure status/null mismatch: {context}")
        if fixed_p is not None:
            fixed_records.append({"row": row, "p_raw": fixed_p})
        elif row["fixed_margin_holm_p_72"] != "":
            raise RuntimeError(f"All-label summary unexpectedly has fixed Holm p: {context}")
        if (
            scope == "five_model_common_clean_correct_entailment"
            and condition_id == "c05_informal_on_god"
        ):
            headline[f"{dataset}:{split}"] = {
                "eligible": population,
                "all_five_wrong": observed,
                "all_five_wrong_rate": observed / population,
                "product_marginal_expected": expected_count,
                "observed_expected_ratio": ratio,
                "same_destination": int(same_destination.sum()),
                "fixed_margin_p_raw": fixed_p,
            }

    fixed_holm = holm_adjust([record["p_raw"] for record in fixed_records])
    for record, adjusted in zip(fixed_records, fixed_holm):
        expect_float(
            record["row"]["fixed_margin_holm_p_72"],
            adjusted,
            (
                f"{record['row']['dataset']}:{record['row']['split']}:"
                f"{record['row']['condition_id']}:fixed_margin_holm_p_72"
            ),
        )

    for row in pair_rows:
        dataset, split = row["dataset"], row["split"]
        left_model, right_model = row["model_a"], row["model_b"]
        scope, condition_id = row["scope"], row["condition_id"]
        context = (
            f"{dataset}:{split}:{left_model}:{right_model}:{scope}:{condition_id}"
        )
        left = data[(left_model, dataset, split)]
        right = data[(right_model, dataset, split)]
        labels = left["labels"]
        eligible = (
            (left["predictions"]["c00_clean"] == labels)
            & (right["predictions"]["c00_clean"] == labels)
        )
        if scope == "pairwise_common_clean_correct_entailment":
            eligible &= labels == 0
        elif scope != "pairwise_common_clean_correct_all_labels":
            raise RuntimeError(f"Unexpected pairwise scope: {context}")
        left_failures = (left["predictions"][condition_id] != labels)[eligible]
        right_failures = (right["predictions"][condition_id] != labels)[eligible]
        n11 = int(np.sum(left_failures & right_failures))
        n10 = int(np.sum(left_failures & ~right_failures))
        n01 = int(np.sum(~left_failures & right_failures))
        n00 = int(np.sum(~left_failures & ~right_failures))
        corrected = any(value == 0 for value in (n11, n10, n01, n00))
        table = (
            [value + 0.5 for value in (n11, n10, n01, n00)]
            if corrected
            else [n11, n10, n01, n00]
        )
        odds = table[0] * table[3] / (table[1] * table[2])
        population = int(eligible.sum())
        expected_rate = float(left_failures.mean() * right_failures.mean())
        excess = n11 / population - expected_rate
        bootstrap = bootstrap_pair_metrics(
            left_failures,
            right_failures,
            (
                f"pair|{scope}|{dataset}|{split}|{condition_id}|"
                f"{left_model}|{right_model}"
            ),
        )
        if (
            row["status"] != "complete"
            or row["primary"] != "false"
            or row["comparator_condition"] != ""
            or row["fixed_margin_p_status"] != "not_applicable_pairwise"
        ):
            raise RuntimeError(f"Pairwise identity/status mismatch: {context}")
        expect_int(row["eligible_denominator"], population, f"{context}:denominator")
        expect_int(row["pair_joint_failures"], n11, f"{context}:joint_failures")
        for field, expected in {
            "pair_expected_rate": expected_rate,
            "pair_excess_joint_risk": excess,
            "pair_excess_joint_risk_ci_low": bootstrap["excess_risk"][0],
            "pair_excess_joint_risk_ci_high": bootstrap["excess_risk"][1],
            "odds_ratio": odds,
            "odds_ratio_ci_low": bootstrap["odds_ratio"][0],
            "odds_ratio_ci_high": bootstrap["odds_ratio"][1],
        }.items():
            expect_float(row[field], expected, f"{context}:{field}")
        if row["continuity_correction"] != str(corrected).lower():
            raise RuntimeError(f"Pairwise continuity-correction mismatch: {context}")

    primary_records: list[dict[str, Any]] = []
    for row in control_rows:
        dataset, split = row["dataset"], row["split"]
        comparator_condition = row["comparator_condition"]
        context = f"{dataset}:{split}:on_god_vs_{comparator_condition}"
        cells = [data[(model, dataset, split)] for model in MODELS]
        labels = cells[0]["labels"]
        eligible = (labels == 0) & np.logical_and.reduce(
            [cell["predictions"]["c00_clean"] == labels for cell in cells]
        )
        full_matrix = np.stack(
            [cell["predictions"]["c05_informal_on_god"] for cell in cells]
        )
        comparator_matrix = np.stack(
            [cell["predictions"][comparator_condition] for cell in cells]
        )
        full = np.all(full_matrix != labels[None, :], axis=0)[eligible]
        comparator = np.all(comparator_matrix != labels[None, :], axis=0)[eligible]
        differences = (full.astype(np.int8) - comparator.astype(np.int8)) * 100.0
        p_value, full_only, comparator_only = exact_mcnemar(full, comparator)
        low, high = bootstrap_mean_ci(
            differences,
            f"primary_all_wrong|{dataset}|{split}|{comparator_condition}",
        )
        if (
            row["status"] != "complete"
            or row["primary"] != "true"
            or row["scope"] != "five_model_common_clean_correct_entailment"
            or row["condition_id"] != "c05_informal_on_god"
            or row["fixed_margin_p_status"]
            != "not_applicable_paired_primary_contrast"
        ):
            raise RuntimeError(f"Cofailure control identity/status mismatch: {context}")
        for field, expected in {
            "eligible_denominator": int(eligible.sum()),
            "primary_full_events": int(full.sum()),
            "primary_comparator_events": int(comparator.sum()),
            "discordant_full_only": full_only,
            "discordant_comparator_only": comparator_only,
        }.items():
            expect_int(row[field], expected, f"{context}:{field}")
        for field, expected in {
            "effect_pp": float(differences.mean()),
            "bootstrap_ci_low_pp": low,
            "bootstrap_ci_high_pp": high,
            "mcnemar_p_raw": p_value,
        }.items():
            expect_float(row[field], expected, f"{context}:{field}")
        primary_records.append(
            {
                "row": row,
                "p_raw": p_value,
                "effect_pp": float(differences.mean()),
                "family": "primary_all_five_cofailure_control",
            }
        )
    return {
        "path": path,
        "rows": len(rows),
        "sha256": sha256_file(path),
        "fixed_margin_tests": len(fixed_records),
        "fixed_margin_holm_significant_0_05": sum(
            value < 0.05 for value in fixed_holm
        ),
        "five_model_summary_rows": len(summary_rows),
        "pairwise_rows": len(pair_rows),
        "pairwise_model_combinations": len(
            {(row["model_a"], row["model_b"]) for row in pair_rows}
        ),
        "control_tests": len(primary_records),
        "headline_on_god": headline,
    }, primary_records, fixed_records


def validate_global_holm(
    contrast_records: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
) -> dict[str, Any]:
    records = contrast_records + control_records
    if len(records) != PRIMARY_FAMILY_SIZE:
        raise RuntimeError(
            f"Frozen primary family must contain {PRIMARY_FAMILY_SIZE}, found {len(records)}"
        )
    adjusted = holm_adjust([record["p_raw"] for record in records])
    for record, expected in zip(records, adjusted):
        row = record["row"]
        context = "|".join(
            str(row.get(key, ""))
            for key in (
                "family",
                "model",
                "dataset",
                "split",
                "full_condition",
                "condition_id",
                "comparator_condition",
            )
        )
        expect_float(row["mcnemar_p_raw"], record["p_raw"], f"{context}:raw_p")
        expect_float(row["holm_p"], expected, f"{context}:holm_p")
    phrase = [
        (record, p)
        for record, p in zip(records, adjusted)
        if record["family"] == "primary_phrase_component_failure"
    ]
    neutral = [
        (record, p)
        for record, p in zip(records, adjusted)
        if record["family"] == "primary_on_god_neutral_destination"
    ]
    controls = [
        (record, p)
        for record, p in zip(records, adjusted)
        if record["family"] == "primary_all_five_cofailure_control"
    ]
    return {
        "total_tests": len(records),
        "holm_significant_0_05": sum(value < 0.05 for value in adjusted),
        "phrase_component_tests": len(phrase),
        "phrase_component_significant_positive": sum(
            record["effect_pp"] > 0 and value < 0.05 for record, value in phrase
        ),
        "phrase_component_significant_reverse": sum(
            record["effect_pp"] < 0 and value < 0.05 for record, value in phrase
        ),
        "on_god_destination_tests": len(neutral),
        "on_god_destination_significant_positive": sum(
            record["effect_pp"] > 0 and value < 0.05 for record, value in neutral
        ),
        "all_five_control_tests": len(controls),
        "all_five_control_significant": sum(value < 0.05 for _, value in controls),
        "all_five_control_significant_positive": sum(
            record["effect_pp"] > 0 and value < 0.05 for record, value in controls
        ),
        "all_five_control_significant_reverse": sum(
            record["effect_pp"] < 0 and value < 0.05 for record, value in controls
        ),
    }


def validate_fingerprints(
    analysis_dir: Path,
    data: dict[tuple[str, str, str], dict[str, Any]],
    registry: list[dict[str, str]],
) -> dict[str, Any]:
    path = analysis_dir / "model_phrase_fingerprints_v2.csv"
    rows = read_csv(path)
    informal = [
        row["condition_id"]
        for row in registry
        if row["condition_type"].startswith("informal_")
    ]
    expected_keys = [
        (model, dataset, split, scope, condition)
        for model in MODELS
        for dataset, split in SUITES
        for scope in ("clean_correct_all_labels", "clean_correct_entailment")
        for condition in informal
    ]
    if [
        (
            row["model"],
            row["dataset"],
            row["split"],
            row["scope"],
            row["condition_id"],
        )
        for row in rows
    ] != expected_keys:
        raise RuntimeError("Model-phrase fingerprint support/order mismatch")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        context = (
            f"{row['model']}:{row['dataset']}:{row['split']}:"
            f"{row['scope']}:{row['condition_id']}"
        )
        cell = data[(row["model"], row["dataset"], row["split"])]
        labels = cell["labels"]
        predictions = cell["predictions"]
        eligible = predictions["c00_clean"] == labels
        if row["scope"] == "clean_correct_entailment":
            eligible &= labels == 0
        elif row["scope"] != "clean_correct_all_labels":
            raise RuntimeError(f"Unexpected fingerprint scope: {context}")
        transformed = predictions[row["condition_id"]]
        failures = eligible & (transformed != labels)
        neutral = failures & (transformed == 1)
        denominator = int(eligible.sum())
        if row["status"] != "complete":
            raise RuntimeError(f"Fingerprint status mismatch: {context}")
        expect_int(row["eligible_denominator"], denominator, f"{context}:denominator")
        expect_int(row["failures"], int(failures.sum()), f"{context}:failures")
        expect_int(
            row["neutral_destinations"],
            int(neutral.sum()),
            f"{context}:neutral_destinations",
        )
        expect_float(
            row["failure_rate"],
            float(failures.sum() / denominator),
            f"{context}:failure_rate",
        )
        expect_float(
            row["neutral_rate"],
            float(neutral.sum() / denominator),
            f"{context}:neutral_rate",
        )
        grouped[
            (row["dataset"], row["split"], row["scope"], row["condition_id"])
        ].append(row)
    for key, group in grouped.items():
        mean = float(np.mean([float(row["failure_rate"]) for row in group]))
        ordered = sorted(group, key=lambda row: (-float(row["failure_rate"]), row["model"]))
        for rank, row in enumerate(ordered, start=1):
            context = ":".join(key) + ":" + row["model"]
            expect_int(row["model_rank"], rank, f"{context}:rank")
            expect_float(
                row["model_minus_five_model_mean_pp"],
                (float(row["failure_rate"]) - mean) * 100.0,
                f"{context}:mean_deviation",
            )
    return {"path": path, "rows": len(rows), "sha256": sha256_file(path)}


def validate_report_headlines(
    analysis_dir: Path,
    primary: dict[str, Any],
    condition: dict[str, Any],
    cofailure: dict[str, Any],
) -> dict[str, Any]:
    path = analysis_dir / "REPORT.md"
    if not path.is_file():
        raise RuntimeError(f"Analyzer report is missing: {path}")
    report = path.read_text(encoding="utf-8")
    required_fragments = (
        "0=entailment, 1=neutral, and 2=contradiction",
        f"{primary['holm_significant_0_05']}/132 tests are",
        (
            f"Phrase/component failure contrasts: "
            f"{primary['phrase_component_significant_positive']}/90 significant"
        ),
        f"{primary['phrase_component_significant_reverse']}/90 significant in the reverse",
        (
            f"`on god` neutral-destination contrasts: "
            f"{primary['on_god_destination_significant_positive']}/30 significant"
        ),
        (
            f"All-five `on god`/control contrasts: "
            f"{primary['all_five_control_significant_positive']}/12 significant in the"
        ),
        f"{primary['all_five_control_significant_reverse']}/12 significant in reverse",
        f"from {condition['on_god_accuracy_change_min_pp']:.3f}",
        f"to {condition['on_god_accuracy_change_max_pp']:.3f}",
    )
    missing = [fragment for fragment in required_fragments if fragment not in report]
    if missing:
        raise RuntimeError(f"Analyzer report headline mismatch; missing fragments: {missing}")
    for suite, headline in cofailure["headline_on_god"].items():
        dataset, split = suite.split(":", 1)
        row_fragment = (
            f"| {dataset}:{split} | {headline['eligible']} | "
            f"{headline['all_five_wrong']} "
            f"({100.0 * headline['all_five_wrong_rate']:.2f}%) | "
            f"{headline['product_marginal_expected']:.2f} | "
            f"{headline['observed_expected_ratio']:.2f} | "
            f"{headline['same_destination']} |"
        )
        if row_fragment not in report:
            raise RuntimeError(f"Analyzer report cofailure headline mismatch: {suite}")
    return {"path": path, "sha256": sha256_file(path), "headline_checks": "pass"}


def validate_analyzer_hash_record(
    analysis_dir: Path,
    output_summaries: Iterable[dict[str, Any]],
    report_summary: dict[str, Any],
) -> dict[str, Any]:
    path = analysis_dir / "validation_v2.json"
    logical = check_json_sidecar(path)
    claim = json.loads(path.read_text(encoding="utf-8"))
    # These checks establish consistency only.  All scientific values and
    # hashes above were derived independently before reading this record.
    if claim.get("status") != "complete" or int(claim.get("schema_version", -1)) != 2:
        raise RuntimeError("Analyzer validation record has unexpected status/schema")
    analyzer_code = Path(__file__).resolve().with_name("analyze_v2.py")
    if claim.get("analysis_code_sha256") != sha256_file(analyzer_code):
        raise RuntimeError("Analyzer code hash claim differs from current analyzer bytes")
    summaries = list(output_summaries) + [report_summary]
    independent_by_name = {
        summary["path"].name: {
            "sha256": summary["sha256"],
            "rows": summary.get("rows"),
        }
        for summary in summaries
    }
    claims_by_name: dict[str, dict[str, Any]] = {}
    for claimed_path, values in claim.get("outputs", {}).items():
        name = Path(claimed_path).name
        if name in claims_by_name:
            raise RuntimeError(f"Duplicate analyzer output basename in validation: {name}")
        claims_by_name[name] = values
    if set(claims_by_name) != set(independent_by_name):
        raise RuntimeError(
            "Analyzer validation output set differs from independently checked outputs"
        )
    for name, expected in independent_by_name.items():
        observed = claims_by_name[name]
        if observed.get("sha256") != expected["sha256"]:
            raise RuntimeError(f"Analyzer validation hash claim mismatch: {name}")
        if observed.get("rows") != expected["rows"]:
            raise RuntimeError(f"Analyzer validation row-count claim mismatch: {name}")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "logical_sha256": logical,
        "logical_sidecar_sha256": sha256_file(
            path.with_suffix(path.suffix + ".logical.sha256")
        ),
        "record_consistency": "pass_not_used_as_evidence",
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_clean_replay_agreement <= 1.0:
        raise ValueError("--min-clean-replay-agreement must lie in [0, 1]")
    source = args.source_workspace.resolve()
    analysis_dir = args.analysis_dir.resolve()
    data, registry, run, input_audit, eval_support, replay = load_inputs(args)

    condition = validate_condition_effects(analysis_dir, data, registry)
    contrasts, contrast_records = validate_primary_contrasts(analysis_dir, data)
    destinations = validate_destination_rows(analysis_dir, data)
    fingerprints = validate_fingerprints(analysis_dir, data, registry)
    cofailure, control_records, fixed_records = validate_cofailure_rows(
        analysis_dir,
        data,
        registry,
    )
    primary = validate_global_holm(contrast_records, control_records)
    report = validate_report_headlines(analysis_dir, primary, condition, cofailure)
    analyzer_record = validate_analyzer_hash_record(
        analysis_dir,
        (condition, contrasts, cofailure, destinations, fingerprints),
        report,
    )

    analyzer_outputs = {
        summary["path"].name: {
            "sha256": summary["sha256"],
            "rows": summary.get("rows"),
        }
        for summary in (
            condition,
            contrasts,
            cofailure,
            destinations,
            fingerprints,
            report,
        )
    }
    output = args.output.resolve()
    payload = {
        "schema_version": 2,
        "status": "pass",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validator_code_sha256": sha256_file(Path(__file__).resolve()),
        "independence_contract": {
            "imports_analyze_v2": False,
            "trusts_validation_v2": False,
            "raw_predictions_reread": True,
            "frozen_evaluation_matrices_reread": True,
            "condition_major_crossing_recomputed": True,
            "statistical_endpoints_recomputed": True,
        },
        "label_semantics": {
            "mapping": {str(index): name for index, name in LABEL_NAMES.items()},
            "assertion": "frozen_project_class_order",
            "destination_columns_bound_to_mapping": True,
            "independent_destination_count_recomputation": True,
        },
        "input_contract": {
            "validated_v1_non_bertweet_cells": sum(
                row["source"] == "validated_v1" for row in input_audit
            ),
            "corrected_v2_bertweet_cells": sum(
                row["source"] == "corrected_v2" for row in input_audit
            ),
            "total_cells": len(input_audit),
            "condition_count": len(registry),
            "validated_prediction_rows": sum(row["prediction_rows"] for row in input_audit),
            "source_run_manifest_sha256": sha256_file(source / "run_manifest.json"),
            "source_run_manifest_logical_sha256": logical_json_checksum(
                source / "run_manifest.json"
            ),
            "approved_pre_execution_manifest_logical_sha256": run.get(
                "pre_execution_run_manifest_logical_sha256"
            ),
            "condition_registry_sha256": sha256_file(
                source / "CONDITION_REGISTRY.csv"
            ),
            "evaluation_support": {
                f"{dataset}:{split}": {
                    key: value
                    for key, value in support.items()
                    if key != "labels"
                }
                for (dataset, split), support in eval_support.items()
            },
            "artifact_audit": input_audit,
            "corrected_clean_replay": replay,
            "invalid_v1_bertweet_read": False,
        },
        "analysis_recomputation": {
            "condition_effects": {
                key: value
                for key, value in condition.items()
                if key not in {"path", "sha256"}
            },
            "primary_family": primary,
            "fixed_margin_family": {
                "tests": len(fixed_records),
                "holm_significant_0_05": cofailure[
                    "fixed_margin_holm_significant_0_05"
                ],
            },
            "cofailure_headline_on_god": cofailure["headline_on_god"],
            "secondary_output_topology": {
                "paired_contrasts": contrasts["secondary_tests"],
                "five_model_condition_summaries": cofailure[
                    "five_model_summary_rows"
                ],
                "pairwise_model_rows": cofailure["pairwise_rows"],
                "pairwise_model_combinations": cofailure[
                    "pairwise_model_combinations"
                ],
                "model_phrase_fingerprints": fingerprints["rows"],
                "destination_rows": destinations["rows"],
            },
            "destination_rows": destinations["rows"],
            "report_headlines": "pass",
        },
        "analyzer_outputs": analyzer_outputs,
        "analyzer_validation_record": analyzer_record,
    }
    if (
        payload["input_contract"]["validated_v1_non_bertweet_cells"] != 12
        or payload["input_contract"]["corrected_v2_bertweet_cells"] != 3
        or primary["total_tests"] != PRIMARY_FAMILY_SIZE
        or len(fixed_records) != FIXED_MARGIN_FAMILY_SIZE
        or destinations["rows"] != 15
        or contrasts["secondary_tests"] != 450
        or cofailure["five_model_summary_rows"] != 144
        or cofailure["pairwise_rows"] != 1440
        or fingerprints["rows"] != 270
    ):
        raise RuntimeError("Independent validation final contract assertion failed")
    write_json_with_sidecar(output, payload)
    print(
        json.dumps(
            {
                "status": "pass",
                "primary_tests": primary["total_tests"],
                "primary_holm_significant": primary["holm_significant_0_05"],
                "fixed_margin_tests": len(fixed_records),
                "validated_prediction_rows": payload["input_contract"][
                    "validated_prediction_rows"
                ],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
