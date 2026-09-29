#!/usr/bin/env python3
"""Auditable analysis for the tokenizer-corrected phrase/component run.

This script preserves the pre-inference v1 scientific protocol.  It loads the
12 validated, non-BERTweet v1 cells and exactly three tokenizer-corrected
BERTweet cells, then recomputes the frozen 132-test primary family.  It never
reads the invalid v1 BERTweet predictions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest, hypergeom


MODELS = ("electra", "roberta", "roberta_base", "timelm", "bertweet")
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
LEGACY_MODELS = tuple(model for model in MODELS if model != "bertweet")
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
FROZEN_BOOTSTRAP_REPLICATES = 5000
FROZEN_BOOTSTRAP_SEED = 20260726
FROZEN_PRIMARY_TESTS = 132
DEFAULT_MIN_REPLAY_AGREEMENT = 0.995
PINNED_BERTWEET_TOKENIZER_REVISION = "b349c1243407b0dcffeabb2337497477286e27ab"
PINNED_BERTWEET_TOKENIZER_REPOSITORY = "vinai/bertweet-base"
PINNED_BERTWEET_TOKENIZER_PACKAGE_SHA256 = (
    "934aa09df86b70e9ca86aace92f7dc58d276fc906ae9c6373584fdf49038094f"
)
PINNED_BERTWEET_TOKENIZER_FILE_SHA256 = {
    "bpe.codes": "77712739cd1a7f638e6694b0dd832494e4f66e3d05c709fc6a6a2f988ff9e589",
    "config.json": "7926dbeefbaabac88352b291f86c2363b5d54164b8e67437ea0edae7010257a6",
    "tokenizer.json": "48a8972b321c93163b78d98f40bb410d898d8869d8439b6abb5f8283f545b85d",
    "vocab.txt": "d3f3d56ed440cdb39bd60a76884b67e1061abca462146fdcc751f1ee40ae9ed3",
}
PINNED_MNLI_WEIGHT_SHA256 = (
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
        help="Frozen v1 workspace containing manifest, registry, and 12 valid cells.",
    )
    parser.add_argument(
        "--corrected-predictions",
        type=Path,
        default=here / "predictions",
        help="Directory recursively searched for the three corrected BERTweet cells.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=repository / "modal_backup_2026-07-26" / "volumes" / "wnut2026-nli-results-v2",
        help="Result-volume backup containing archived seed-42 clean predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here,
        help="Destination for v2 CSV, JSON, and Markdown outputs.",
    )
    parser.add_argument(
        "--min-clean-replay-agreement",
        type=float,
        default=DEFAULT_MIN_REPLAY_AGREEMENT,
        help="Independent hard gate for corrected-vs-archived clean predictions.",
    )
    parser.add_argument(
        "--require-exact-clean-replay",
        action="store_true",
        help="Raise unless every corrected clean prediction matches the archive.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_json_checksum(path: Path) -> str:
    encoded = json.dumps(
        json.loads(path.read_text(encoding="utf-8")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logical_object_checksum(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def portable_path(path: Path, repository: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository.resolve()).as_posix()
    except ValueError:
        return resolved.name


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Two-sided exact McNemar p and the two discordant counts."""

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(a_only, n=discordant, p=0.5, alternative="two-sided").pvalue
    )
    return p_value, a_only, b_only


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: float(p_values[index]))
    adjusted = [1.0] * len(p_values)
    running = 0.0
    family_size = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def exact_intersection_upper_tail(n: int, margins: Sequence[int], observed: int) -> float:
    """Fixed-margin upper tail for the intersection of independent subsets.

    Conditional on each model's observed failure count, each failure subset is
    treated as uniformly located over the eligible sources.  Recursive
    hypergeometric convolution gives the all-model intersection distribution.
    """

    # Intersection is permutation-invariant; starting with the smallest margin
    # reduces the recursive support without changing the null distribution.
    margins = sorted(int(value) for value in margins)
    if not margins:
        return 1.0
    if n < 0 or any(value < 0 or value > n for value in margins):
        raise ValueError((n, margins, observed))
    if observed < 0 or observed > min(margins):
        raise ValueError((n, margins, observed))
    pmf = np.zeros(n + 1, dtype=np.float64)
    pmf[margins[0]] = 1.0
    current_max = margins[0]
    for subset_size in margins[1:]:
        updated = np.zeros(n + 1, dtype=np.float64)
        for current in range(current_max + 1):
            weight = pmf[current]
            if weight == 0.0:
                continue
            lower = max(0, subset_size - (n - current))
            upper = min(current, subset_size)
            support = np.arange(lower, upper + 1, dtype=np.int32)
            updated[support] += weight * hypergeom.pmf(support, n, current, subset_size)
        pmf = updated
        current_max = min(current_max, subset_size)
    total = float(pmf.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"Intersection PMF failed normalization: {total}")
    return float(np.clip(pmf[observed:].sum(), 0.0, 1.0))


def expected_crossing_position(
    row_index: int,
    source_pairs: int,
    condition_ids: Sequence[str],
) -> tuple[str, int]:
    """Return the only valid condition/source identity for a global row index."""

    if row_index < 0 or source_pairs <= 0:
        raise ValueError((row_index, source_pairs))
    condition_index, source_index = divmod(row_index, source_pairs)
    if condition_index >= len(condition_ids):
        raise ValueError(f"Row {row_index} exceeds the frozen condition matrix")
    return condition_ids[condition_index], source_index


def corrected_runtime_profile_is_valid(runtime: Any) -> bool:
    device = runtime.get("device") if isinstance(runtime, dict) else None
    device_valid = device == "cpu" or (
        device == "cuda"
        and isinstance(runtime.get("accelerator_name"), str)
        and "H100" in runtime["accelerator_name"].upper()
        and isinstance(runtime.get("accelerator_count"), int)
        and runtime["accelerator_count"] >= 1
        and isinstance(runtime.get("cuda_version"), str)
        and bool(runtime["cuda_version"])
        and runtime.get("allow_tf32") is False
    )
    return (
        isinstance(runtime, dict)
        and device_valid
        and isinstance(runtime.get("threads"), int)
        and runtime["threads"] > 0
        and runtime.get("interop_threads") == 1
        and isinstance(runtime.get("batch_size"), int)
        and runtime["batch_size"] > 0
        and runtime.get("max_length") == 128
        and runtime.get("input_preprocessing") == "none"
        and runtime.get("tokenizer_backend") == "slow"
        and runtime.get("tokenizer_normalization") == "disabled"
        and runtime.get("logit_dtype") == "float32"
        and runtime.get("seed") == 42
        and runtime.get("deterministic_algorithms") is True
        and isinstance(runtime.get("torch_version"), str)
        and bool(runtime["torch_version"])
    )


def recompute_correction_identity(
    completion: dict[str, Any],
    cell: dict[str, Any],
) -> str:
    checkpoint = completion.get("checkpoint_identity")
    tokenizer = completion.get("tokenizer_identity")
    runtime = completion.get("runtime_profile")
    if not isinstance(checkpoint, dict) or not isinstance(tokenizer, dict):
        raise RuntimeError("Corrected completion lacks checkpoint/tokenizer identity")
    weight_source = checkpoint.get("weight_source")
    if weight_source not in {"checkpoint_package", "verified_remote_override"}:
        raise RuntimeError(f"Unknown corrected weight source: {weight_source!r}")
    weight_sha = checkpoint.get("weight_sha256") if weight_source == "verified_remote_override" else None
    payload = {
        "schema_version": 2,
        "cell_id": cell["cell_id"],
        "source_cell_identity_sha256": cell["cell_identity_sha256"],
        "checkpoint_package_sha256": cell["trained_checkpoint_package_sha256"],
        "checkpoint_weight_sha256": weight_sha,
        "evaluation_logical_checksum_sha256": cell["condition_matrix_checksum_sha256"],
        "tokenizer_repository": tokenizer.get("repository"),
        "tokenizer_revision": tokenizer.get("revision"),
        "tokenizer_package_sha256": tokenizer.get("package_sha256"),
        "inference_code_sha256": completion.get("inference_code_sha256"),
        "runtime_profile": runtime,
    }
    return logical_object_checksum(payload)


def bootstrap_rng(key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{FROZEN_BOOTSTRAP_SEED}|{key}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def bootstrap_mean_ci(values: np.ndarray, key: str) -> tuple[float, float]:
    """5000-replicate source bootstrap via the exact outcome-category counts."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    unique, counts = np.unique(values, return_counts=True)
    draws = bootstrap_rng(key).multinomial(
        values.size,
        counts / values.size,
        size=FROZEN_BOOTSTRAP_REPLICATES,
    )
    estimates = (draws @ unique) / values.size
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def bootstrap_joint_metrics(failures: np.ndarray, key: str) -> dict[str, tuple[float, float]]:
    failures = np.asarray(failures, dtype=np.int8)
    patterns, counts = np.unique(failures.T, axis=0, return_counts=True)
    n = int(counts.sum())
    if n == 0:
        return {
            name: (math.nan, math.nan)
            for name in ("joint_rate", "expected_count", "excess_count", "ratio")
        }
    draws = bootstrap_rng(key).multinomial(
        n,
        counts / n,
        size=FROZEN_BOOTSTRAP_REPLICATES,
    )
    marginal_rates = (draws @ patterns) / n
    joint_rates = (draws @ np.all(patterns == 1, axis=1).astype(float)) / n
    expected_rates = np.prod(marginal_rates, axis=1)
    ratios = np.divide(
        joint_rates,
        expected_rates,
        out=np.full_like(joint_rates, np.nan),
        where=expected_rates > 0,
    )
    vectors = {
        "joint_rate": joint_rates,
        "expected_count": n * expected_rates,
        "excess_count": n * (joint_rates - expected_rates),
        "ratio": ratios,
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
    a: np.ndarray,
    b: np.ndarray,
    key: str,
) -> dict[str, tuple[float, float]]:
    patterns, counts = np.unique(
        np.column_stack((a, b)).astype(np.int8),
        axis=0,
        return_counts=True,
    )
    n = int(counts.sum())
    if n == 0:
        return {
            "excess_risk": (math.nan, math.nan),
            "odds_ratio": (math.nan, math.nan),
        }
    draws = bootstrap_rng(key).multinomial(
        n,
        counts / n,
        size=FROZEN_BOOTSTRAP_REPLICATES,
    )
    masks = {
        (left, right): np.all(patterns == (left, right), axis=1).astype(float)
        for left in (0, 1)
        for right in (0, 1)
    }
    n00 = draws @ masks[(0, 0)]
    n01 = draws @ masks[(0, 1)]
    n10 = draws @ masks[(1, 0)]
    n11 = draws @ masks[(1, 1)]
    marginal_a = (n10 + n11) / n
    marginal_b = (n01 + n11) / n
    excess = n11 / n - marginal_a * marginal_b
    zero = (n00 == 0) | (n01 == 0) | (n10 == 0) | (n11 == 0)
    odds = ((n11 + 0.5 * zero) * (n00 + 0.5 * zero)) / (
        (n10 + 0.5 * zero) * (n01 + 0.5 * zero)
    )
    return {
        "excess_risk": tuple(float(value) for value in np.quantile(excess, [0.025, 0.975])),
        "odds_ratio": tuple(float(value) for value in np.quantile(odds, [0.025, 0.975])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json_with_checksum(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".logical.sha256").write_text(
        logical_json_checksum(path) + "\n",
        encoding="ascii",
    )


def discover_artifact(root: Path, cell_id: str, recursive: bool) -> Path:
    pattern = f"{cell_id}__*.jsonl.gz"
    candidates = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one artifact for {cell_id} under {root}, found {len(candidates)}"
        )
    return candidates[0]


def completion_paths(artifact: Path) -> tuple[Path, Path]:
    completion = artifact.with_suffix(artifact.suffix + ".complete.json")
    logical = completion.with_suffix(completion.suffix + ".logical.sha256")
    if not completion.is_file() or not logical.is_file():
        raise RuntimeError(f"Missing completion JSON or logical checksum for {artifact.name}")
    if logical.read_text(encoding="ascii").strip() != logical_json_checksum(completion):
        raise RuntimeError(f"Completion logical checksum mismatch for {artifact.name}")
    return completion, logical


def load_registry(source_workspace: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    path = source_workspace / "CONDITION_REGISTRY.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = [f"c{index:02d}_" for index in range(24)]
    if len(rows) != 24 or any(
        not row.get("condition_id", "").startswith(prefix)
        for row, prefix in zip(rows, expected_ids)
    ):
        raise RuntimeError("Condition registry must contain c00--c23 in frozen order")
    if rows[0]["condition_id"] != "c00_clean":
        raise RuntimeError("First registry condition must be c00_clean")
    return rows, {row["condition_id"]: row for row in rows}


def validate_legacy_completion(
    completion: dict[str, Any],
    artifact: Path,
    cell: dict[str, Any],
    run: dict[str, Any],
) -> None:
    expected_manifest = run.get("pre_execution_run_manifest_logical_sha256")
    checks = {
        "status": completion.get("status") == "complete",
        "prediction_sha256": completion.get("prediction_sha256") == sha256_file(artifact),
        "cell_identity": completion.get("cell_identity_sha256") == cell["cell_identity_sha256"],
        "checkpoint": completion.get("checkpoint_package_sha256")
        == cell["trained_checkpoint_package_sha256"],
        "matrix": completion.get("evaluation_logical_checksum_sha256")
        == cell["condition_matrix_checksum_sha256"],
        "inference_code": completion.get("inference_code_sha256") == run["inference_code_sha256"],
        "runtime": completion.get("runtime_profile") == cell["runtime_profile"],
        "manifest": completion.get("run_manifest_logical_sha256") == expected_manifest,
        "rows": int(completion.get("prediction_rows", -1)) == int(cell["evaluation_rows"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Invalid legacy completion for {cell['cell_id']}: {failed}")


def validate_corrected_completion(
    completion: dict[str, Any],
    artifact: Path,
    cell: dict[str, Any],
    registry_sha256: str,
    source_manifest_logical_sha256: str,
    correction_code_sha256: str,
) -> None:
    matrix = completion.get("matrix_validation")
    tokenizer = completion.get("tokenizer_identity")
    checkpoint = completion.get("checkpoint_identity")
    runtime = completion.get("runtime_profile")
    observed_correction_identity = completion.get("correction_identity_sha256")
    recomputed_correction_identity = recompute_correction_identity(completion, cell)
    expected_filename = (
        f"{cell['cell_id']}__tokenizerfix__{recomputed_correction_identity[:16]}.jsonl.gz"
    )
    checks = {
        "status": completion.get("status") == "complete",
        "cell_id": completion.get("cell_id") == cell["cell_id"],
        "prediction_sha256": completion.get("prediction_sha256") == sha256_file(artifact),
        "checkpoint": completion.get("checkpoint_package_sha256")
        == cell["trained_checkpoint_package_sha256"],
        "matrix": completion.get("evaluation_logical_checksum_sha256")
        == cell["condition_matrix_checksum_sha256"],
        "registry": completion.get("condition_registry_sha256") == registry_sha256,
        "rows": int(completion.get("prediction_rows", -1)) == int(cell["evaluation_rows"]),
        "logical_prediction_hash": is_sha256(
            completion.get("prediction_logical_checksum_sha256")
        ),
        "correction_identity": is_sha256(observed_correction_identity)
        and observed_correction_identity == recomputed_correction_identity,
        "filename_identity": artifact.name == expected_filename
        and completion.get("prediction_artifact") == expected_filename,
        "source_cell_identity": completion.get("source_cell_identity_sha256")
        == cell["cell_identity_sha256"],
        "source_manifest_identity": is_sha256(
            completion.get("source_run_manifest_logical_sha256")
        )
        and completion.get("source_run_manifest_logical_sha256")
        == source_manifest_logical_sha256,
        "inference_code": completion.get("inference_code_sha256")
        == correction_code_sha256,
        "tokenizer_identity": isinstance(tokenizer, dict)
        and tokenizer.get("repository") == PINNED_BERTWEET_TOKENIZER_REPOSITORY
        and tokenizer.get("revision") == PINNED_BERTWEET_TOKENIZER_REVISION
        and tokenizer.get("package_sha256") == PINNED_BERTWEET_TOKENIZER_PACKAGE_SHA256
        and tokenizer.get("file_sha256") == PINNED_BERTWEET_TOKENIZER_FILE_SHA256
        and tokenizer.get("backend") == "slow"
        and tokenizer.get("normalization") is False,
        "checkpoint_identity": isinstance(checkpoint, dict)
        and checkpoint.get("package_sha256") == cell["trained_checkpoint_package_sha256"]
        and (
            (
                cell["dataset"] == "snli"
                and checkpoint.get("weight_source") == "checkpoint_package"
            )
            or (
                cell["dataset"] == "multi_nli"
                and checkpoint.get("weight_source") == "verified_remote_override"
                and checkpoint.get("weight_sha256") == PINNED_MNLI_WEIGHT_SHA256
            )
        ),
        "runtime_profile": corrected_runtime_profile_is_valid(runtime),
        "matrix_validation": isinstance(matrix, dict)
        and matrix.get("passed") is True
        and int(matrix.get("condition_count", -1)) == 24
        and int(matrix.get("source_pairs", -1)) == int(cell["source_pairs"])
        and int(matrix.get("matrix_rows", -1)) == int(cell["evaluation_rows"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Invalid corrected completion for {cell['cell_id']}: {failed}")


def load_prediction_artifact(
    artifact: Path,
    completion: dict[str, Any],
    cell: dict[str, Any],
    registry: list[dict[str, str]],
    registry_by_id: dict[str, dict[str, str]],
) -> dict[str, Any]:
    condition_ids = [row["condition_id"] for row in registry]
    expected_n = int(cell["source_pairs"])
    predictions: dict[str, list[int]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    source_ids: dict[str, list[int]] = defaultdict(list)
    logical = hashlib.sha256()
    row_count = 0
    with gzip.open(artifact, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            canonical = json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            logical.update(canonical.encode("utf-8") + b"\n")
            expected_condition, expected_source = expected_crossing_position(
                row_count,
                expected_n,
                condition_ids,
            )
            row_count += 1
            condition = row.get("condition_id")
            logits = row.get("logits")
            checks = (
                int(row.get("schema_version", -1)) == 1,
                condition == expected_condition,
                int(row.get("source_index", -1)) == expected_source,
                condition in registry_by_id,
                condition in registry_by_id
                and row.get("condition_type") == registry_by_id[condition]["condition_type"],
                row.get("cell_id") == cell["cell_id"],
                row.get("model") == cell["model"],
                row.get("dataset") == cell["dataset"],
                row.get("split") == cell["split"],
                row.get("checkpoint_package_sha256") == cell["trained_checkpoint_package_sha256"],
                int(row.get("label", -1)) in LABEL_NAMES,
                int(row.get("prediction", -1)) in LABEL_NAMES,
                isinstance(logits, list) and len(logits) == 3,
                isinstance(logits, list)
                and len(logits) == 3
                and all(math.isfinite(float(value)) for value in logits),
            )
            if not all(checks):
                raise RuntimeError(f"Prediction schema/identity failure in {artifact.name}")
            if max(range(3), key=lambda index: float(logits[index])) != int(row["prediction"]):
                raise RuntimeError(f"Prediction/logit argmax mismatch in {artifact.name}")
            predictions[condition].append(int(row["prediction"]))
            labels[condition].append(int(row["label"]))
            source_ids[condition].append(int(row["source_index"]))

    if list(predictions) != condition_ids:
        raise RuntimeError(f"Condition order mismatch in {artifact.name}")
    expected_sources = list(range(expected_n))
    reference_labels = labels["c00_clean"]
    for condition in condition_ids:
        if source_ids[condition] != expected_sources or labels[condition] != reference_labels:
            raise RuntimeError(f"Crossing/source/label mismatch: {cell['cell_id']}:{condition}")
    if row_count != int(cell["evaluation_rows"]):
        raise RuntimeError(f"Prediction row-count mismatch for {cell['cell_id']}")
    if logical.hexdigest() != completion.get("prediction_logical_checksum_sha256"):
        raise RuntimeError(f"Prediction logical checksum mismatch for {cell['cell_id']}")
    return {
        "labels": np.asarray(reference_labels, dtype=np.int8),
        "predictions": {
            condition: np.asarray(values, dtype=np.int8)
            for condition, values in predictions.items()
        },
        "row_count": row_count,
    }


def archive_prediction_path(archive_root: Path, dataset: str, split: str) -> Path:
    return (
        archive_root
        / "publication_results"
        / dataset
        / "seed_42"
        / f"bertweet_baseline_final_{split}"
        / "predictions.json"
    )


def validate_clean_replay(
    *,
    corrected: dict[str, Any],
    completion: dict[str, Any],
    archive_path: Path,
    minimum_agreement: float,
    require_exact: bool,
) -> dict[str, Any]:
    archive_payload = json.loads(archive_path.read_text(encoding="utf-8"))
    original = archive_payload.get("original")
    if not isinstance(original, dict):
        raise RuntimeError(f"Archived clean prediction block missing: {archive_path}")
    archived_predictions = np.asarray(original.get("predictions"), dtype=np.int8)
    archived_labels = np.asarray(original.get("labels"), dtype=np.int8)
    archived_sources = np.asarray(original.get("source_indices"), dtype=np.int64)
    corrected_clean = corrected["predictions"]["c00_clean"]
    corrected_labels = corrected["labels"]
    n = len(corrected_labels)
    if (
        archived_predictions.shape != (n,)
        or not np.array_equal(archived_labels, corrected_labels)
        or not np.array_equal(archived_sources, np.arange(n))
    ):
        raise RuntimeError(f"Archived clean source/label identity mismatch: {archive_path}")
    matched = int(np.sum(archived_predictions == corrected_clean))
    agreement = matched / n
    replay_accuracy = float(np.mean(corrected_clean == corrected_labels))
    reference_accuracy = float(np.mean(archived_predictions == archived_labels))
    reference_sha = sha256_file(archive_path)
    original_logical_sha = logical_object_checksum(original)
    claim = completion.get("clean_replay")
    if not isinstance(claim, dict):
        raise RuntimeError("Corrected completion must contain a clean_replay object")
    claim_checks = {
        "reference_prediction_sha256": claim.get("reference_prediction_sha256") == reference_sha,
        "reference_original_logical_sha256": claim.get("reference_original_logical_sha256")
        == original_logical_sha,
        "matched": int(claim.get("matched", -1)) == matched,
        "total": int(claim.get("total", -1)) == n,
        "agreement": math.isclose(float(claim.get("agreement", -1.0)), agreement, abs_tol=1e-15),
        "replay_accuracy": math.isclose(
            float(claim.get("replay_accuracy", -1.0)), replay_accuracy, abs_tol=1e-15
        ),
        "reference_accuracy": math.isclose(
            float(claim.get("reference_accuracy", -1.0)), reference_accuracy, abs_tol=1e-15
        ),
        "agreement_passed": claim.get("agreement_passed") is True,
        "accuracy_deviation_passed": claim.get("accuracy_deviation_passed") is True,
        "passed": claim.get("passed") is True,
    }
    failed = [name for name, passed in claim_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Clean-replay sidecar mismatch for {archive_path.name}: {failed}")
    claimed_threshold = float(claim.get("agreement_threshold", claim.get("threshold", math.nan)))
    claimed_accuracy_threshold = float(claim.get("accuracy_deviation_threshold", math.nan))
    claimed_accuracy_deviation = float(claim.get("accuracy_absolute_deviation", math.nan))
    if (
        not math.isfinite(claimed_threshold)
        or agreement < claimed_threshold
        or not math.isfinite(claimed_accuracy_threshold)
        or not math.isclose(
            claimed_accuracy_deviation,
            abs(replay_accuracy - reference_accuracy),
            abs_tol=1e-15,
        )
        or claimed_accuracy_deviation > claimed_accuracy_threshold
    ):
        raise RuntimeError(f"Clean-replay completion gate is internally inconsistent: {archive_path}")
    required = 1.0 if require_exact else minimum_agreement
    if agreement < required:
        raise RuntimeError(
            f"Independent clean-replay gate failed: {agreement:.8f} < {required:.8f}"
        )
    return {
        "reference_prediction_sha256": reference_sha,
        "reference_original_logical_sha256": original_logical_sha,
        "matched": matched,
        "total": n,
        "agreement": agreement,
        "independent_required_agreement": required,
        "replay_accuracy": replay_accuracy,
        "reference_accuracy": reference_accuracy,
        "accuracy_absolute_deviation": abs(replay_accuracy - reference_accuracy),
        "passed": True,
    }


def load_all_cells(args: argparse.Namespace) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    list[dict[str, str]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    source = args.source_workspace.resolve()
    run_path = source / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run_logical_sidecar = run_path.with_suffix(run_path.suffix + ".logical.sha256")
    if (
        not run_logical_sidecar.is_file()
        or run_logical_sidecar.read_text(encoding="ascii").strip()
        != logical_json_checksum(run_path)
    ):
        raise RuntimeError("Frozen v1 run-manifest logical sidecar mismatch")
    source_manifest_identity = run.get("pre_execution_run_manifest_logical_sha256")
    if not is_sha256(source_manifest_identity) or run.get("approval_granted") is not True:
        raise RuntimeError("Frozen v1 manifest lacks approved pre-execution identity")
    correction_code = Path(__file__).resolve().with_name("run_bertweet_cpu.py")
    if not correction_code.is_file():
        raise RuntimeError(f"Correction runner is missing: {correction_code}")
    correction_code_sha = sha256_file(correction_code)
    registry, registry_by_id = load_registry(source)
    registry_path = source / "CONDITION_REGISTRY.csv"
    registry_sha = sha256_file(registry_path)
    if registry_sha != run.get("condition_registry_sha256"):
        raise RuntimeError("Registry hash differs from the frozen run manifest")
    cells = run.get("cells")
    if not isinstance(cells, list) or len(cells) != 15:
        raise RuntimeError("Frozen manifest must contain exactly 15 cells")
    cell_keys = {(cell["model"], cell["dataset"], cell["split"]) for cell in cells}
    expected_keys = {
        (model, dataset, split)
        for model in MODELS
        for dataset, split in SUITES
    }
    if cell_keys != expected_keys:
        raise RuntimeError("Frozen manifest cell support differs from the required 5x3 grid")

    data: dict[tuple[str, str, str], dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for cell in cells:
        is_corrected = cell["model"] == "bertweet"
        artifact_root = args.corrected_predictions if is_corrected else source / "predictions"
        artifact = discover_artifact(artifact_root.resolve(), cell["cell_id"], recursive=is_corrected)
        completion_path, _ = completion_paths(artifact)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if is_corrected:
            validate_corrected_completion(
                completion,
                artifact,
                cell,
                registry_sha,
                source_manifest_identity,
                correction_code_sha,
            )
        else:
            validate_legacy_completion(completion, artifact, cell, run)
        loaded = load_prediction_artifact(
            artifact,
            completion,
            cell,
            registry,
            registry_by_id,
        )
        replay: dict[str, Any] | None = None
        if is_corrected:
            replay = validate_clean_replay(
                corrected=loaded,
                completion=completion,
                archive_path=archive_prediction_path(
                    args.archive_root.resolve(),
                    cell["dataset"],
                    cell["split"],
                ),
                minimum_agreement=args.min_clean_replay_agreement,
                require_exact=args.require_exact_clean_replay,
            )
        data[(cell["model"], cell["dataset"], cell["split"])] = loaded
        audit_rows.append(
            {
                "cell_id": cell["cell_id"],
                "model": cell["model"],
                "dataset": cell["dataset"],
                "split": cell["split"],
                "source": "corrected_v2" if is_corrected else "validated_v1",
                "prediction_artifact": artifact,
                "prediction_sha256": sha256_file(artifact),
                "prediction_logical_checksum_sha256": completion.get(
                    "prediction_logical_checksum_sha256"
                ),
                "completion_sha256": sha256_file(completion_path),
                "completion_logical_sha256": logical_json_checksum(completion_path),
                "prediction_rows": loaded["row_count"],
                "clean_replay": replay,
                "source_cell_identity_sha256": cell["cell_identity_sha256"],
                "correction_identity_sha256": (
                    completion.get("correction_identity_sha256") if is_corrected else None
                ),
                "inference_code_sha256": completion.get("inference_code_sha256"),
                "runtime_profile": completion.get("runtime_profile"),
                "tokenizer_identity": (
                    completion.get("tokenizer_identity") if is_corrected else None
                ),
            }
        )

    for dataset, split in SUITES:
        suite_cells = [data[(model, dataset, split)] for model in MODELS]
        labels = suite_cells[0]["labels"]
        if any(not np.array_equal(labels, cell["labels"]) for cell in suite_cells[1:]):
            raise RuntimeError(f"Five-model label support differs for {dataset}:{split}")
        matrix_hashes = {
            cell["condition_matrix_checksum_sha256"]
            for cell in cells
            if cell["dataset"] == dataset and cell["split"] == split
        }
        if len(matrix_hashes) != 1:
            raise RuntimeError(f"Five-model matrix identity differs for {dataset}:{split}")
    return data, registry, run, audit_rows


def add_contrast(
    rows: list[dict[str, Any]],
    *,
    family: str,
    primary: bool,
    endpoint: str,
    model: str,
    dataset: str,
    split: str,
    full: str,
    comparator: str,
    eligible: np.ndarray,
    full_event: np.ndarray,
    comparator_event: np.ndarray,
) -> None:
    a = np.asarray(full_event[eligible], dtype=np.int8)
    b = np.asarray(comparator_event[eligible], dtype=np.int8)
    differences = (a - b).astype(float) * 100.0
    key = "|".join((family, endpoint, model, dataset, split, full, comparator))
    low, high = bootstrap_mean_ci(differences, key)
    p_value, full_only, comparator_only = exact_mcnemar(a, b)
    rows.append(
        {
            "status": "complete",
            "family": family,
            "primary": str(primary).lower(),
            "endpoint": endpoint,
            "model": model,
            "dataset": dataset,
            "split": split,
            "full_condition": full,
            "comparator_condition": comparator,
            "eligible_denominator": int(eligible.sum()),
            "full_events": int(a.sum()),
            "comparator_events": int(b.sum()),
            "effect_pp": float(differences.mean()) if differences.size else math.nan,
            "bootstrap_ci_low_pp": low,
            "bootstrap_ci_high_pp": high,
            "discordant_full_only": full_only,
            "discordant_comparator_only": comparator_only,
            "mcnemar_p_raw": p_value,
            "holm_p": "",
        }
    )


def compute_results(
    data: dict[tuple[str, str, str], dict[str, Any]],
    registry: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    condition_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    cofailure_rows: list[dict[str, Any]] = []
    destination_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []

    for model in MODELS:
        for dataset, split in SUITES:
            cell = data[(model, dataset, split)]
            labels = cell["labels"]
            predictions = cell["predictions"]
            clean = predictions["c00_clean"]
            clean_correct = clean == labels
            clean_accuracy = float(clean_correct.mean())
            for condition in registry:
                condition_id = condition["condition_id"]
                prediction = predictions[condition_id]
                correct = prediction == labels
                difference = (correct.astype(np.int8) - clean_correct.astype(np.int8)) * 100.0
                low, high = bootstrap_mean_ci(
                    difference,
                    f"effect|{model}|{dataset}|{split}|{condition_id}",
                )
                failures = clean_correct & ~correct
                clean_n = int(clean_correct.sum())
                failure_n = int(failures.sum())
                failure_low, failure_high = bootstrap_mean_ci(
                    failures[clean_correct].astype(float),
                    f"failure_rate|{model}|{dataset}|{split}|{condition_id}",
                )
                destination_counts = [
                    int(np.sum(failures & (prediction == destination)))
                    for destination in range(3)
                ]
                condition_rows.append(
                    {
                        "status": "complete",
                        "model": model,
                        "dataset": dataset,
                        "split": split,
                        "condition_id": condition_id,
                        "condition_type": condition["condition_type"],
                        "denominator": len(labels),
                        "accuracy": float(correct.mean()),
                        "clean_accuracy": clean_accuracy,
                        "transformed_minus_clean_pp": float(difference.mean()),
                        "bootstrap_ci_low_pp": low,
                        "bootstrap_ci_high_pp": high,
                        "clean_correct_denominator": clean_n,
                        "failures": failure_n,
                        "failure_rate": failure_n / clean_n if clean_n else math.nan,
                        "failure_rate_ci_low": failure_low,
                        "failure_rate_ci_high": failure_high,
                        "destination_entailment_count": destination_counts[0],
                        "destination_neutral_count": destination_counts[1],
                        "destination_contradiction_count": destination_counts[2],
                        "destination_entailment_rate_clean_correct": destination_counts[0] / clean_n if clean_n else math.nan,
                        "destination_neutral_rate_clean_correct": destination_counts[1] / clean_n if clean_n else math.nan,
                        "destination_contradiction_rate_clean_correct": destination_counts[2] / clean_n if clean_n else math.nan,
                        "destination_entailment_rate_failures": destination_counts[0] / failure_n if failure_n else math.nan,
                        "destination_neutral_rate_failures": destination_counts[1] / failure_n if failure_n else math.nan,
                        "destination_contradiction_rate_failures": destination_counts[2] / failure_n if failure_n else math.nan,
                    }
                )

            eligible = clean_correct & (labels == 0)
            on_god = predictions["c05_informal_on_god"]
            on_god_failures = eligible & (on_god != labels)
            neutral = on_god_failures & (on_god == 1)
            contradiction = on_god_failures & (on_god == 2)
            eligible_n = int(eligible.sum())
            failure_n = int(on_god_failures.sum())
            destination_rows.append(
                {
                    "status": "complete",
                    "primary": "false",
                    "model": model,
                    "dataset": dataset,
                    "split": split,
                    "condition_id": "c05_informal_on_god",
                    "scope": "model_clean_correct_entailment",
                    "eligible_denominator": eligible_n,
                    "failures": failure_n,
                    "failure_rate_eligible": failure_n / eligible_n if eligible_n else math.nan,
                    "neutral_failures": int(neutral.sum()),
                    "contradiction_failures": int(contradiction.sum()),
                    "neutral_rate_eligible": float(neutral.sum() / eligible_n) if eligible_n else math.nan,
                    "contradiction_rate_eligible": float(contradiction.sum() / eligible_n) if eligible_n else math.nan,
                    "neutral_rate_failures": float(neutral.sum() / failure_n) if failure_n else math.nan,
                    "contradiction_rate_failures": float(contradiction.sum() / failure_n) if failure_n else math.nan,
                }
            )
            for full, comparator in COMPONENT_PAIRS:
                add_contrast(
                    contrast_rows,
                    family="primary_phrase_component_failure",
                    primary=True,
                    endpoint="clean_correct_entailment_failure",
                    model=model,
                    dataset=dataset,
                    split=split,
                    full=full,
                    comparator=comparator,
                    eligible=eligible,
                    full_event=predictions[full] != labels,
                    comparator_event=predictions[comparator] != labels,
                )
                all_sources = np.ones(len(labels), dtype=bool)
                add_contrast(
                    contrast_rows,
                    family="secondary_phrase_component_accuracy",
                    primary=False,
                    endpoint="unconditional_accuracy",
                    model=model,
                    dataset=dataset,
                    split=split,
                    full=full,
                    comparator=comparator,
                    eligible=all_sources,
                    full_event=predictions[full] == labels,
                    comparator_event=predictions[comparator] == labels,
                )
            for comparator in ON_GOD_COMPONENTS:
                add_contrast(
                    contrast_rows,
                    family="primary_on_god_neutral_destination",
                    primary=True,
                    endpoint="clean_correct_entailment_to_neutral",
                    model=model,
                    dataset=dataset,
                    split=split,
                    full="c05_informal_on_god",
                    comparator=comparator,
                    eligible=eligible,
                    full_event=predictions["c05_informal_on_god"] == 1,
                    comparator_event=predictions[comparator] == 1,
                )

            for full in FULL_PHRASES:
                for comparator in TWO_TOKEN_CONTROLS:
                    add_contrast(
                        contrast_rows,
                        family="secondary_full_phrase_same_length_control",
                        primary=False,
                        endpoint="clean_correct_entailment_failure",
                        model=model,
                        dataset=dataset,
                        split=split,
                        full=full,
                        comparator=comparator,
                        eligible=eligible,
                        full_event=predictions[full] != labels,
                        comparator_event=predictions[comparator] != labels,
                    )
                    all_sources = np.ones(len(labels), dtype=bool)
                    add_contrast(
                        contrast_rows,
                        family="secondary_full_phrase_same_length_control",
                        primary=False,
                        endpoint="unconditional_accuracy",
                        model=model,
                        dataset=dataset,
                        split=split,
                        full=full,
                        comparator=comparator,
                        eligible=all_sources,
                        full_event=predictions[full] == labels,
                        comparator_event=predictions[comparator] == labels,
                    )

            informal_ids = [
                row["condition_id"]
                for row in registry
                if row["condition_type"].startswith("informal_")
            ]
            for scope, fingerprint_eligible in (
                ("clean_correct_all_labels", clean_correct),
                ("clean_correct_entailment", eligible),
            ):
                denominator = int(fingerprint_eligible.sum())
                for condition_id in informal_ids:
                    prediction = predictions[condition_id]
                    failures = fingerprint_eligible & (prediction != labels)
                    neutral_failures = failures & (prediction == 1)
                    fingerprint_rows.append(
                        {
                            "status": "complete",
                            "model": model,
                            "dataset": dataset,
                            "split": split,
                            "scope": scope,
                            "condition_id": condition_id,
                            "eligible_denominator": denominator,
                            "failures": int(failures.sum()),
                            "failure_rate": float(failures.sum() / denominator) if denominator else math.nan,
                            "neutral_destinations": int(neutral_failures.sum()),
                            "neutral_rate": float(neutral_failures.sum() / denominator) if denominator else math.nan,
                            "model_rank": "",
                            "model_minus_five_model_mean_pp": "",
                        }
                    )

    grouped_fingerprints: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fingerprint_rows:
        grouped_fingerprints[
            (row["dataset"], row["split"], row["scope"], row["condition_id"])
        ].append(row)
    for rows in grouped_fingerprints.values():
        mean = float(np.mean([float(row["failure_rate"]) for row in rows]))
        ordered = sorted(rows, key=lambda row: (-float(row["failure_rate"]), row["model"]))
        for rank, row in enumerate(ordered, start=1):
            row["model_rank"] = rank
            row["model_minus_five_model_mean_pp"] = (
                float(row["failure_rate"]) - mean
            ) * 100.0

    for dataset, split in SUITES:
        cells = [data[(model, dataset, split)] for model in MODELS]
        labels = cells[0]["labels"]
        clean_common = np.logical_and.reduce(
            [cell["predictions"]["c00_clean"] == labels for cell in cells]
        )
        for scope, eligible in (
            ("five_model_common_clean_correct_all_labels", clean_common),
            ("five_model_common_clean_correct_entailment", clean_common & (labels == 0)),
        ):
            n = int(eligible.sum())
            for condition in registry:
                condition_id = condition["condition_id"]
                prediction_matrix = np.stack(
                    [cell["predictions"][condition_id] for cell in cells]
                )[:, eligible]
                failure_matrix = prediction_matrix != labels[eligible][None, :]
                all_wrong = np.all(failure_matrix, axis=0)
                same_destination = all_wrong & np.all(
                    prediction_matrix == prediction_matrix[0:1, :], axis=0
                )
                margins = [int(value) for value in failure_matrix.sum(axis=1)]
                marginal_rates = [value / n if n else math.nan for value in margins]
                expected_rate = float(np.prod(marginal_rates)) if n else math.nan
                expected_count = n * expected_rate if n else math.nan
                observed = int(all_wrong.sum())
                ratio = observed / expected_count if expected_count > 0 else math.nan
                key = f"cofailure|{scope}|{dataset}|{split}|{condition_id}"
                joint_bootstrap = bootstrap_joint_metrics(failure_matrix, key)
                same_low, same_high = bootstrap_mean_ci(
                    same_destination.astype(float), key + "|same_destination"
                )
                same_by_destination = [
                    int(np.sum(same_destination & (prediction_matrix[0] == destination)))
                    for destination in range(3)
                ]
                if scope == "five_model_common_clean_correct_entailment":
                    fixed_margin_p: float | str = exact_intersection_upper_tail(
                        n, margins, observed
                    )
                    fixed_margin_status = (
                        "underflow_or_below_reporting_floor_p_lt_1e-300"
                        if fixed_margin_p < 1e-300
                        else "finite"
                    )
                    fixed_margin_null = (
                        "independent_uniform_failure_subsets_conditional_on_model_margins"
                    )
                else:
                    fixed_margin_p = ""
                    fixed_margin_status = "not_targeted_all_label_scope"
                    fixed_margin_null = ""
                cofailure_rows.append(
                    {
                        "status": "complete",
                        "record_type": "condition_summary",
                        "primary": "false",
                        "scope": scope,
                        "dataset": dataset,
                        "split": split,
                        "condition_id": condition_id,
                        "comparator_condition": "",
                        "eligible_denominator": n,
                        "model_failure_margins_json": json.dumps(dict(zip(MODELS, margins)), sort_keys=True),
                        "model_failure_rates_json": json.dumps(dict(zip(MODELS, marginal_rates)), sort_keys=True),
                        "observed_joint_failures": observed,
                        "observed_joint_rate": observed / n if n else math.nan,
                        "observed_joint_rate_ci_low": joint_bootstrap["joint_rate"][0],
                        "observed_joint_rate_ci_high": joint_bootstrap["joint_rate"][1],
                        "independence_expected_failures": expected_count,
                        "independence_expected_failures_ci_low": joint_bootstrap["expected_count"][0],
                        "independence_expected_failures_ci_high": joint_bootstrap["expected_count"][1],
                        "observed_minus_independence_expected_failures": observed - expected_count,
                        "observed_minus_independence_expected_failures_ci_low": joint_bootstrap["excess_count"][0],
                        "observed_minus_independence_expected_failures_ci_high": joint_bootstrap["excess_count"][1],
                        "observed_expected_ratio": ratio,
                        "observed_expected_ratio_ci_low": joint_bootstrap["ratio"][0],
                        "observed_expected_ratio_ci_high": joint_bootstrap["ratio"][1],
                        "observed_expected_status": "defined" if expected_count > 0 else "undefined_zero_expected",
                        "fixed_margin_intersection_p_upper": fixed_margin_p,
                        "fixed_margin_p_status": fixed_margin_status,
                        "fixed_margin_holm_p_72": "",
                        "fixed_margin_null": fixed_margin_null,
                        "synchronized_destination_failures": int(same_destination.sum()),
                        "synchronized_destination_rate": float(same_destination.mean()) if n else math.nan,
                        "synchronized_destination_rate_ci_low": same_low,
                        "synchronized_destination_rate_ci_high": same_high,
                        "synchronized_entailment_count": same_by_destination[0],
                        "synchronized_neutral_count": same_by_destination[1],
                        "synchronized_contradiction_count": same_by_destination[2],
                        "model_a": "",
                        "model_b": "",
                        "pair_joint_failures": "",
                        "pair_expected_rate": "",
                        "pair_excess_joint_risk": "",
                        "pair_excess_joint_risk_ci_low": "",
                        "pair_excess_joint_risk_ci_high": "",
                        "odds_ratio": "",
                        "odds_ratio_ci_low": "",
                        "odds_ratio_ci_high": "",
                        "continuity_correction": "",
                        "primary_full_events": "",
                        "primary_comparator_events": "",
                        "effect_pp": "",
                        "bootstrap_ci_low_pp": "",
                        "bootstrap_ci_high_pp": "",
                        "discordant_full_only": "",
                        "discordant_comparator_only": "",
                        "mcnemar_p_raw": "",
                        "holm_p": "",
                    }
                )

        for left in range(len(MODELS)):
            for right in range(left + 1, len(MODELS)):
                pair_clean = (
                    (cells[left]["predictions"]["c00_clean"] == labels)
                    & (cells[right]["predictions"]["c00_clean"] == labels)
                )
                for scope, pair_eligible in (
                    ("pairwise_common_clean_correct_all_labels", pair_clean),
                    ("pairwise_common_clean_correct_entailment", pair_clean & (labels == 0)),
                ):
                    n = int(pair_eligible.sum())
                    for condition in registry:
                        condition_id = condition["condition_id"]
                        a = (cells[left]["predictions"][condition_id] != labels)[pair_eligible]
                        b = (cells[right]["predictions"][condition_id] != labels)[pair_eligible]
                        n11 = int(np.sum(a & b))
                        n10 = int(np.sum(a & ~b))
                        n01 = int(np.sum(~a & b))
                        n00 = int(np.sum(~a & ~b))
                        corrected = any(value == 0 for value in (n11, n10, n01, n00))
                        table = (
                            [value + 0.5 for value in (n11, n10, n01, n00)]
                            if corrected
                            else [n11, n10, n01, n00]
                        )
                        odds = (table[0] * table[3]) / (table[1] * table[2])
                        expected_rate = float(a.mean() * b.mean()) if n else math.nan
                        joint_rate = n11 / n if n else math.nan
                        pair_bootstrap = bootstrap_pair_metrics(
                            a,
                            b,
                            f"pair|{scope}|{dataset}|{split}|{condition_id}|{MODELS[left]}|{MODELS[right]}",
                        )
                        cofailure_rows.append(
                            {
                                "status": "complete",
                                "record_type": "pairwise",
                                "primary": "false",
                                "scope": scope,
                                "dataset": dataset,
                                "split": split,
                                "condition_id": condition_id,
                                "comparator_condition": "",
                                "eligible_denominator": n,
                                "model_failure_margins_json": "",
                                "model_failure_rates_json": "",
                                "observed_joint_failures": "",
                                "observed_joint_rate": "",
                                "fixed_margin_intersection_p_upper": "",
                                "fixed_margin_p_status": "not_applicable_pairwise",
                                "fixed_margin_holm_p_72": "",
                                "fixed_margin_null": "",
                                "synchronized_destination_failures": "",
                                "synchronized_destination_rate": "",
                                "model_a": MODELS[left],
                                "model_b": MODELS[right],
                                "pair_joint_failures": n11,
                                "pair_expected_rate": expected_rate,
                                "pair_excess_joint_risk": joint_rate - expected_rate,
                                "pair_excess_joint_risk_ci_low": pair_bootstrap["excess_risk"][0],
                                "pair_excess_joint_risk_ci_high": pair_bootstrap["excess_risk"][1],
                                "odds_ratio": odds,
                                "odds_ratio_ci_low": pair_bootstrap["odds_ratio"][0],
                                "odds_ratio_ci_high": pair_bootstrap["odds_ratio"][1],
                                "continuity_correction": str(corrected).lower(),
                                "primary_full_events": "",
                                "primary_comparator_events": "",
                                "effect_pp": "",
                                "bootstrap_ci_low_pp": "",
                                "bootstrap_ci_high_pp": "",
                                "discordant_full_only": "",
                                "discordant_comparator_only": "",
                                "mcnemar_p_raw": "",
                                "holm_p": "",
                            }
                        )

        eligible = clean_common & (labels == 0)
        n = int(eligible.sum())

        on_god_matrix = np.stack(
            [cell["predictions"]["c05_informal_on_god"] for cell in cells]
        )
        on_god_all_wrong = np.all(on_god_matrix != labels[None, :], axis=0)
        for comparator in TWO_TOKEN_CONTROLS:
            comparator_matrix = np.stack(
                [cell["predictions"][comparator] for cell in cells]
            )
            comparator_all_wrong = np.all(comparator_matrix != labels[None, :], axis=0)
            a = on_god_all_wrong[eligible]
            b = comparator_all_wrong[eligible]
            differences = (a.astype(np.int8) - b.astype(np.int8)).astype(float) * 100.0
            low, high = bootstrap_mean_ci(
                differences,
                f"primary_all_wrong|{dataset}|{split}|{comparator}",
            )
            p_value, full_only, comparator_only = exact_mcnemar(a, b)
            cofailure_rows.append(
                {
                    "status": "complete",
                    "record_type": "primary_all_wrong_control_contrast",
                    "primary": "true",
                    "scope": "five_model_common_clean_correct_entailment",
                    "dataset": dataset,
                    "split": split,
                    "condition_id": "c05_informal_on_god",
                    "comparator_condition": comparator,
                    "eligible_denominator": n,
                    "model_failure_margins_json": "",
                    "model_failure_rates_json": "",
                    "observed_joint_failures": "",
                    "observed_joint_rate": "",
                    "fixed_margin_intersection_p_upper": "",
                    "fixed_margin_p_status": "not_applicable_paired_primary_contrast",
                    "fixed_margin_holm_p_72": "",
                    "fixed_margin_null": "",
                    "synchronized_destination_failures": "",
                    "synchronized_destination_rate": "",
                    "primary_full_events": int(a.sum()),
                    "primary_comparator_events": int(b.sum()),
                    "effect_pp": float(differences.mean()) if differences.size else math.nan,
                    "bootstrap_ci_low_pp": low,
                    "bootstrap_ci_high_pp": high,
                    "discordant_full_only": full_only,
                    "discordant_comparator_only": comparator_only,
                    "mcnemar_p_raw": p_value,
                    "holm_p": "",
                }
            )

    secondary_fixed_margin = [
        row for row in cofailure_rows
        if row["record_type"] == "condition_summary"
        and row["scope"] == "five_model_common_clean_correct_entailment"
    ]
    if len(secondary_fixed_margin) != 72:
        raise RuntimeError(
            f"Expected 72 split/condition fixed-margin tests, found {len(secondary_fixed_margin)}"
        )
    fixed_adjusted = holm_adjust(
        [float(row["fixed_margin_intersection_p_upper"]) for row in secondary_fixed_margin]
    )
    for row, adjusted_p in zip(secondary_fixed_margin, fixed_adjusted):
        row["fixed_margin_holm_p_72"] = adjusted_p

    primary_locations = [
        ("contrast", index, float(row["mcnemar_p_raw"]))
        for index, row in enumerate(contrast_rows)
        if row["primary"] == "true"
    ] + [
        ("cofailure", index, float(row["mcnemar_p_raw"]))
        for index, row in enumerate(cofailure_rows)
        if row["primary"] == "true"
    ]
    if len(primary_locations) != FROZEN_PRIMARY_TESTS:
        raise RuntimeError(
            f"Frozen family must contain {FROZEN_PRIMARY_TESTS} tests; found {len(primary_locations)}"
        )
    adjusted = holm_adjust([item[2] for item in primary_locations])
    for (kind, index, _), value in zip(primary_locations, adjusted):
        target = contrast_rows if kind == "contrast" else cofailure_rows
        target[index]["holm_p"] = value
    if len(destination_rows) != 15:
        raise RuntimeError(f"Expected 15 denominator-specific destination rows, found {len(destination_rows)}")
    expected_counts = {
        "condition_effects": (len(condition_rows), 360),
        "contrasts": (len(contrast_rows), 570),
        "cofailure": (len(cofailure_rows), 1596),
        "destinations": (len(destination_rows), 15),
        "fingerprints": (len(fingerprint_rows), 270),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in expected_counts.items()
        if observed != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen output topology mismatch: {mismatches}")
    return (
        condition_rows,
        contrast_rows,
        cofailure_rows,
        destination_rows,
        fingerprint_rows,
    )


CONDITION_FIELDS = (
    "status", "model", "dataset", "split", "condition_id", "condition_type",
    "denominator", "accuracy", "clean_accuracy", "transformed_minus_clean_pp",
    "bootstrap_ci_low_pp", "bootstrap_ci_high_pp", "clean_correct_denominator",
    "failures", "failure_rate", "failure_rate_ci_low", "failure_rate_ci_high",
    "destination_entailment_count", "destination_neutral_count",
    "destination_contradiction_count", "destination_entailment_rate_clean_correct",
    "destination_neutral_rate_clean_correct", "destination_contradiction_rate_clean_correct",
    "destination_entailment_rate_failures", "destination_neutral_rate_failures",
    "destination_contradiction_rate_failures",
)
CONTRAST_FIELDS = (
    "status", "family", "primary", "endpoint", "model", "dataset", "split",
    "full_condition", "comparator_condition", "eligible_denominator", "full_events",
    "comparator_events", "effect_pp", "bootstrap_ci_low_pp", "bootstrap_ci_high_pp",
    "discordant_full_only", "discordant_comparator_only", "mcnemar_p_raw", "holm_p",
)
COFAILURE_FIELDS = (
    "status", "record_type", "primary", "scope", "dataset", "split", "condition_id",
    "comparator_condition", "eligible_denominator", "model_failure_margins_json",
    "model_failure_rates_json", "observed_joint_failures", "observed_joint_rate",
    "observed_joint_rate_ci_low", "observed_joint_rate_ci_high",
    "independence_expected_failures", "independence_expected_failures_ci_low",
    "independence_expected_failures_ci_high", "observed_minus_independence_expected_failures",
    "observed_minus_independence_expected_failures_ci_low",
    "observed_minus_independence_expected_failures_ci_high", "observed_expected_ratio",
    "observed_expected_ratio_ci_low", "observed_expected_ratio_ci_high",
    "observed_expected_status", "fixed_margin_intersection_p_upper", "fixed_margin_p_status",
    "fixed_margin_holm_p_72", "fixed_margin_null",
    "synchronized_destination_failures", "synchronized_destination_rate",
    "synchronized_destination_rate_ci_low", "synchronized_destination_rate_ci_high",
    "synchronized_entailment_count", "synchronized_neutral_count",
    "synchronized_contradiction_count", "model_a", "model_b", "pair_joint_failures",
    "pair_expected_rate", "pair_excess_joint_risk", "pair_excess_joint_risk_ci_low",
    "pair_excess_joint_risk_ci_high", "odds_ratio", "odds_ratio_ci_low",
    "odds_ratio_ci_high", "continuity_correction", "primary_full_events",
    "primary_comparator_events",
    "effect_pp", "bootstrap_ci_low_pp", "bootstrap_ci_high_pp", "discordant_full_only",
    "discordant_comparator_only", "mcnemar_p_raw", "holm_p",
)
DESTINATION_FIELDS = (
    "status", "primary", "model", "dataset", "split", "condition_id", "scope",
    "eligible_denominator", "failures", "failure_rate_eligible", "neutral_failures",
    "contradiction_failures", "neutral_rate_eligible", "contradiction_rate_eligible",
    "neutral_rate_failures", "contradiction_rate_failures",
)
FINGERPRINT_FIELDS = (
    "status", "model", "dataset", "split", "scope", "condition_id",
    "eligible_denominator", "failures", "failure_rate", "neutral_destinations",
    "neutral_rate", "model_rank", "model_minus_five_model_mean_pp",
)


def format_tail_p(value: float) -> str:
    """Format a floating exact tail without ever presenting underflow as p=0."""

    if value <= 0.0 or value < 1e-300:
        return "<1e-300"
    return f"{value:.3g}"


def publication_report(
    condition_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    cofailure_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    total_decisions: int,
) -> str:
    primary_contrasts = [row for row in contrast_rows if row["primary"] == "true"]
    primary_cofailure = [row for row in cofailure_rows if row["primary"] == "true"]
    all_primary = primary_contrasts + primary_cofailure
    significant = sum(float(row["holm_p"]) < 0.05 for row in all_primary)
    phrase = [
        row for row in primary_contrasts
        if row["family"] == "primary_phrase_component_failure"
    ]
    neutral = [
        row for row in primary_contrasts
        if row["family"] == "primary_on_god_neutral_destination"
    ]
    on_god_effects = [
        row for row in condition_rows if row["condition_id"] == "c05_informal_on_god"
    ]
    replay_lines = []
    for row in replay_rows:
        replay = row["clean_replay"]
        replay_lines.append(
            f"- {row['dataset']} / {row['split']}: {replay['matched']}/{replay['total']} "
            f"({100.0 * replay['agreement']:.3f}%) clean predictions match the archived run; "
            f"accuracy deviation {100.0 * replay['accuracy_absolute_deviation']:.4f} pp."
        )
    summary_rows = {
        (row["dataset"], row["split"]): row
        for row in cofailure_rows
        if row["record_type"] == "condition_summary"
        and row["scope"] == "five_model_common_clean_correct_entailment"
        and row["condition_id"] == "c05_informal_on_god"
    }
    cofailure_lines = []
    for dataset, split in SUITES:
        row = summary_rows[(dataset, split)]
        cofailure_lines.append(
            f"| {dataset}:{split} | {row['eligible_denominator']} | "
            f"{row['observed_joint_failures']} ({100.0 * float(row['observed_joint_rate']):.2f}%) | "
            f"{float(row['independence_expected_failures']):.2f} | "
            f"{float(row['observed_expected_ratio']):.2f} | "
            f"{row['synchronized_destination_failures']} | "
            f"{format_tail_p(float(row['fixed_margin_intersection_p_upper']))} | "
            f"{format_tail_p(float(row['fixed_margin_holm_p_72']))} |"
        )
    fixed_margin_lines = []
    fixed_targets = {"c05_informal_on_god", *TWO_TOKEN_CONTROLS}
    for dataset, split in SUITES:
        target_rows = [
            row for row in cofailure_rows
            if row["record_type"] == "condition_summary"
            and row["scope"] == "five_model_common_clean_correct_entailment"
            and row["dataset"] == dataset
            and row["split"] == split
            and row["condition_id"] in fixed_targets
        ]
        target_by_id = {row["condition_id"]: row for row in target_rows}
        for condition_id in ("c05_informal_on_god", *TWO_TOKEN_CONTROLS):
            row = target_by_id[condition_id]
            fixed_margin_lines.append(
                f"| {dataset}:{split} | `{condition_id}` | "
                f"{row['observed_joint_failures']}/{row['eligible_denominator']} | "
                f"{format_tail_p(float(row['fixed_margin_intersection_p_upper']))} | "
                f"{format_tail_p(float(row['fixed_margin_holm_p_72']))} |"
            )
    phrase_positive = sum(float(row["effect_pp"]) > 0 and float(row["holm_p"]) < 0.05 for row in phrase)
    phrase_reverse = sum(float(row["effect_pp"]) < 0 and float(row["holm_p"]) < 0.05 for row in phrase)
    neutral_positive = sum(float(row["effect_pp"]) > 0 and float(row["holm_p"]) < 0.05 for row in neutral)
    neutral_reverse = sum(float(row["effect_pp"]) < 0 and float(row["holm_p"]) < 0.05 for row in neutral)
    control_positive = sum(
        float(row["effect_pp"]) > 0 and float(row["holm_p"]) < 0.05
        for row in primary_cofailure
    )
    control_reverse = sum(
        float(row["effect_pp"]) < 0 and float(row["holm_p"]) < 0.05
        for row in primary_cofailure
    )
    return f"""# Tokenizer-corrected phrase/component counterfactual report

## Audit status

All 15 model/split cells passed artifact-hash, completion-sidecar, row-schema,
condition-order, source-index, label-crossing, and five-model support checks.
Destination semantics use the explicitly asserted frozen class order
0=entailment, 1=neutral, and 2=contradiction.
The analysis covers {total_decisions:,} model-example decisions: 12 untouched,
validated v1 non-BERTweet cells plus three isolated tokenizer-corrected BERTweet
cells. The invalid v1 BERTweet artifacts are never read.

Corrected BERTweet clean replay:

{chr(10).join(replay_lines)}

## Frozen primary family

The pre-inference family contains 132 two-sided exact McNemar tests: 90
phrase-versus-component failure contrasts, 30 `on god` neutral-destination
contrasts, and 12 all-five `on god`-versus-control cofailure contrasts. All
132 raw p-values are adjusted together by Holm; 95% intervals use 5,000
deterministic source bootstraps (seed 20260726). {significant}/132 tests are
Holm-significant at 0.05.

- Phrase/component failure contrasts: {phrase_positive}/90 significant in the
  predicted positive direction, {phrase_reverse}/90 significant in the reverse
  direction, and {len(phrase) - phrase_positive - phrase_reverse}/90 null after
  Holm correction.
- `on god` neutral-destination contrasts: {neutral_positive}/30 significant in
  the predicted positive direction, {neutral_reverse}/30 significant in reverse,
  and {len(neutral) - neutral_positive - neutral_reverse}/30 null after Holm.
- All-five `on god`/control contrasts: {control_positive}/12 significant in the
  predicted direction, {control_reverse}/12 significant in reverse, and
  {len(primary_cofailure) - control_positive - control_reverse}/12 null after Holm.
- Across the 15 model/split cells, unconditional `on god` accuracy change ranges
  from {min(float(row['transformed_minus_clean_pp']) for row in on_god_effects):.3f}
  to {max(float(row['transformed_minus_clean_pp']) for row in on_god_effects):.3f}
  percentage points.

## All-five cofailure on clean-correct entailments

| suite | eligible | all five wrong | product-marginal expected | O/E | same wrong destination | fixed-margin raw p | Holm-72 p |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(cofailure_lines)}

The product-of-marginals expectation and O/E ratio are descriptive. The
fixed-margin p-value conditions on each model's observed failure count and
computes the upper tail of the five-way intersection under independently and
uniformly located failure subsets using recursive hypergeometric convolution.
It tests excess overlap under that conditional null; it does not make the
shared checkpoints independent, explain the mechanism, or establish causality.
Exploratory fixed-margin p-values for every condition, including `on god` and all four
prespecified two-token controls, are in `cofailure_results_v2.csv`; their
Holm-72 values adjust the complete 24-condition by 3-split secondary family.
This secondary adjustment is separate from and does not alter the frozen 132
paired-test family.

| suite | condition | all five wrong / eligible | fixed-margin raw p | Holm-72 p |
|---|---|---:|---:|---:|
{chr(10).join(fixed_margin_lines)}

The 15 secondary model/split destination rows are in
`on_god_destination_v2.csv`. Each row is restricted to that model's own
clean-correct entailments and reports both eligible-denominator and
failure-denominator rates; these denominators must not be pooled or exchanged.

## Selection and interpretation

`on god` was selected from a retrospective result on these same released final
suites. The primary family, directions, source support, conditions, and controls
were frozen before the original 15 prediction jobs; the v2 tokenizer repair
changes only BERTweet tokenization and inherits that family unchanged. This is
therefore a targeted prospective replay on fully crossed support, not an
independent-dataset confirmation. Component rows are synthetic lexical
ablations and are not natural or human-validated meaning-preserving examples.

The frozen secondary topology is retained in full: 450 secondary paired
contrasts share `phrase_component_contrasts_v2.csv` with the 120 primary rows;
both five-model scopes and all ten pairwise model combinations are in
`cofailure_results_v2.csv`; and 270 descriptive heterogeneous-response rows are
in `model_phrase_fingerprints_v2.csv`.
"""


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.min_clean_replay_agreement <= 1.0):
        raise ValueError("--min-clean-replay-agreement must lie in [0, 1]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data, registry, run, input_audit = load_all_cells(args)
    (
        condition_rows,
        contrast_rows,
        cofailure_rows,
        destination_rows,
        fingerprint_rows,
    ) = compute_results(data, registry)

    condition_path = output_dir / "condition_effects_v2.csv"
    contrast_path = output_dir / "phrase_component_contrasts_v2.csv"
    cofailure_path = output_dir / "cofailure_results_v2.csv"
    destination_path = output_dir / "on_god_destination_v2.csv"
    fingerprint_path = output_dir / "model_phrase_fingerprints_v2.csv"
    report_path = output_dir / "REPORT.md"
    validation_path = output_dir / "validation_v2.json"
    write_csv(condition_path, condition_rows, CONDITION_FIELDS)
    write_csv(contrast_path, contrast_rows, CONTRAST_FIELDS)
    write_csv(cofailure_path, cofailure_rows, COFAILURE_FIELDS)
    write_csv(destination_path, destination_rows, DESTINATION_FIELDS)
    write_csv(fingerprint_path, fingerprint_rows, FINGERPRINT_FIELDS)

    replay_rows = [row for row in input_audit if row["clean_replay"] is not None]
    total_decisions = sum(int(row["prediction_rows"]) for row in input_audit)
    report_path.write_text(
        publication_report(
            condition_rows,
            contrast_rows,
            cofailure_rows,
            replay_rows,
            total_decisions,
        ),
        encoding="utf-8",
    )

    repository = args.source_workspace.resolve().parents[1]
    primary_contrasts = [row for row in contrast_rows if row["primary"] == "true"]
    primary_cofailure = [row for row in cofailure_rows if row["primary"] == "true"]
    all_primary = primary_contrasts + primary_cofailure
    protocol_path = args.source_workspace.resolve() / "RUN_PROTOCOL.md"
    validation = {
        "schema_version": 2,
        "status": "complete",
        "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap": {
            "replicates": FROZEN_BOOTSTRAP_REPLICATES,
            "seed": FROZEN_BOOTSTRAP_SEED,
            "unit": "source_pair_complete_outcome_vector",
        },
        "label_semantics": {
            "mapping": {str(index): name for index, name in LABEL_NAMES.items()},
            "assertion": "frozen_project_class_order",
            "destination_columns_bound_to_mapping": True,
        },
        "input_contract": {
            "validated_v1_non_bertweet_cells": len(
                [row for row in input_audit if row["source"] == "validated_v1"]
            ),
            "corrected_v2_bertweet_cells": len(replay_rows),
            "total_cells": len(input_audit),
            "total_prediction_rows": total_decisions,
            "condition_count": len(registry),
            "source_workspace_manifest_sha256": sha256_file(
                args.source_workspace.resolve() / "run_manifest.json"
            ),
            "source_workspace_manifest_logical_sha256": logical_json_checksum(
                args.source_workspace.resolve() / "run_manifest.json"
            ),
            "condition_registry_sha256": sha256_file(
                args.source_workspace.resolve() / "CONDITION_REGISTRY.csv"
            ),
            "five_model_crossing_passed": True,
            "invalid_v1_bertweet_read": False,
        },
        "selection_provenance": {
            "focal_condition": "c05_informal_on_god",
            "selection_basis": "retrospective_result_on_same_released_final_suites",
            "primary_family_frozen_before_original_predictions": True,
            "tokenizer_repair_changed_primary_family": False,
            "interpretation": "targeted_prospective_replay_not_independent_confirmation",
            "frozen_protocol": portable_path(protocol_path, repository),
            "frozen_protocol_sha256": sha256_file(protocol_path),
        },
        "primary_family": {
            "phrase_component_failure_tests": sum(
                row["family"] == "primary_phrase_component_failure"
                for row in primary_contrasts
            ),
            "on_god_neutral_destination_tests": sum(
                row["family"] == "primary_on_god_neutral_destination"
                for row in primary_contrasts
            ),
            "all_five_cofailure_control_tests": len(primary_cofailure),
            "total_tests": len(all_primary),
            "global_adjustment": "Holm",
            "holm_significant_0_05": sum(float(row["holm_p"]) < 0.05 for row in all_primary),
        },
        "secondary_output_topology": {
            "paired_contrasts": len(
                [row for row in contrast_rows if row["primary"] == "false"]
            ),
            "five_model_condition_summaries": len(
                [row for row in cofailure_rows if row["record_type"] == "condition_summary"]
            ),
            "pairwise_model_rows": len(
                [row for row in cofailure_rows if row["record_type"] == "pairwise"]
            ),
            "pairwise_model_combinations": len(
                {
                    (row["model_a"], row["model_b"])
                    for row in cofailure_rows
                    if row["record_type"] == "pairwise"
                }
            ),
            "model_phrase_fingerprints": len(fingerprint_rows),
            "destination_rows": len(destination_rows),
        },
        "fixed_margin_intersection_test": {
            "status": "post_hoc_exploratory_added_after_independent_audit",
            "tail": "upper",
            "conditioning": "each_model_observed_failure_margin",
            "null": "independent_uniform_failure_subsets_over_eligible_sources",
            "algorithm": "recursive_hypergeometric_convolution",
            "conditions_tested": len(
                [
                    row for row in cofailure_rows
                    if row["record_type"] == "condition_summary"
                    and row["scope"] == "five_model_common_clean_correct_entailment"
                ]
            ),
            "multiplicity_adjustment": "Holm_over_all_24_conditions_x_3_splits",
            "holm_significant_0_05": sum(
                float(row["fixed_margin_holm_p_72"]) < 0.05
                for row in cofailure_rows
                if row["record_type"] == "condition_summary"
                and row["scope"] == "five_model_common_clean_correct_entailment"
            ),
            "values_below_reporting_floor": sum(
                row.get("fixed_margin_p_status")
                == "underflow_or_below_reporting_floor_p_lt_1e-300"
                for row in cofailure_rows
            ),
            "prose_reporting_floor": "p < 1e-300",
            "interpretive_limit": "does_not_remove_shared_checkpoint_dependence_or_establish_causality",
        },
        "clean_replay": [
            {
                "cell_id": row["cell_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                **row["clean_replay"],
            }
            for row in replay_rows
        ],
        "inputs": [
            {
                key: (
                    portable_path(value, repository)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in row.items()
                if key != "clean_replay"
            }
            for row in input_audit
        ],
        "outputs": {
            portable_path(path, repository): {
                "sha256": sha256_file(path),
                "rows": row_count,
            }
            for path, row_count in (
                (condition_path, len(condition_rows)),
                (contrast_path, len(contrast_rows)),
                (cofailure_path, len(cofailure_rows)),
                (destination_path, len(destination_rows)),
                (fingerprint_path, len(fingerprint_rows)),
                (report_path, None),
            )
        },
        "source_run_manifest_primary_family": {
            "pre_execution_logical_sha256": run.get(
                "pre_execution_run_manifest_logical_sha256"
            ),
            "approval_granted": run.get("approval_granted"),
        },
    }
    if validation["primary_family"]["total_tests"] != FROZEN_PRIMARY_TESTS:
        raise RuntimeError("Validation family-size assertion failed")
    if validation["input_contract"]["validated_v1_non_bertweet_cells"] != 12:
        raise RuntimeError("Validation requires exactly 12 legacy cells")
    if validation["input_contract"]["corrected_v2_bertweet_cells"] != 3:
        raise RuntimeError("Validation requires exactly three corrected BERTweet cells")
    expected_secondary_topology = {
        "paired_contrasts": 450,
        "five_model_condition_summaries": 144,
        "pairwise_model_rows": 1440,
        "pairwise_model_combinations": 10,
        "model_phrase_fingerprints": 270,
        "destination_rows": 15,
    }
    if validation["secondary_output_topology"] != expected_secondary_topology:
        raise RuntimeError(
            "Validation secondary-topology assertion failed: "
            f"{validation['secondary_output_topology']}"
        )
    write_json_with_checksum(validation_path, validation)
    print(
        json.dumps(
            {
                "status": "complete",
                "primary_tests": FROZEN_PRIMARY_TESTS,
                "holm_significant": validation["primary_family"]["holm_significant_0_05"],
                "validation": str(validation_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
