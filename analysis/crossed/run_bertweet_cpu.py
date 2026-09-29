#!/usr/bin/env python3
"""Rerun the three BERTweet counterfactual cells with the base tokenizer.

The original v1 inference runner reloaded BERTweet's tokenizer from each
fine-tuned ``final_model`` directory.  With transformers 5.13.0, the saved
``bpe.codes`` file cannot be round-tripped by ``BertweetTokenizer`` and loses
its merge table on reload.  This correction keeps the frozen model weights and
evaluation matrices, but instantiates the slow tokenizer explicitly from the
pinned ``vinai/bertweet-base`` snapshot used for training.

This program is deliberately offline and defaults to CPU, with an explicit
CUDA mode for the isolated H100 correction run.  It never modifies the v1
workspace.  A successful run writes a deterministic gzip JSONL artifact and a
content-addressed completion record under this directory's ``predictions/``.
The temporary prediction is not published unless clean-condition predictions
replay the archived baseline at >=99.5% agreement and <=0.2 percentage-point
accuracy deviation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


WORKSPACE = Path(__file__).resolve().parent
SOURCE_WORKSPACE = WORKSPACE.parent / "phrase_component_counterfactuals"
REPOSITORY_ROOT = WORKSPACE.parents[1]
DEFAULT_CHECKPOINT_ROOT = (
    REPOSITORY_ROOT
    / "modal_backup_2026-07-26"
    / "volumes"
    / "wnut2026-nli-results-v2"
)
TOKENIZER_REVISION = "b349c1243407b0dcffeabb2337497477286e27ab"
TOKENIZER_REPOSITORY = "vinai/bertweet-base"
DEFAULT_TOKENIZER_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--vinai--bertweet-base"
    / "snapshots"
    / TOKENIZER_REVISION
)
PREDICTIONS = WORKSPACE / "predictions"
DEFAULT_MNLI_WEIGHTS_OVERRIDE = WORKSPACE / "remote_mnli_model.safetensors"

SOURCE_RUN_MANIFEST_LOGICAL_SHA256 = (
    "614de061e854e14789553be479366f2f78db95434147e9d72e00fd19ac64ac8e"
)
CONDITION_REGISTRY_SHA256 = (
    "038c12ca2ee0bc354ec10f01aa924b2b7e3b77250bb9a2ec492f66eb299a4676"
)
TOKENIZER_PACKAGE_SHA256 = (
    "934aa09df86b70e9ca86aace92f7dc58d276fc906ae9c6373584fdf49038094f"
)
TOKENIZER_FILE_HASHES = {
    "bpe.codes": "77712739cd1a7f638e6694b0dd832494e4f66e3d05c709fc6a6a2f988ff9e589",
    "config.json": "7926dbeefbaabac88352b291f86c2363b5d54164b8e67437ea0edae7010257a6",
    "tokenizer.json": "48a8972b321c93163b78d98f40bb410d898d8869d8439b6abb5f8283f545b85d",
    "vocab.txt": "d3f3d56ed440cdb39bd60a76884b67e1061abca462146fdcc751f1ee40ae9ed3",
}
MNLI_WEIGHTS_SHA256 = (
    "d5715e15a66ebbfac5724a66d73680186d82d24f7f15ce3063b45878638edeba"
)
CONDITION_IDS = [
    "c00_clean",
    "c01_informal_fr",
    "c02_informal_tbh",
    "c03_informal_ngl",
    "c04_informal_real_talk",
    "c05_informal_on_god",
    "c06_informal_istg",
    "c07_informal_frfr",
    "c08_informal_deadass",
    "c09_informal_no_cap",
    "c10_component_real",
    "c11_component_talk",
    "c12_component_on",
    "c13_component_god",
    "c14_component_no",
    "c15_component_cap",
    "c16_formal_honestly",
    "c17_formal_seriously",
    "c18_formal_in_fact",
    "c19_random_nearby",
    "c20_random_outside",
    "c21_random_in_the",
    "c22_random_with_it",
    "c23_random_at_this",
]

# Every identity below is copied from the frozen v1 manifests or the archived
# seed-42 prediction bundle.  Keeping it here makes the correction independently
# auditable and prevents a path with the right name but wrong bytes from running.
CELL_SPECS: dict[str, dict[str, Any]] = {
    "bertweet__snli__test": {
        "dataset": "snli",
        "split": "test",
        "source_cell_identity_sha256": "87065ce076b8149487900b6f6b2ff5424fda34884967f1bebcc5f8604deb7979",
        "source_pairs": 9824,
        "evaluation_rows": 235776,
        "evaluation_artifact_path": "eval_data/snli__final__test.jsonl.gz",
        "evaluation_artifact_sha256": "5ec58deb929bf2421305b2798c6b323b89e37d4ea04b9ced493f8c153ea0cad2",
        "evaluation_logical_checksum_sha256": "0c38889d6c6d4f8acc7a7efb23a88e3eb6cd13ec06a38ca4f41549c0130130b5",
        "checkpoint_path": "publication_results/snli/seed_42/bertweet_baseline_final_test/final_model",
        "checkpoint_package_sha256": "6cd32a40300a161f9e103c7ddb9ea628f4d9f89c44440e5e662f4cf2eed37709",
        "reference_prediction_path": "publication_results/snli/seed_42/bertweet_baseline_final_test/predictions.json",
        "reference_prediction_sha256": "e0c25d9d9d4f96c776bb1a4be0b7a90cd6505b696abbed28baf84532ad358878",
        "reference_original_logical_sha256": "c1abac24962382246a9ff433b7772c82dd23c81abe20ea34057d5e27176b9b50",
    },
    "bertweet__multi_nli__validation_matched": {
        "dataset": "multi_nli",
        "split": "validation_matched",
        "source_cell_identity_sha256": "ea0173ab3d102b88801a6ad76a38b9a09209435a32ef646e95d195f3df699647",
        "source_pairs": 9815,
        "evaluation_rows": 235560,
        "evaluation_artifact_path": "eval_data/multi_nli__final__validation_matched.jsonl.gz",
        "evaluation_artifact_sha256": "281ca9a824f6320318be8a87b0ce578bb9e381359784c8e7cbbf2a94772d3732",
        "evaluation_logical_checksum_sha256": "2c0eb6d4fb29027ef975c12e40f7fb564ae20639fe6ae201199f9f401097e4ba",
        "checkpoint_path": "publication_results/multi_nli/seed_42/bertweet_baseline_final_validation_matched/final_model",
        "checkpoint_package_sha256": "38353dabfed957351e0d68cbe74f4f66be449a6e487608d96b19c1f6eabc751b",
        "reference_prediction_path": "publication_results/multi_nli/seed_42/bertweet_baseline_final_validation_matched/predictions.json",
        "reference_prediction_sha256": "a939c702a2a3c90e54a5253312c2dd32cb17465345676f9d8ef89563b11aa7dc",
        "reference_original_logical_sha256": "716ff0953eeadbbace32c64d221c42435f88e2f2031b19679c2f4ec793c07e59",
    },
    "bertweet__multi_nli__validation_mismatched": {
        "dataset": "multi_nli",
        "split": "validation_mismatched",
        "source_cell_identity_sha256": "45f5b4854b30c81547ed6b14f1af69f6508f4be05aa9648e05f7fc1f5db5a2b9",
        "source_pairs": 9832,
        "evaluation_rows": 235968,
        "evaluation_artifact_path": "eval_data/multi_nli__final__validation_mismatched.jsonl.gz",
        "evaluation_artifact_sha256": "c1f4371e07a7377abc967c236c0445175a1957e8e25f7cd2d48f54bfdce95d67",
        "evaluation_logical_checksum_sha256": "6f817d60da24e81d5bffec0f346dd1dbaf6793ad1733f5f23597aae728d15f82",
        "checkpoint_path": "publication_results/multi_nli/seed_42/bertweet_baseline_final_validation_matched/final_model",
        "checkpoint_package_sha256": "38353dabfed957351e0d68cbe74f4f66be449a6e487608d96b19c1f6eabc751b",
        "reference_prediction_path": "publication_results/multi_nli/seed_42/bertweet_baseline_final_validation_mismatched/predictions.json",
        "reference_prediction_sha256": "2704f8bb0944bcc8721da20f37a311b9def12ec51123101b5ca5f47f47c54540",
        "reference_original_logical_sha256": "de1675cc1b00b42a83e56a431620334bf9f23a6074a6c1d5b10aa48a1307fb51",
    },
}

CLEAN_REPLAY_AGREEMENT_THRESHOLD = 0.995
CLEAN_REPLAY_ACCURACY_DEVIATION_THRESHOLD = 0.002
MAX_LENGTH = 128


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_json_checksum(path: Path) -> str:
    return hashlib.sha256(canonical_json(json.loads(path.read_text())).encode()).hexdigest()


def logical_jsonl_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                digest.update(canonical_json(json.loads(line)).encode("utf-8") + b"\n")
    return digest.hexdigest()


def package_inventory(
    path: Path,
    file_overrides: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Hash a package, optionally substituting verified bytes for logical files.

    The downloaded local MNLI backup contains the six correct small files but a
    stale ``model.safetensors``.  The exact remote weight file is stored
    separately and read-only.  Treating it as the logical package member lets us
    verify the original seven-file package hash without copying or modifying the
    v1 backup directory.
    """
    if not path.is_dir():
        raise FileNotFoundError(f"Package directory is missing: {path}")
    overrides = file_overrides or {}
    physical_files = {
        item.relative_to(path).as_posix(): item
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    if not set(overrides).issubset(physical_files):
        missing = sorted(set(overrides).difference(physical_files))
        raise ValueError(f"Package override targets are absent: {missing}")
    files = []
    for relative, physical in physical_files.items():
        source = overrides.get(relative, physical)
        if not source.is_file():
            raise FileNotFoundError(f"Package override file is missing: {source}")
        files.append(
            {
                "path": relative,
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    package_sha256 = hashlib.sha256(canonical_json(files).encode()).hexdigest()
    return {"files": files, "package_sha256": package_sha256}


def iter_eval_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def batches(iterator: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def validate_source_manifests(cell_id: str, spec: dict[str, Any]) -> dict[str, str]:
    manifest_path = SOURCE_WORKSPACE / "run_manifest.json"
    observed_manifest_hash = logical_json_checksum(manifest_path)
    recorded_manifest_hash = (
        SOURCE_WORKSPACE / "run_manifest.json.logical.sha256"
    ).read_text().strip()
    if recorded_manifest_hash != observed_manifest_hash:
        raise ValueError("Frozen v1 run manifest logical sidecar mismatch")
    manifest = json.loads(manifest_path.read_text())
    # The manifest was legitimately advanced after inference.  Its durable
    # pre-execution pointer must still bind it to the approved 614de... version.
    if (
        observed_manifest_hash != SOURCE_RUN_MANIFEST_LOGICAL_SHA256
        and manifest.get("pre_execution_run_manifest_logical_sha256")
        != SOURCE_RUN_MANIFEST_LOGICAL_SHA256
    ):
        raise ValueError("Frozen v1 manifest is not bound to the approved pre-execution hash")
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    if cell_id not in cells:
        raise ValueError(f"Frozen v1 manifest does not contain {cell_id}")
    source_cell = cells[cell_id]
    expected_pairs = {
        "model": "bertweet",
        "dataset": spec["dataset"],
        "split": spec["split"],
        "cell_identity_sha256": spec["source_cell_identity_sha256"],
        "source_pairs": spec["source_pairs"],
        "evaluation_rows": spec["evaluation_rows"],
        "evaluation_artifact_path": spec["evaluation_artifact_path"],
        "condition_matrix_checksum_sha256": spec["evaluation_logical_checksum_sha256"],
        "condition_registry_sha256": CONDITION_REGISTRY_SHA256,
        "trained_checkpoint_volume_path": spec["checkpoint_path"],
        "trained_checkpoint_package_sha256": spec["checkpoint_package_sha256"],
        "tokenizer_backend": "slow",
        "tokenizer_normalization": "disabled",
        "max_length": MAX_LENGTH,
    }
    for key, expected in expected_pairs.items():
        if source_cell.get(key) != expected:
            raise ValueError(
                f"Frozen v1 cell mismatch for {key}: "
                f"expected {expected!r}, observed {source_cell.get(key)!r}"
            )
    registry = SOURCE_WORKSPACE / "CONDITION_REGISTRY.csv"
    if sha256_file(registry) != CONDITION_REGISTRY_SHA256:
        raise ValueError("Frozen condition registry physical checksum mismatch")
    eval_manifest = json.loads((SOURCE_WORKSPACE / "eval_manifest.json").read_text())
    suites = {
        (suite["dataset"], suite["split"]): suite for suite in eval_manifest["suites"]
    }
    suite = suites[(spec["dataset"], spec["split"])]
    suite_expected = {
        "artifact_path": spec["evaluation_artifact_path"],
        "artifact_sha256": spec["evaluation_artifact_sha256"],
        "logical_checksum_sha256": spec["evaluation_logical_checksum_sha256"],
        "matrix_rows": spec["evaluation_rows"],
        "source_examples": spec["source_pairs"],
        "condition_count": len(CONDITION_IDS),
    }
    for key, expected in suite_expected.items():
        if suite.get(key) != expected:
            raise ValueError(
                f"Frozen evaluation manifest mismatch for {key}: "
                f"expected {expected!r}, observed {suite.get(key)!r}"
            )
    return {
        "approved_pre_execution_logical_sha256": SOURCE_RUN_MANIFEST_LOGICAL_SHA256,
        "observed_post_execution_logical_sha256": observed_manifest_hash,
    }


def validate_tokenizer_package(tokenizer_root: Path) -> dict[str, Any]:
    resolved = tokenizer_root.resolve()
    if resolved.name != TOKENIZER_REVISION:
        raise ValueError(
            f"Tokenizer directory must be pinned revision {TOKENIZER_REVISION}; "
            f"resolved path ends in {resolved.name!r}"
        )
    inventory = package_inventory(resolved)
    observed = {item["path"]: item["sha256"] for item in inventory["files"]}
    if observed != TOKENIZER_FILE_HASHES:
        raise ValueError(
            "Pinned tokenizer file inventory mismatch: "
            f"expected {TOKENIZER_FILE_HASHES}, observed {observed}"
        )
    if inventory["package_sha256"] != TOKENIZER_PACKAGE_SHA256:
        raise ValueError("Pinned tokenizer package checksum mismatch")
    return inventory


def validate_eval_matrix(eval_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(eval_path) != spec["evaluation_artifact_sha256"]:
        raise ValueError("Evaluation artifact physical checksum mismatch")
    logical = hashlib.sha256()
    counts: Counter[str] = Counter()
    clean_labels: list[int] = []
    source_pairs = int(spec["source_pairs"])
    expected_rows = int(spec["evaluation_rows"])
    row_count = 0
    for row_count, row in enumerate(iter_eval_rows(eval_path), start=1):
        zero_index = row_count - 1
        condition_index = zero_index // source_pairs
        source_index = zero_index % source_pairs
        if condition_index >= len(CONDITION_IDS):
            raise ValueError("Evaluation matrix contains more than 24 conditions")
        expected_condition = CONDITION_IDS[condition_index]
        checks = {
            "schema_version": 1,
            "dataset": spec["dataset"],
            "split": spec["split"],
            "condition_id": expected_condition,
            "condition_index": condition_index,
            "source_index": source_index,
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Evaluation crossing mismatch at row {row_count}, {key}: "
                    f"expected {expected!r}, observed {row.get(key)!r}"
                )
        label = int(row["label"])
        if label not in {0, 1, 2}:
            raise ValueError(f"Invalid NLI label at evaluation row {row_count}")
        if condition_index == 0:
            clean_labels.append(label)
        elif clean_labels[source_index] != label:
            raise ValueError(
                f"Label changed across conditions for source_index {source_index}"
            )
        counts[expected_condition] += 1
        logical.update(canonical_json(row).encode("utf-8") + b"\n")
    if row_count != expected_rows:
        raise ValueError(
            f"Evaluation row count mismatch: expected {expected_rows}, observed {row_count}"
        )
    expected_counts = {condition_id: source_pairs for condition_id in CONDITION_IDS}
    if dict(counts) != expected_counts:
        raise ValueError(
            f"Evaluation matrix is not a complete 24 x N crossing: {dict(counts)}"
        )
    if logical.hexdigest() != spec["evaluation_logical_checksum_sha256"]:
        raise ValueError("Evaluation artifact logical checksum mismatch")
    return {
        "passed": True,
        "condition_count": len(CONDITION_IDS),
        "source_pairs": source_pairs,
        "matrix_rows": row_count,
        "condition_row_counts": expected_counts,
        "condition_order": CONDITION_IDS,
        "clean_labels": clean_labels,
    }


def load_reference(
    reference_path: Path,
    spec: dict[str, Any],
    clean_labels: list[int],
) -> dict[str, Any]:
    if sha256_file(reference_path) != spec["reference_prediction_sha256"]:
        raise ValueError("Archived baseline prediction physical checksum mismatch")
    payload = json.loads(reference_path.read_text())
    if "original" not in payload or not isinstance(payload["original"], dict):
        raise ValueError("Archived baseline prediction file lacks an original condition")
    original = payload["original"]
    observed_original_hash = hashlib.sha256(canonical_json(original).encode()).hexdigest()
    if observed_original_hash != spec["reference_original_logical_sha256"]:
        raise ValueError("Archived original-condition logical checksum mismatch")
    predictions = [int(value) for value in original.get("predictions", [])]
    labels = [int(value) for value in original.get("labels", [])]
    source_indices = [int(value) for value in original.get("source_indices", [])]
    source_pairs = int(spec["source_pairs"])
    if not (
        len(predictions) == len(labels) == len(source_indices) == source_pairs
    ):
        raise ValueError("Archived original-condition arrays have inconsistent lengths")
    if source_indices != list(range(source_pairs)):
        raise ValueError("Archived original source indices are not the expected ordered range")
    if labels != clean_labels:
        raise ValueError("Archived original labels differ from the frozen clean condition")
    if any(value not in {0, 1, 2} for value in predictions):
        raise ValueError("Archived original predictions contain an invalid class")
    return {
        "predictions": predictions,
        "accuracy": sum(a == b for a, b in zip(predictions, labels)) / source_pairs,
        "physical_sha256": spec["reference_prediction_sha256"],
        "original_logical_sha256": observed_original_hash,
    }


def write_json_with_logical_sidecar(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    logical_path = path.with_suffix(path.suffix + ".logical.sha256")
    logical_path.write_text(logical_json_checksum(path) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True, choices=sorted(CELL_SPECS))
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--tokenizer-root", type=Path, default=DEFAULT_TOKENIZER_ROOT)
    parser.add_argument(
        "--mnli-weights-override",
        type=Path,
        default=DEFAULT_MNLI_WEIGHTS_OVERRIDE,
        help=(
            "Read-only exact remote MNLI model.safetensors used to reconstruct "
            "the frozen package identity without modifying the stale local backup."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=PREDICTIONS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device. CUDA mode requires an NVIDIA H100.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all frozen inputs and identities without loading the model.",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    # Set offline and deterministic runtime controls before importing torch/transformers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    spec = CELL_SPECS[args.cell_id]
    source_manifest_identity = validate_source_manifests(args.cell_id, spec)
    tokenizer_root = args.tokenizer_root.resolve()
    tokenizer_inventory = validate_tokenizer_package(tokenizer_root)
    eval_path = SOURCE_WORKSPACE / spec["evaluation_artifact_path"]
    matrix = validate_eval_matrix(eval_path, spec)
    clean_labels = matrix.pop("clean_labels")
    checkpoint_root = args.checkpoint_root.resolve()
    checkpoint = checkpoint_root / spec["checkpoint_path"]
    is_mnli = spec["dataset"] == "multi_nli"
    weights_override = args.mnli_weights_override.resolve() if is_mnli else None
    weights_override_sha256 = (
        sha256_file(weights_override) if weights_override is not None else None
    )
    if weights_override_sha256 is not None and weights_override_sha256 != MNLI_WEIGHTS_SHA256:
        raise ValueError(
            "Exact remote MNLI weight override checksum mismatch: expected "
            f"{MNLI_WEIGHTS_SHA256}, observed {weights_override_sha256}"
        )
    checkpoint_overrides = (
        {"model.safetensors": weights_override}
        if weights_override is not None
        else None
    )
    checkpoint_inventory = package_inventory(checkpoint, checkpoint_overrides)
    if checkpoint_inventory["package_sha256"] != spec["checkpoint_package_sha256"]:
        raise ValueError(
            "Resolved checkpoint package hash mismatch: "
            f"expected {spec['checkpoint_package_sha256']}, "
            f"observed {checkpoint_inventory['package_sha256']} at {checkpoint}"
        )
    reference_path = checkpoint_root / spec["reference_prediction_path"]
    reference = load_reference(reference_path, spec, clean_labels)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated_only",
                    "cell_id": args.cell_id,
                    "checkpoint_package_sha256": checkpoint_inventory["package_sha256"],
                    "checkpoint_weight_source": (
                        "verified_remote_override" if is_mnli else "checkpoint_package"
                    ),
                    "source_manifest_identity": source_manifest_identity,
                    "evaluation_artifact_sha256": spec["evaluation_artifact_sha256"],
                    "evaluation_logical_checksum_sha256": spec[
                        "evaluation_logical_checksum_sha256"
                    ],
                    "matrix_validation": matrix,
                    "reference_prediction_sha256": reference["physical_sha256"],
                    "tokenizer_revision": TOKENIZER_REVISION,
                    "tokenizer_package_sha256": tokenizer_inventory["package_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        BertweetTokenizer,
    )

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA mode requested, but CUDA is unavailable")
        accelerator_name = torch.cuda.get_device_name(0)
        if "H100" not in accelerator_name.upper():
            raise RuntimeError(
                f"CUDA correction requires an H100, observed {accelerator_name!r}"
            )
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        accelerator_name = None
    torch.use_deterministic_algorithms(True)
    tokenizer = BertweetTokenizer(
        vocab_file=str(tokenizer_root / "vocab.txt"),
        merges_file=str(tokenizer_root / "bpe.codes"),
        normalization=False,
    )
    if tokenizer.is_fast or getattr(tokenizer, "normalization", None) is not False:
        raise ValueError("Correction requires the slow BERTweet tokenizer with normalization=False")
    if weights_override is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint), local_files_only=True
        )
    else:
        from safetensors.torch import load_file as load_safetensors

        config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
        model = AutoModelForSequenceClassification.from_config(config)
        state_dict = load_safetensors(str(weights_override), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        del state_dict
    if int(model.config.num_labels) != 3:
        raise ValueError(
            f"Checkpoint must expose exactly three NLI labels, found {model.config.num_labels}"
        )
    model.to(args.device).eval()

    inference_code_sha256 = sha256_file(Path(__file__).resolve())
    runtime_profile = {
        "device": args.device,
        "threads": args.threads,
        "interop_threads": 1,
        "batch_size": args.batch_size,
        "max_length": MAX_LENGTH,
        "input_preprocessing": "none",
        "tokenizer_backend": "slow",
        "tokenizer_normalization": "disabled",
        "logit_dtype": "float32",
        "seed": 42,
        "deterministic_algorithms": True,
        "torch_version": torch.__version__,
    }
    if args.device == "cuda":
        runtime_profile.update(
            {
                "accelerator_name": accelerator_name,
                "accelerator_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "allow_tf32": False,
            }
        )
    correction_identity_payload = {
        "schema_version": 2,
        "cell_id": args.cell_id,
        "source_cell_identity_sha256": spec["source_cell_identity_sha256"],
        "checkpoint_package_sha256": spec["checkpoint_package_sha256"],
        "checkpoint_weight_sha256": (
            MNLI_WEIGHTS_SHA256 if weights_override is not None else None
        ),
        "evaluation_logical_checksum_sha256": spec[
            "evaluation_logical_checksum_sha256"
        ],
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_package_sha256": TOKENIZER_PACKAGE_SHA256,
        "inference_code_sha256": inference_code_sha256,
        "runtime_profile": runtime_profile,
    }
    correction_identity = hashlib.sha256(
        canonical_json(correction_identity_payload).encode()
    ).hexdigest()
    output_dir = args.output_dir.resolve()
    output = (
        output_dir
        / f"{args.cell_id}__tokenizerfix__{correction_identity[:16]}.jsonl.gz"
    )
    completion = output.with_suffix(output.suffix + ".complete.json")
    completion_logical = completion.with_suffix(completion.suffix + ".logical.sha256")
    if output.exists() or completion.exists() or completion_logical.exists():
        if not (output.is_file() and completion.is_file() and completion_logical.is_file()):
            raise ValueError("A partial corrected artifact blocks unsafe overwrite")
        prior = json.loads(completion.read_text())
        valid_reuse = (
            prior.get("status") == "complete"
            and prior.get("correction_identity_sha256") == correction_identity
            and prior.get("prediction_sha256") == sha256_file(output)
            and prior.get("prediction_logical_checksum_sha256")
            == logical_jsonl_checksum(output)
            and prior.get("clean_replay", {}).get("passed") is True
            and completion_logical.read_text().strip()
            == logical_json_checksum(completion)
        )
        if valid_reuse:
            print(json.dumps({"status": "already_complete", "artifact": str(output)}))
            return
        raise ValueError("A stale or hash-mismatched corrected artifact blocks unsafe reuse")

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    prediction_logical = hashlib.sha256()
    row_count = 0
    clean_predictions: list[int] = []
    clean_correct = 0
    truncation_count = 0
    truncation_count_by_condition: Counter[str] = Counter()
    exact_max_length_count = 0
    max_length_hits = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{args.cell_id}.", suffix=".tmp", dir=output_dir, delete=False
        ) as raw:
            temporary = Path(raw.name)
            compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
            text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
            try:
                for batch in batches(iter_eval_rows(eval_path), args.batch_size):
                    premises = [row["premise"] for row in batch]
                    hypotheses = [row["hypothesis"] for row in batch]
                    encoded = tokenizer(
                        premises,
                        hypotheses,
                        truncation=True,
                        max_length=MAX_LENGTH,
                        padding=True,
                        return_length=True,
                        return_tensors="pt",
                    )
                    lengths = [int(value) for value in encoded.pop("length").tolist()]
                    candidates = [i for i, length in enumerate(lengths) if length == MAX_LENGTH]
                    max_length_hits += len(candidates)
                    if candidates:
                        untruncated = tokenizer(
                            [premises[i] for i in candidates],
                            [hypotheses[i] for i in candidates],
                            truncation=False,
                            padding=False,
                        )["input_ids"]
                        untruncated_lengths = [len(values) for values in untruncated]
                        for batch_index, length in zip(
                            candidates, untruncated_lengths
                        ):
                            if length > MAX_LENGTH:
                                truncation_count += 1
                                truncation_count_by_condition[
                                    batch[batch_index]["condition_id"]
                                ] += 1
                        exact_max_length_count += sum(
                            length == MAX_LENGTH for length in untruncated_lengths
                        )
                    if args.device == "cuda":
                        encoded = {
                            key: value.to("cuda", non_blocking=True)
                            for key, value in encoded.items()
                        }
                    with torch.inference_mode():
                        logits = model(**encoded).logits.float().cpu()
                    if (
                        logits.ndim != 2
                        or logits.shape[1] != 3
                        or not torch.isfinite(logits).all()
                    ):
                        raise ValueError(
                            "Model output must contain exactly three finite logits per row"
                        )
                    predictions = logits.argmax(dim=-1).tolist()
                    for row, prediction, vector in zip(
                        batch, predictions, logits.tolist()
                    ):
                        prediction = int(prediction)
                        label = int(row["label"])
                        if prediction not in {0, 1, 2}:
                            raise ValueError("Prediction is outside the three NLI classes")
                        if max(range(3), key=vector.__getitem__) != prediction:
                            raise ValueError("Prediction does not equal logit argmax")
                        if row["condition_id"] == "c00_clean":
                            expected_position = len(clean_predictions)
                            if int(row["source_index"]) != expected_position:
                                raise ValueError("Clean replay rows are out of source-index order")
                            clean_predictions.append(prediction)
                            clean_correct += int(prediction == label)
                        result = {
                            "schema_version": 1,
                            "cell_id": args.cell_id,
                            "model": "bertweet",
                            "dataset": row["dataset"],
                            "split": row["split"],
                            "source_index": row["source_index"],
                            "condition_id": row["condition_id"],
                            "condition_type": row["condition_type"],
                            "label": row["label"],
                            "prediction": prediction,
                            "logits": [float(value) for value in vector],
                            "checkpoint_package_sha256": checkpoint_inventory[
                                "package_sha256"
                            ],
                        }
                        encoded_result = canonical_json(result)
                        text.write(encoded_result + "\n")
                        prediction_logical.update(
                            encoded_result.encode("utf-8") + b"\n"
                        )
                        row_count += 1
            finally:
                text.close()

        if row_count != int(spec["evaluation_rows"]):
            raise ValueError(
                f"Prediction row count mismatch: expected {spec['evaluation_rows']}, "
                f"observed {row_count}"
            )
        total = int(spec["source_pairs"])
        if len(clean_predictions) != total:
            raise ValueError("Clean replay prediction count mismatch")
        matched = sum(
            observed == archived
            for observed, archived in zip(clean_predictions, reference["predictions"])
        )
        agreement = matched / total
        replay_accuracy = clean_correct / total
        accuracy_deviation = abs(replay_accuracy - reference["accuracy"])
        agreement_passed = agreement >= CLEAN_REPLAY_AGREEMENT_THRESHOLD
        accuracy_passed = (
            accuracy_deviation <= CLEAN_REPLAY_ACCURACY_DEVIATION_THRESHOLD
        )
        clean_replay = {
            "reference_prediction_sha256": reference["physical_sha256"],
            "reference_original_logical_sha256": reference[
                "original_logical_sha256"
            ],
            "matched": matched,
            "total": total,
            "agreement": agreement,
            "threshold": CLEAN_REPLAY_AGREEMENT_THRESHOLD,
            "agreement_threshold": CLEAN_REPLAY_AGREEMENT_THRESHOLD,
            "agreement_passed": agreement_passed,
            "replay_accuracy": replay_accuracy,
            "reference_accuracy": reference["accuracy"],
            "accuracy_absolute_deviation": accuracy_deviation,
            "accuracy_deviation_threshold": (
                CLEAN_REPLAY_ACCURACY_DEVIATION_THRESHOLD
            ),
            "accuracy_deviation_passed": accuracy_passed,
            "passed": agreement_passed and accuracy_passed,
        }
        if not clean_replay["passed"]:
            raise ValueError(
                "Clean replay gate failed; corrected prediction artifact was not published: "
                + canonical_json(clean_replay)
            )
        if temporary is None:
            raise RuntimeError("Internal error: prediction temporary path is unavailable")
        temporary.replace(output)
        temporary = None
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

    completed = {
        "schema_version": 2,
        "status": "complete",
        "correction": "bertweet_base_tokenizer_roundtrip_fix",
        "cell_id": args.cell_id,
        "source_cell_identity_sha256": spec["source_cell_identity_sha256"],
        "correction_identity_sha256": correction_identity,
        "source_run_manifest_logical_sha256": SOURCE_RUN_MANIFEST_LOGICAL_SHA256,
        "source_manifest_identity": source_manifest_identity,
        "checkpoint_package_sha256": checkpoint_inventory["package_sha256"],
        "checkpoint_identity": {
            "logical_volume": "wnut2026-nli-results-v2",
            "logical_checkpoint_path": spec["checkpoint_path"],
            "package_sha256": checkpoint_inventory["package_sha256"],
            "weight_source": (
                "verified_remote_override" if weights_override is not None else "checkpoint_package"
            ),
            "weight_artifact": (
                weights_override.name if weights_override is not None else "model.safetensors"
            ),
            "weight_sha256": (
                MNLI_WEIGHTS_SHA256
                if weights_override is not None
                else next(
                    item["sha256"]
                    for item in checkpoint_inventory["files"]
                    if item["path"] == "model.safetensors"
                )
            ),
            "provenance": (
                "Exact read-only weight downloaded from the frozen Modal volume; "
                "the six small package files come from the local backup."
                if weights_override is not None
                else "All seven package files come from the frozen read-only checkpoint package."
            ),
        },
        "evaluation_artifact_sha256": spec["evaluation_artifact_sha256"],
        "evaluation_logical_checksum_sha256": spec[
            "evaluation_logical_checksum_sha256"
        ],
        "condition_registry_sha256": CONDITION_REGISTRY_SHA256,
        "inference_code_sha256": inference_code_sha256,
        "tokenizer_identity": {
            "repository": TOKENIZER_REPOSITORY,
            "revision": TOKENIZER_REVISION,
            "package_sha256": tokenizer_inventory["package_sha256"],
            "file_sha256": TOKENIZER_FILE_HASHES,
            "backend": "slow",
            "normalization": False,
        },
        "runtime_profile": runtime_profile,
        "cpu_identity": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "accelerator_identity": (
            {
                "name": accelerator_name,
                "count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
            }
            if args.device == "cuda"
            else None
        ),
        "matrix_validation": matrix,
        "clean_replay": clean_replay,
        "truncation_count": truncation_count,
        "clean_truncation_count": truncation_count_by_condition["c00_clean"],
        "truncation_count_by_condition": {
            condition_id: truncation_count_by_condition[condition_id]
            for condition_id in CONDITION_IDS
        },
        "truncation_rate": truncation_count / row_count,
        "max_length_hits": max_length_hits,
        "exact_max_length_count": exact_max_length_count,
        "prediction_rows": row_count,
        "prediction_logical_checksum_sha256": prediction_logical.hexdigest(),
        "prediction_sha256": sha256_file(output),
        "prediction_artifact": output.name,
        "logits_retained": True,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_seconds": time.monotonic() - started,
    }
    write_json_with_logical_sidecar(completion, completed)
    print(json.dumps(completed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
