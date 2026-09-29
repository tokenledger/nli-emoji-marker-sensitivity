"""Shared Phase 4 training-budget and provenance helpers."""

from __future__ import annotations

import importlib.metadata
import math
import platform
import sys
from typing import Any


SYSTEM_STAGES = {
    "clean_baseline": {
        "training": [],
        "inference": [],
        "classification": "principal",
    },
    "emoji_normalization": {
        "training": [],
        "inference": ["bijective_emoji_normalization"],
        "classification": "principal",
    },
    "marker_augmentation": {
        "training": ["deterministic_epoch_marker_augmentation"],
        "inference": [],
        "classification": "principal",
    },
    "stage_matched_hybrid": {
        "training": ["deterministic_epoch_marker_augmentation"],
        "inference": ["bijective_emoji_normalization"],
        "classification": "principal",
    },
    "known_marker_deletion_oracle": {
        "training": [],
        "inference": ["known_marker_deletion"],
        "classification": "oracle",
    },
    "compute_matched_clean_control": {
        "training": ["deterministic_epoch_clean_presentation"],
        "inference": [],
        "classification": "control",
    },
}


def system_stage_metadata(system_id: str) -> dict[str, Any]:
    """Return a fresh, explicit train/inference transformation declaration."""

    if system_id not in SYSTEM_STAGES:
        raise ValueError(f"Unknown Phase 4 system: {system_id}")
    stages = SYSTEM_STAGES[system_id]
    return {
        "system_id": system_id,
        "classification": stages["classification"],
        "transformations": {
            "training": list(stages["training"]),
            "inference": list(stages["inference"]),
        },
    }


def planned_optimizer_steps(
    train_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    world_size: int = 1,
) -> int:
    """Match the step-count calculation for a full, non-dropping sampler."""

    values = (train_examples, batch_size, gradient_accumulation_steps, epochs, world_size)
    if any(int(value) <= 0 for value in values):
        raise ValueError("Optimizer-step inputs must all be positive integers")
    batches_per_epoch = math.ceil(train_examples / (batch_size * world_size))
    return math.ceil(batches_per_epoch / gradient_accumulation_steps) * epochs


def runtime_environment(torch_module=None) -> dict[str, Any]:
    """Capture the library and hardware details needed to reproduce a run."""

    packages = {}
    for package in ("torch", "transformers", "datasets", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python": sys.version,
        "accelerator": "cpu",
        "device_name": platform.processor() or "CPU",
        "cuda_device_count": 0,
    }
    if torch_module is not None and torch_module.cuda.is_available():
        hardware.update({
            "accelerator": "cuda",
            "device_name": torch_module.cuda.get_device_name(0),
            "cuda_device_count": torch_module.cuda.device_count(),
            "cuda_version": torch_module.version.cuda,
        })
    return {"library_versions": packages, "hardware": hardware}


def training_run_metadata(
    *,
    system_id: str,
    train_examples: int,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    learning_rate: float,
    actual_optimizer_steps: int,
    runtime_seconds: float,
    checkpoint_selection: str,
    presentation_audit: dict[str, Any] | None = None,
    world_size: int = 1,
    torch_module=None,
    frozen_run: bool = False,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = system_stage_metadata(system_id)
    metadata["run_origin"] = "frozen_config" if frozen_run else "direct_override"
    if not frozen_run and metadata["classification"] == "principal":
        metadata["classification"] = "exploratory"
    effective_batch_size = batch_size * gradient_accumulation_steps * world_size
    metadata.update({
        "epochs": epochs,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": world_size,
        "effective_batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "planned_optimizer_steps": planned_optimizer_steps(
            train_examples, batch_size, gradient_accumulation_steps, epochs, world_size
        ),
        "actual_optimizer_steps": int(actual_optimizer_steps),
        "checkpoint_selection": checkpoint_selection,
        "runtime_seconds": float(runtime_seconds),
    })
    if presentation_audit is not None:
        metadata["training_presentations"] = {
            key: presentation_audit[key]
            for key in (
                "policy", "dataset_examples", "completed_epochs",
                "expected_presentations", "unique_presentations", "total_fetches",
                "duplicate_presentations", "missing_presentations",
                "transformed_presentations", "observed",
            )
        }
    if provenance is not None:
        metadata["provenance"] = dict(provenance)
    metadata.update(runtime_environment(torch_module))
    return metadata


def validate_training_run_metadata(metadata: dict[str, Any]) -> None:
    """Fail closed when a completed training run violates its frozen budget."""

    planned = metadata.get("planned_optimizer_steps")
    actual = metadata.get("actual_optimizer_steps")
    if not isinstance(planned, int) or not isinstance(actual, int):
        raise RuntimeError("Training metadata must contain integer optimizer-step counts")
    if actual != planned:
        raise RuntimeError(
            f"Optimizer-step budget mismatch: planned {planned}, observed {actual}"
        )

    audit = metadata.get("training_presentations")
    if not isinstance(audit, dict):
        raise RuntimeError("Training metadata is missing the presentation audit")
    if audit.get("observed") is not True:
        raise RuntimeError("Training presentation audit was not observed")
    if audit.get("duplicate_presentations") or audit.get("missing_presentations"):
        raise RuntimeError("Training presentation audit violates compute matching")
    if audit.get("unique_presentations") != audit.get("expected_presentations"):
        raise RuntimeError("Training presentation totals do not match the frozen plan")
