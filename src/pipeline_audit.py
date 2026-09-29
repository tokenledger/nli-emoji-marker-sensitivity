"""Stage-by-stage identity checks for reconstructed NLI inputs.

The audit is intentionally independent of a particular Hugging Face checkpoint:
tests can supply a small deterministic tokenizer/model, while experiment runs use
the frozen tokenizer/model pairs.  Text, token IDs, attention masks, and logits
are counted separately so a later-stage success cannot hide an earlier failure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping

import torch

from transforms import canonicalize_whitespace, invert_from_provenance
from preprocessing import normalize_emoji, preprocess_text, remove_markers


NLI_FIELDS = ("premise", "hypothesis")


@dataclass(frozen=True)
class PipelineFailure:
    example_index: int
    stage: str
    clean: object
    reconstructed: object


def reconstruct_from_metadata(example: Mapping) -> dict:
    """Undo a transformed example using its exact Phase 1 provenance."""

    metadata = example.get("transform_metadata")
    if not metadata or "events" not in metadata:
        raise ValueError("Transformed example is missing transform_metadata.events")
    result = dict(example)
    for field in NLI_FIELDS:
        result[field] = invert_from_provenance(
            example[field], metadata["events"].get(field, [])
        )
    return result


def reconstruct_with_preprocessing(mode: str) -> Callable[[Mapping], Mapping]:
    """Return the actual inference-time reconstruction for a principal mode."""

    if mode not in {"emoji", "noise", "combined"}:
        raise ValueError(f"No principal preprocessing pipeline for mode {mode!r}")

    def reconstruct(example: Mapping) -> dict:
        result = dict(example)
        for field in NLI_FIELDS:
            if mode == "emoji":
                result[field] = normalize_emoji(example[field])
            elif mode == "noise":
                result[field] = remove_markers(example[field])
            else:
                result[field] = preprocess_text(
                    example[field], deslang=False, deemoji=True, denoise=True
                )
        return result

    return reconstruct


def _tokenize_pair(tokenizer, example: Mapping, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        example["premise"],
        example["hypothesis"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        key: value if torch.is_tensor(value) else torch.as_tensor(value)
        for key, value in encoded.items()
    }


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _logits(model, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits if hasattr(output, "logits") else output[0]
    return logits.detach().cpu()


def audit_pipeline_identity(
    clean_examples: Iterable[Mapping],
    noisy_examples: Iterable[Mapping],
    *,
    tokenizer,
    reconstruct: Callable[[Mapping], Mapping] = reconstruct_from_metadata,
    model=None,
    max_length: int = 128,
    logit_atol: float = 1e-6,
    raise_on_failure: bool = False,
) -> dict:
    """Compare clean inputs with reconstructed noisy inputs at every stage."""

    clean_rows = list(clean_examples)
    noisy_rows = list(noisy_examples)
    if len(clean_rows) != len(noisy_rows):
        raise ValueError(
            f"Identity audit requires paired inputs; got {len(clean_rows)} clean and "
            f"{len(noisy_rows)} noisy examples"
        )

    counts = {
        "text_equal": 0,
        "input_ids_equal": 0,
        "attention_mask_equal": 0,
        "logits_equal": 0,
    }
    failures: list[PipelineFailure] = []

    for index, (clean, noisy) in enumerate(zip(clean_rows, noisy_rows)):
        reconstructed = reconstruct(noisy)
        clean_text = tuple(canonicalize_whitespace(clean[field]) for field in NLI_FIELDS)
        reconstructed_text = tuple(
            canonicalize_whitespace(reconstructed[field]) for field in NLI_FIELDS
        )
        text_equal = clean_text == reconstructed_text
        counts["text_equal"] += int(text_equal)
        if not text_equal:
            failures.append(PipelineFailure(index, "text", clean_text, reconstructed_text))

        clean_tokens = _tokenize_pair(tokenizer, clean, max_length)
        reconstructed_tokens = _tokenize_pair(tokenizer, reconstructed, max_length)
        ids_equal = torch.equal(
            clean_tokens["input_ids"], reconstructed_tokens["input_ids"]
        )
        counts["input_ids_equal"] += int(ids_equal)
        if not ids_equal:
            failures.append(PipelineFailure(
                index,
                "input_ids",
                clean_tokens["input_ids"].tolist(),
                reconstructed_tokens["input_ids"].tolist(),
            ))

        clean_mask = clean_tokens.get("attention_mask")
        reconstructed_mask = reconstructed_tokens.get("attention_mask")
        masks_equal = (
            clean_mask is None and reconstructed_mask is None
        ) or (
            clean_mask is not None
            and reconstructed_mask is not None
            and torch.equal(clean_mask, reconstructed_mask)
        )
        counts["attention_mask_equal"] += int(masks_equal)
        if not masks_equal:
            failures.append(PipelineFailure(
                index,
                "attention_mask",
                None if clean_mask is None else clean_mask.tolist(),
                None if reconstructed_mask is None else reconstructed_mask.tolist(),
            ))

        if model is not None:
            clean_logits = _logits(model, clean_tokens)
            reconstructed_logits = _logits(model, reconstructed_tokens)
            logits_equal = torch.allclose(
                clean_logits, reconstructed_logits, rtol=0.0, atol=logit_atol
            )
            counts["logits_equal"] += int(logits_equal)
            if not logits_equal:
                failures.append(PipelineFailure(
                    index,
                    "logits",
                    clean_logits.tolist(),
                    reconstructed_logits.tolist(),
                ))

    total = len(clean_rows)
    stages = ("text_equal", "input_ids_equal", "attention_mask_equal")
    if model is not None:
        stages += ("logits_equal",)
    report = {
        "examples": total,
        "logit_atol": logit_atol if model is not None else None,
        "stages": {
            stage: {
                "passed": counts[stage],
                "failed": total - counts[stage],
                "rate": counts[stage] / total if total else 1.0,
            }
            for stage in stages
        },
        "failures": [asdict(failure) for failure in failures],
    }
    if raise_on_failure and failures:
        first = failures[0]
        raise AssertionError(
            f"Pipeline identity failure at example {first.example_index}, stage {first.stage}"
        )
    return report
