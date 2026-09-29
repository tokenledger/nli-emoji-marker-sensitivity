#!/usr/bin/env python3
"""Shared loader for the label-validity analyses.

Reads the saved crossed predictions of the five seed-42 clean encoders
(12 validated v1 cells plus the three tokenizer-corrected BERTweet cells,
exactly the cells that ``analyze_v2.py`` uses) and the controlled DeBERTa-v3-base
crossed predictions, and caches them as compact int8 arrays under ``cache/``.

Label order everywhere: 0 = entailment, 1 = neutral, 2 = contradiction.
No model is run; no annotation is read.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "dataprep" / "CONDITION_REGISTRY.csv"
CACHE = paths.output("cache", "crossed")

FIVE = ("electra", "roberta", "roberta_base", "timelm", "bertweet")
DEBERTA = "deberta_v3_base"
MODELS = FIVE + (DEBERTA,)
SPLITS = (
    ("snli", "test"),
    ("multi_nli", "validation_matched"),
    ("multi_nli", "validation_mismatched"),
)
SPLIT_SHORT = {"test": "SNLI", "validation_matched": "MNLI-m", "validation_mismatched": "MNLI-mm"}
MODEL_LABEL = {
    "electra": "ELECTRA-small",
    "roberta": "RoBERTa-large",
    "roberta_base": "RoBERTa-base",
    "timelm": "TimeLM",
    "bertweet": "BERTweet",
    DEBERTA: "DeBERTa-v3-base",
}
ENT, NEU, CON = 0, 1, 2


def registry() -> list[dict[str, str]]:
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


REGISTRY = registry()
CONDITIONS = [r["condition_id"] for r in REGISTRY]
NONCLEAN = [c for c in CONDITIONS if c != "c00_clean"]
COND_TEXT = {r["condition_id"]: (r["inserted_text"] or "clean") for r in REGISTRY}
COND_TYPE = {r["condition_id"]: r["condition_type"] for r in REGISTRY}
GROUP = {
    "informal_single_token": "Informal markers",
    "informal_full_phrase": "Informal markers",
    "component_ablation": "Component words",
    "formal_control": "Formal controls",
    "random_control": "Random controls",
    "clean": "Clean",
}
INFORMAL = [c for c in NONCLEAN if COND_TYPE[c].startswith("informal_")]
CONTROLS = [c for c in NONCLEAN if COND_TYPE[c] in ("formal_control", "random_control")]


def _artifact(model: str, dataset: str, split: str) -> Path:
    if model == DEBERTA:
        root = paths.predictions("crossed", "deberta")
        pattern = f"deberta_v3_base_controlled_*__{dataset}__{split}.jsonl.gz"
    elif model == "bertweet":
        root = paths.predictions("crossed", "bertweet_corrected")
        pattern = f"bertweet__{dataset}__{split}__tokenizerfix__*.jsonl.gz"
    else:
        root = paths.predictions("crossed", "workspace", "predictions")
        pattern = f"{model}__{dataset}__{split}__*.jsonl.gz"
    found = sorted(root.rglob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"expected one artifact for {model}/{dataset}/{split}, found {found}")
    return found[0]


def _parse(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per_condition: dict[str, dict[int, int]] = {c: {} for c in CONDITIONS}
    labels: dict[int, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cond = row["condition_id"]
            src = int(row["source_index"])
            pred = int(row["prediction"])
            lab = int(row["label"])
            if pred not in (0, 1, 2) or lab not in (0, 1, 2):
                raise RuntimeError(f"bad label/prediction in {path.name}")
            if src in per_condition[cond]:
                raise RuntimeError(f"duplicate {cond}:{src} in {path.name}")
            per_condition[cond][src] = pred
            if labels.setdefault(src, lab) != lab:
                raise RuntimeError(f"label conflict at source {src} in {path.name}")
    n = len(labels)
    if sorted(labels) != list(range(n)):
        raise RuntimeError(f"source indices are not 0..N-1 in {path.name}")
    lab_arr = np.fromiter((labels[i] for i in range(n)), dtype=np.int8, count=n)
    preds = {}
    for cond, mapping in per_condition.items():
        if len(mapping) != n:
            raise RuntimeError(f"{cond} has {len(mapping)} rows, expected {n} in {path.name}")
        preds[cond] = np.fromiter((mapping[i] for i in range(n)), dtype=np.int8, count=n)
    return lab_arr, preds


def load_cell(model: str, dataset: str, split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (labels[N], {condition_id: predictions[N]}) for one model-split cell."""
    cache = CACHE / f"{model}__{dataset}__{split}.npz"
    if cache.is_file():
        with np.load(cache) as data:
            return data["labels"], {c: data[c] for c in CONDITIONS}
    labels, preds = _parse(_artifact(model, dataset, split))
    np.savez_compressed(cache, labels=labels, **preds)
    return labels, preds


def load_all() -> dict[tuple[str, str, str], dict]:
    """{(model, dataset, split): {"labels": ..., "preds": {...}}} for all 18 cells.

    Cross-checks that every model in a split saw identical gold labels.
    """
    out = {}
    for dataset, split in SPLITS:
        reference = None
        for model in MODELS:
            labels, preds = load_cell(model, dataset, split)
            if reference is None:
                reference = labels
            elif not np.array_equal(reference, labels):
                raise RuntimeError(f"gold labels differ across models in {dataset}/{split}")
            out[(model, dataset, split)] = {"labels": labels, "preds": preds}
    return out


def source_rows(dataset: str, split: str) -> list[dict]:
    """Clean (c00) rows of the frozen eval artifact, in source_index order."""
    path = paths.data("crossed", f"{dataset}__final__{split}.jsonl.gz")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition_id"] == "c00_clean":
                rows.append(row)
    rows.sort(key=lambda r: r["source_index"])
    if [r["source_index"] for r in rows] != list(range(len(rows))):
        raise RuntimeError("eval artifact source indices are not contiguous")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    cells = load_all()
    for key, cell in cells.items():
        clean = cell["preds"]["c00_clean"]
        print(key, len(cell["labels"]), f"clean acc {np.mean(clean == cell['labels']):.4f}")
