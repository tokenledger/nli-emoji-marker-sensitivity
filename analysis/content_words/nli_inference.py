#!/usr/bin/env python3
"""Shared local (Apple MPS or CPU, fp32) inference helpers for the seed-42 clean checkpoints.

Used by content_words/run_inference.py and case_study/transfer/run_inference.py.
Inference only; no training. Tokenizers are loaded through src/training_runtime.load_tokenizer
with each model's frozen contract (configs/experiment_v6.json,
configs/experiment_v6_deberta_ext.json); models are loaded in fp32 and asserted
with training_runtime.assert_float32_parameters.

Label order everywhere: frozen convention 0 = entailment, 1 = neutral,
2 = contradiction (the crossed runner used argmax indices directly; confirmed per
checkpoint by the clean-accuracy gate against the archived clean predictions).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

FROZEN_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
FIVE = ("electra", "roberta", "roberta_base", "timelm", "bertweet")
DEBERTA = "deberta_v3_base"
MODELS = FIVE + (DEBERTA,)
MODEL_LABEL = {"electra": "ELECTRA-small", "roberta": "RoBERTa-large", "roberta_base": "RoBERTa-base",
               "timelm": "TimeLM", "bertweet": "BERTweet", DEBERTA: "DeBERTa-v3-base"}
# Frozen tokenizer / input contracts (configs/experiment_v6.json; deberta ext config).
MODEL_SPECS = {
    "electra": dict(backend="fast", normalization="default", preprocessing="none"),
    "roberta": dict(backend="fast", normalization="default", preprocessing="none"),
    "roberta_base": dict(backend="fast", normalization="default", preprocessing="none"),
    "timelm": dict(backend="fast", normalization="default", preprocessing="timelm"),
    "bertweet": dict(backend="slow", normalization="disabled", preprocessing="none"),
    DEBERTA: dict(backend="fast", normalization="spm_byte_fallback", preprocessing="none"),
}
# Archived clean accuracy ('accuracies.original' in each run's results.json; DeBERTa: modal_results.json
# and the controlled crossed manifest for mismatched).
ARCHIVED_CLEAN_ACC = {
    ("electra", "snli", "test"): 0.8879275244299675,
    ("electra", "multi_nli", "validation_matched"): 0.8140601120733572,
    ("electra", "multi_nli", "validation_mismatched"): 0.8203824247355573,
    ("roberta", "snli", "test"): 0.9251832247557004,
    ("roberta", "multi_nli", "validation_matched"): 0.9066734589913398,
    ("roberta", "multi_nli", "validation_mismatched"): 0.9056143205858421,
    ("roberta_base", "snli", "test"): 0.9121539087947883,
    ("roberta_base", "multi_nli", "validation_matched"): 0.8709118695873663,
    ("roberta_base", "multi_nli", "validation_mismatched"): 0.8738812042310822,
    ("timelm", "snli", "test"): 0.9099144951140065,
    ("timelm", "multi_nli", "validation_matched"): 0.8633723892002038,
    ("timelm", "multi_nli", "validation_mismatched"): 0.8596419853539463,
    ("bertweet", "snli", "test"): 0.9044177524429967,
    ("bertweet", "multi_nli", "validation_matched"): 0.8469689251146205,
    ("bertweet", "multi_nli", "validation_mismatched"): 0.842860048820179,
    (DEBERTA, "snli", "test"): 0.9248778501628665,
    (DEBERTA, "multi_nli", "validation_matched"): 0.8984207845134997,
    (DEBERTA, "multi_nli", "validation_mismatched"): 0.9034377542,  # controlled_crossed_mnli/results.md: 90.34
}
# BERTweet: the fine-tuned checkpoints' saved bpe.codes lost the merge table (QUARANTINE_NOTICE.md), so the
# tokenizer is loaded from the pinned vinai/bertweet-base snapshot used for training and by the corrected v2
# run (sha256 of bpe.codes / vocab.txt verified against the v2 tokenizer_identity record).
BERTWEET_TOKENIZER_REVISION = "b349c1243407b0dcffeabb2337497477286e27ab"
BERTWEET_TOKENIZER_SHA256 = {"bpe.codes": "77712739cd1a7f638e6694b0dd832494e4f66e3d05c709fc6a6a2f988ff9e589",
                             "vocab.txt": "d3f3d56ed440cdb39bd60a76884b67e1061abca462146fdcc751f1ee40ae9ed3"}
SPLITS = (("snli", "test"), ("multi_nli", "validation_matched"), ("multi_nli", "validation_mismatched"))
SPLIT_SHORT = {"test": "SNLI", "validation_matched": "MNLI-m", "validation_mismatched": "MNLI-mm"}
GATE_TOL_PP = 0.5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_dir(model: str, dataset: str) -> Path:
    tag = "snli" if dataset == "snli" else "mnli"
    return paths.checkpoints(f"{model}_{tag}")


def append_suffix(hypothesis: str, phrase: str) -> str:
    """The crossed evaluation's hypothesis-suffix rule: a single space then the phrase, no other change."""
    return f"{hypothesis} {phrase}"


def eval_clean_rows(dataset: str, split: str) -> list[dict]:
    """Clean (c00) rows of the frozen crossed eval artifact, in source_index order."""
    path = paths.data("crossed", f"{dataset}__final__{split}.jsonl.gz")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition_id"] == "c00_clean":
                rows.append({"dataset": dataset, "split": split, "source_index": row["source_index"],
                             "premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]})
    rows.sort(key=lambda r: r["source_index"])
    assert [r["source_index"] for r in rows] == list(range(len(rows)))
    return rows


def archived_prediction_artifact(model: str, dataset: str, split: str) -> Path:
    if model == DEBERTA:
        found = sorted(paths.predictions("crossed", "deberta").rglob(f"deberta_v3_base_controlled_*__{dataset}__{split}.jsonl.gz"))
    elif model == "bertweet":
        found = sorted(paths.predictions("crossed", "bertweet_corrected").rglob(f"bertweet__{dataset}__{split}__tokenizerfix__*.jsonl.gz"))
    else:
        found = sorted(paths.predictions("crossed", "workspace", "predictions").glob(f"{model}__{dataset}__{split}__*.jsonl.gz"))
    if len(found) != 1:
        raise RuntimeError(f"expected one archived artifact for {model}/{dataset}/{split}, found {found}")
    return found[0]


def archived_predictions(model: str, dataset: str, split: str, conditions: tuple[str, ...] = ("c00_clean",)) -> dict[str, dict[int, int]]:
    """{condition_id: {source_index: prediction}} from the archived crossed predictions (validated cells)."""
    out: dict[str, dict[int, int]] = {c: {} for c in conditions}
    with gzip.open(archived_prediction_artifact(model, dataset, split), "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            c = row["condition_id"]
            if c in out:
                out[c][int(row["source_index"])] = int(row["prediction"])
    return out


def load_model_and_tokenizer(model: str, dataset: str, device: str):
    """Load the seed-42 clean checkpoint with its frozen tokenizer contract, fp32, on device."""
    import torch
    from transformers import AutoModelForSequenceClassification
    from training_runtime import assert_float32_parameters, load_tokenizer, unigram_byte_fallback_state

    spec = MODEL_SPECS[model]
    ckpt = checkpoint_dir(model, dataset)
    if not ckpt.is_dir():
        raise FileNotFoundError(ckpt)
    tok_src = ckpt
    if model == "bertweet":
        from huggingface_hub import snapshot_download
        tok_src = Path(snapshot_download("vinai/bertweet-base", revision=BERTWEET_TOKENIZER_REVISION,
                                         allow_patterns=["bpe.codes", "vocab.txt", "config.json", "tokenizer.json"]))
        for name, expected in BERTWEET_TOKENIZER_SHA256.items():
            observed = sha256_file(tok_src / name)
            assert observed == expected, f"bertweet tokenizer file {name} sha256 {observed} != {expected}"
    tokenizer = load_tokenizer(str(tok_src), backend=spec["backend"], normalization=spec["normalization"])
    contract = {"backend": spec["backend"], "normalization": spec["normalization"], "input_preprocessing": spec["preprocessing"],
                "loaded_from": str(tok_src), "tokenizer_class": type(tokenizer).__name__, "is_fast": bool(tokenizer.is_fast)}
    if model == "bertweet":
        contract["tokenizer_file_sha256"] = dict(BERTWEET_TOKENIZER_SHA256)
        contract["tokenizer_repository"] = f"vinai/bertweet-base@{BERTWEET_TOKENIZER_REVISION}"
    if spec["backend"] == "fast":
        assert tokenizer.is_fast, f"{model}: fast backend required"
    else:
        assert not tokenizer.is_fast, f"{model}: slow backend required"
    if spec["normalization"] == "disabled":
        assert getattr(tokenizer, "normalization", None) is False, f"{model}: tokenizer normalization must be False"
        contract["tokenizer_normalization_attr"] = False
    if spec["normalization"] == "spm_byte_fallback":
        assert unigram_byte_fallback_state(tokenizer) is True, "byte_fallback not set on backend Unigram model"
        cat_ids = tokenizer("🐈", add_special_tokens=False)["input_ids"]
        assert tokenizer.unk_token_id not in cat_ids, f"emoji collapsed to [UNK]: {cat_ids}"
        contract["cat_emoji_ids_no_unk"] = cat_ids
    contract["on_god_ids"] = tokenizer(" on god", add_special_tokens=False)["input_ids"]

    hf_model = AutoModelForSequenceClassification.from_pretrained(str(ckpt), dtype=torch.float32, local_files_only=True)
    dtype = assert_float32_parameters(hf_model, label=f"{model}/{dataset}")
    assert int(hf_model.config.num_labels) == 3
    hf_model.to(device).eval()
    contract["param_dtype"] = dtype
    contract["id2label"] = {int(k): v for k, v in hf_model.config.id2label.items()}
    contract["checkpoint_model_safetensors_sha256"] = sha256_file(ckpt / "model.safetensors")
    return tokenizer, hf_model, contract


def predict(tokenizer, hf_model, rows: list[dict], *, device: str, preprocessing: str, batch_size: int = 128,
            max_length: int = 128, log_prefix: str = "", first_eta_rows: int = 2000, log_every_rows: int = 25600):
    """Length-sorted batched inference. Returns (predictions, logits, seconds)."""
    import torch
    from helpers import model_input_text

    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["premise"]) + len(rows[i]["hypothesis"]))
    preds = [None] * len(rows)
    logits_out = [None] * len(rows)
    t0 = time.monotonic()
    seen = 0
    next_log = first_eta_rows
    for start in range(0, len(rows), batch_size):
        idx = order[start:start + batch_size]
        prem = [model_input_text(rows[i]["premise"], preprocessing) for i in idx]
        hyp = [model_input_text(rows[i]["hypothesis"], preprocessing) for i in idx]
        enc = tokenizer(prem, hyp, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            lg = hf_model(**enc).logits.float().cpu()
        if not torch.isfinite(lg).all():
            raise ValueError("non-finite logits")
        pr = lg.argmax(-1).tolist()
        for j, i in enumerate(idx):
            preds[i] = int(pr[j])
            logits_out[i] = [round(float(x), 5) for x in lg[j].tolist()]
        seen += len(idx)
        if seen >= next_log or seen == len(rows):
            el = time.monotonic() - t0
            rate = seen / el
            print(f"{log_prefix}  {seen}/{len(rows)} rows  {rate:.0f} rows/s  elapsed {el/60:.1f} min  "
                  f"ETA {(len(rows) - seen) / rate / 60:.1f} min", flush=True)
            next_log = seen + log_every_rows
    return preds, logits_out, time.monotonic() - t0


def gate_against_archive(model: str, dataset: str, split: str, clean_rows: list[dict], clean_preds: list[int]) -> dict:
    """Clean accuracy on the run's sources vs the archived clean accuracy and predictions on the same sources."""
    arch = archived_predictions(model, dataset, split)["c00_clean"]
    n = len(clean_rows)
    correct = sum(p == r["label"] for p, r in zip(clean_preds, clean_rows))
    arch_correct = sum(arch[r["source_index"]] == r["label"] for r in clean_rows)
    agree = sum(arch[r["source_index"]] == p for p, r in zip(clean_preds, clean_rows))
    acc = correct / n
    archived_full = ARCHIVED_CLEAN_ACC[(model, dataset, split)]
    diff_pp = (acc - archived_full) * 100
    return {"model": model, "dataset": dataset, "split": split, "sources": n, "clean_accuracy": acc,
            "archived_clean_accuracy_full_split": archived_full, "difference_pp_vs_archived_full": diff_pp,
            "archived_clean_accuracy_same_sources": arch_correct / n, "difference_pp_vs_archived_same_sources": (acc - arch_correct / n) * 100,
            "agreement_with_archived_clean_predictions": agree / n, "tolerance_pp": GATE_TOL_PP,
            "passed": abs(diff_pp) <= GATE_TOL_PP}


def write_predictions(path: Path, records: list[dict]) -> dict:
    h = hashlib.sha256()
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for rec in records:
            s = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            f.write(s + "\n")
            h.update((s + "\n").encode())
    tmp.replace(path)
    return {"rows": len(records), "logical_sha256": h.hexdigest(), "file_sha256": sha256_file(path)}


def runtime_info(device: str) -> dict:
    import platform
    import torch
    return {"device": device, "dtype": "torch.float32", "torch": torch.__version__, "python": sys.version.split()[0],
            "platform": platform.platform()}
