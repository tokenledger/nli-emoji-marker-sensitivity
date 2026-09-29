#!/usr/bin/env python3
"""Content-word two-word controls: six seed-42 clean encoders x three final splits x
(clean + 9 new hypothesis-suffix conditions). Local Apple MPS, fp32. Inference only.

Per model x split: (1) run all clean sources and gate the clean accuracy against the
archived clean accuracy (tolerance 0.5 pp) and the archived clean predictions;
(2) run the nine new conditions; (3) write predictions/<model>__<dataset>__<split>.jsonl.gz
and manifests/<model>__<dataset>__<split>.json below PREDICTIONS_DIR/content_words/.
Cells with an existing manifest are skipped.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nli_inference import (MODELS, MODEL_SPECS, SPLITS, append_suffix, checkpoint_dir, eval_clean_rows,
                           gate_against_archive, load_model_and_tokenizer, predict, runtime_info, write_predictions)
from conditions import NEW_CONDITIONS
import paths

PRED = paths.predictions() / "content_words" / "predictions"
MAN = paths.predictions() / "content_words" / "manifests"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--splits", nargs="+", default=[f"{d}/{s}" for d, s in SPLITS])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--limit-sources", type=int, default=None, help="debug only")
    args = ap.parse_args()
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    PRED.mkdir(parents=True, exist_ok=True); MAN.mkdir(parents=True, exist_ok=True)
    splits = [tuple(s.split("/")) for s in args.splits]
    t_all = time.monotonic()
    for model in args.models:
        for dataset in ("snli", "multi_nli"):
            todo = [(d, s) for d, s in splits if d == dataset and not (MAN / f"{model}__{d}__{s}.json").is_file()]
            if not todo:
                continue
            tok, hf, contract = load_model_and_tokenizer(model, dataset, device)
            print(f"[{model}/{dataset}] loaded {checkpoint_dir(model, dataset)} id2label={contract['id2label']} "
                  f"tok={contract['tokenizer_class']} fast={contract['is_fast']} on_god_ids={contract['on_god_ids']}", flush=True)
            for d, s in todo:
                tag = f"{model}__{d}__{s}"
                rows = eval_clean_rows(d, s)
                if args.limit_sources:
                    rows = rows[: args.limit_sources]
                t0 = time.monotonic()
                cp, cl, csec = predict(tok, hf, rows, device=device, preprocessing=MODEL_SPECS[model]["preprocessing"],
                                       batch_size=args.batch_size, max_length=args.max_length, log_prefix=f"[{tag}/clean]")
                gate = gate_against_archive(model, d, s, rows, cp)
                print(f"SANITY GATE [{tag}]: clean acc {gate['clean_accuracy']*100:.2f}% vs archived "
                      f"{gate['archived_clean_accuracy_full_split']*100:.2f}% (diff {gate['difference_pp_vs_archived_full']:+.2f} pp; "
                      f"agreement with archived clean predictions {gate['agreement_with_archived_clean_predictions']*100:.2f}%) "
                      f"-> {'PASS' if gate['passed'] else 'FAIL'}", flush=True)
                if not gate["passed"] and not args.limit_sources:
                    (MAN / f"{tag}.FAILED_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
                    print(f"gate failed for {tag}; skipping conditions", flush=True)
                    continue
                records = []
                for r, p, lg in zip(rows, cp, cl):
                    records.append({"schema_version": 1, "cell_id": tag, "model": model, "dataset": d, "split": s,
                                    "source_index": r["source_index"], "condition_id": "c00_clean", "condition_type": "clean",
                                    "inserted_text": "", "label": r["label"], "prediction": p, "logits": lg,
                                    "label_order": "frozen:0=entailment,1=neutral,2=contradiction"})
                cond_secs = {"c00_clean": csec}
                for cid, phrase, group in NEW_CONDITIONS:
                    trows = [dict(r, hypothesis=append_suffix(r["hypothesis"], phrase)) for r in rows]
                    p_, l_, sec = predict(tok, hf, trows, device=device, preprocessing=MODEL_SPECS[model]["preprocessing"],
                                          batch_size=args.batch_size, max_length=args.max_length, log_prefix=f"[{tag}/{phrase}]",
                                          first_eta_rows=len(trows) + 1)
                    cond_secs[cid] = sec
                    acc = sum(x == r["label"] for x, r in zip(p_, rows)) / len(rows)
                    print(f"[{tag}] {cid:32s} acc {acc*100:.2f}% (clean {gate['clean_accuracy']*100:.2f}%)  {sec/60:.1f} min", flush=True)
                    for r, p, lg in zip(rows, p_, l_):
                        records.append({"schema_version": 1, "cell_id": tag, "model": model, "dataset": d, "split": s,
                                        "source_index": r["source_index"], "condition_id": cid, "condition_type": group,
                                        "inserted_text": phrase, "label": r["label"], "prediction": p, "logits": lg,
                                        "label_order": "frozen:0=entailment,1=neutral,2=contradiction"})
                if args.limit_sources:
                    print("debug run; not writing artifacts"); continue
                out = PRED / f"{tag}.jsonl.gz"
                info = write_predictions(out, records)
                manifest = {"schema_version": 1, "cell_id": tag, "model": model, "dataset": d, "split": s,
                            "checkpoint_dir": str(checkpoint_dir(model, d)), "checkpoint_resolved": str(checkpoint_dir(model, d).resolve()),
                            "tokenizer_contract": contract, "sanity_gate": gate, "conditions": ["c00_clean"] + [c for c, _, _ in NEW_CONDITIONS],
                            "sources": len(rows), "prediction_artifact": out.name, **info, "seconds_per_condition": cond_secs,
                            "cell_seconds": time.monotonic() - t0, "batch_size": args.batch_size, "max_length": args.max_length,
                            **runtime_info(device), "finished_at_utc": datetime.now(timezone.utc).isoformat()}
                (MAN / f"{tag}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
                print(f"[{tag}] done {info['rows']} rows in {(time.monotonic()-t0)/60:.1f} min", flush=True)
            del hf, tok
            if device == "mps":
                torch.mps.empty_cache()
    print(f"all done in {(time.monotonic()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
