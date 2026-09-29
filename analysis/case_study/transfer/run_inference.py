#!/usr/bin/env python3
"""Independent-data confirmation of the on-god effect: SICK test and ANLI dev R1-R3 under 12
hypothesis-suffix conditions, six seed-42 clean checkpoints (MultiNLI-trained by default;
--train snli for the SNLI-trained checkpoints on SICK). Local Apple MPS, fp32, inference only.

Checkpoint gate: before scoring the new dataset, each checkpoint classifies the first 2,000 clean
sources of its home split (MNLI-m or SNLI test) and must reproduce the archived clean accuracy on
those same sources within 0.5 pp (agreement with the archived predictions is also recorded).
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "content_words"))
from nli_inference import (MODELS, MODEL_SPECS, append_suffix, archived_predictions, checkpoint_dir, eval_clean_rows,
                           load_model_and_tokenizer, predict, runtime_info, write_predictions, ARCHIVED_CLEAN_ACC, GATE_TOL_PP)
from task_b_conditions import CONDITIONS, DATASETS  # local module; "conditions" would resolve to content_words/conditions.py
import paths

PRED = paths.predictions() / "transfer" / "predictions"
MAN = paths.predictions() / "transfer" / "manifests"
HOME = {"mnli": ("multi_nli", "validation_matched"), "snli": ("snli", "test")}


def load_rows(name: str) -> list[dict]:
    rows = [json.loads(l) for l in paths.data("external", DATASETS[name]).read_text().splitlines() if l.strip()]
    assert [r["source_index"] for r in rows] == list(range(len(rows)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--train", default="mnli", choices=["mnli", "snli"])
    ap.add_argument("--datasets", nargs="+", default=["sick", "anli"], choices=list(DATASETS))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--gate-sources", type=int, default=2000)
    args = ap.parse_args()
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    PRED.mkdir(parents=True, exist_ok=True); MAN.mkdir(parents=True, exist_ok=True)
    home_ds, home_split = HOME[args.train]
    t_all = time.monotonic()
    for model in args.models:
        todo = [d for d in args.datasets if not (MAN / f"{model}__{args.train}__{d}.json").is_file()]
        if not todo:
            continue
        tok, hf, contract = load_model_and_tokenizer(model, home_ds, device)
        prep = MODEL_SPECS[model]["preprocessing"]
        print(f"[{model}/{args.train}] loaded {checkpoint_dir(model, home_ds)} tok={contract['tokenizer_class']} on_god_ids={contract['on_god_ids']}", flush=True)
        # ---- checkpoint gate on the home split ----
        g_rows = eval_clean_rows(home_ds, home_split)[: args.gate_sources]
        gp, _, gsec = predict(tok, hf, g_rows, device=device, preprocessing=prep, batch_size=args.batch_size, max_length=args.max_length,
                              log_prefix=f"[{model}/gate]", first_eta_rows=10**9)
        arch = archived_predictions(model, home_ds, home_split)["c00_clean"]
        acc = sum(p == r["label"] for p, r in zip(gp, g_rows)) / len(g_rows)
        arch_acc = sum(arch[r["source_index"]] == r["label"] for r in g_rows) / len(g_rows)
        agree = sum(arch[r["source_index"]] == p for p, r in zip(gp, g_rows)) / len(g_rows)
        gate = {"home_split": f"{home_ds}/{home_split}", "sources": len(g_rows), "clean_accuracy": acc, "archived_clean_accuracy_same_sources": arch_acc,
                "difference_pp": (acc - arch_acc) * 100, "archived_clean_accuracy_full_split": ARCHIVED_CLEAN_ACC[(model, home_ds, home_split)],
                "agreement_with_archived_clean_predictions": agree, "tolerance_pp": GATE_TOL_PP, "passed": abs(acc - arch_acc) * 100 <= GATE_TOL_PP, "seconds": gsec}
        print(f"CHECKPOINT GATE [{model}/{args.train}]: clean acc on first {len(g_rows)} {home_split} sources {acc*100:.2f}% vs archived same sources "
              f"{arch_acc*100:.2f}% (diff {gate['difference_pp']:+.2f} pp; agreement {agree*100:.2f}%) -> {'PASS' if gate['passed'] else 'FAIL'}", flush=True)
        if not gate["passed"]:
            (MAN / f"{model}__{args.train}.FAILED_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
            continue
        for d in todo:
            tag = f"{model}__{args.train}__{d}"
            rows = load_rows(d)
            t0 = time.monotonic(); records = []; secs = {}; accs = {}
            for cid, phrase, group in CONDITIONS:
                trows = rows if not phrase else [dict(r, hypothesis=append_suffix(r["hypothesis"], phrase)) for r in rows]
                p_, l_, sec = predict(tok, hf, trows, device=device, preprocessing=prep, batch_size=args.batch_size, max_length=args.max_length,
                                      log_prefix=f"[{tag}/{phrase or 'clean'}]", first_eta_rows=2000 if cid == "c00_clean" else 10**9)
                secs[cid] = sec
                accs[cid] = sum(x == r["label"] for x, r in zip(p_, rows)) / len(rows)
                print(f"[{tag}] {cid:26s} acc {accs[cid]*100:.2f}%  {sec/60:.1f} min", flush=True)
                for r, p, lg in zip(rows, p_, l_):
                    records.append({"schema_version": 1, "cell_id": tag, "model": model, "train_dataset": args.train, "dataset": d, "split": r["split"],
                                    "round": r["round"], "source_index": r["source_index"], "orig_id": r["orig_id"], "condition_id": cid,
                                    "condition_type": group, "inserted_text": phrase, "label": r["label"], "prediction": p, "logits": lg,
                                    "label_order": "frozen:0=entailment,1=neutral,2=contradiction"})
            out = PRED / f"{tag}.jsonl.gz"
            info = write_predictions(out, records)
            manifest = {"schema_version": 1, "cell_id": tag, "model": model, "train_dataset": args.train, "dataset": d, "data_file": DATASETS[d],
                        "checkpoint_dir": str(checkpoint_dir(model, home_ds)), "checkpoint_resolved": str(checkpoint_dir(model, home_ds).resolve()),
                        "tokenizer_contract": contract, "checkpoint_gate": gate, "conditions": [c for c, _, _ in CONDITIONS], "sources": len(rows),
                        "accuracy_per_condition": accs, "seconds_per_condition": secs, "prediction_artifact": out.name, **info,
                        "cell_seconds": time.monotonic() - t0, "batch_size": args.batch_size, "max_length": args.max_length, **runtime_info(device),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()}
            (MAN / f"{tag}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
            print(f"[{tag}] done {info['rows']} rows in {(time.monotonic()-t0)/60:.1f} min", flush=True)
        del hf, tok
        if device == "mps":
            torch.mps.empty_cache()
    print(f"all done in {(time.monotonic()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
