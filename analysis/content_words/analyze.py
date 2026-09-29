#!/usr/bin/env python3
"""Analysis of the content-word two-word controls (preregistered in PREDICTIONS.md).

Baseline: the archived clean predictions of each model (validated crossed cells, identical
to the paper's). Reference conditions on god / god / in fact / nearby / seriously come from
the same archived predictions. New conditions come from
PREDICTIONS_DIR/content_words/predictions/*.jsonl.gz.
Writes results.csv (long), mcnemar.csv, token_lengths.csv, verdict.json, results.md.
"""
from __future__ import annotations
import csv, gzip, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nli_inference import (DEBERTA, FIVE, MODELS, MODEL_LABEL, SPLITS, SPLIT_SHORT, archived_prediction_artifact,
                           eval_clean_rows, load_model_and_tokenizer, MODEL_SPECS)
from conditions import NEW_CONDITIONS, REFERENCE_CONDITIONS
from stats_utils import mcnemar_exact, holm
import paths

OUT = paths.output("content_words")

PAPER_DENOM = {("snli", "test"): 2887, ("multi_nli", "validation_matched"): 2518, ("multi_nli", "validation_mismatched"): 2544}
REF_IDS = [c for c, _, _ in REFERENCE_CONDITIONS]
NEW_IDS = [c for c, _, _ in NEW_CONDITIONS]
PHRASE = {c: p for c, p, _ in REFERENCE_CONDITIONS + NEW_CONDITIONS}
GROUP = {c: g for c, p, g in REFERENCE_CONDITIONS + NEW_CONDITIONS}
ON_GOD = "c05_informal_on_god"
RELIGIOUS = ["c28_religious_heaven_knows", "c29_religious_lord_knows", "c30_religious_by_heaven", "c31_religious_good_heavens"]
FORMAL_CONTENT = ["c24_formal_content_in_truth", "c25_formal_content_in_essence", "c26_formal_content_in_short", "c27_formal_content_in_reality"]
CACHE = paths.output("cache", "content_words")


def parse(path: Path, wanted: set[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per: dict[str, dict[int, int]] = {c: {} for c in wanted}
    labels: dict[int, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c = r["condition_id"]
            if c not in per:
                continue
            s = int(r["source_index"])
            per[c][s] = int(r["prediction"])
            labels[s] = int(r["label"])
    n = max(labels) + 1
    lab = np.array([labels[i] for i in range(n)], dtype=np.int8)
    preds = {}
    for c, m in per.items():
        if len(m) != n:
            raise RuntimeError(f"{path.name}: {c} has {len(m)} rows, expected {n}")
        preds[c] = np.array([m[i] for i in range(n)], dtype=np.int8)
    return lab, preds


def load_cell(model: str, dataset: str, split: str):
    cache = CACHE / f"{model}__{dataset}__{split}.npz"
    if cache.is_file():
        with np.load(cache) as d:
            return d["labels"], {k: d[k] for k in d.files if k != "labels"}
    lab_a, arch = parse(archived_prediction_artifact(model, dataset, split), set(["c00_clean"] + REF_IDS))
    new_path = paths.predictions("content_words", "predictions", f"{model}__{dataset}__{split}.jsonl.gz")
    lab_n, new = parse(new_path, set(["c00_clean"] + NEW_IDS))
    if not np.array_equal(lab_a, lab_n):
        raise RuntimeError(f"labels differ between archive and new run for {model}/{dataset}/{split}")
    preds = {"c00_clean": arch["c00_clean"], "c00_clean_rerun": new["c00_clean"]}
    preds.update({c: arch[c] for c in REF_IDS})
    preds.update({c: new[c] for c in NEW_IDS})
    np.savez_compressed(cache, labels=lab_a, **preds)
    return lab_a, preds


def main() -> None:
    cells = {}
    missing = []
    for model in MODELS:
        for dataset, split in SPLITS:
            if not (paths.predictions("content_words", "manifests") / f"{model}__{dataset}__{split}.json").is_file():
                missing.append((model, dataset, split)); continue
            cells[(model, dataset, split)] = load_cell(model, dataset, split)
    if missing:
        print("WARNING: missing cells:", missing)
    gates = {}
    for (model, dataset, split) in cells:
        m = json.loads((paths.predictions("content_words", "manifests") / f"{model}__{dataset}__{split}.json").read_text())
        gates[(model, dataset, split)] = m["sanity_gate"]

    rows = []       # long results
    mc_rows = []
    per_split_common = {}
    for dataset, split in SPLITS:
        avail = [m for m in MODELS if (m, dataset, split) in cells]
        five = [m for m in FIVE if m in avail]
        labels = cells[(avail[0], dataset, split)][0]
        ent = labels == 0
        common5 = ent.copy()
        for m in five:
            common5 &= cells[(m, dataset, split)][1]["c00_clean"] == 0
        common6 = common5 & ((cells[(DEBERTA, dataset, split)][1]["c00_clean"] == 0) if DEBERTA in avail else True)
        per_split_common[(dataset, split)] = {"n_five": int(common5.sum()), "n_six": int(common6.sum()), "five_complete": len(five) == 5,
                                              "paper_denominator": PAPER_DENOM[(dataset, split)]}
        for cond in REF_IDS + NEW_IDS:
            all5 = common5.copy(); all6 = common6.copy()
            for m in five:
                all5 &= cells[(m, dataset, split)][1][cond] != 0
            if DEBERTA in avail:
                all6 = all5 & (cells[(DEBERTA, dataset, split)][1][cond] != 0)
            for m in avail:
                lab, pr = cells[(m, dataset, split)]
                clean = pr["c00_clean"]; p = pr[cond]
                cc = (lab == 0) & (clean == 0)
                fail = cc & (p != 0)
                n_fail = int(fail.sum())
                rows.append({
                    "model": m, "model_label": MODEL_LABEL[m], "dataset": dataset, "split": split, "split_short": SPLIT_SHORT[split],
                    "condition_id": cond, "phrase": PHRASE[cond], "group": GROUP[cond], "source": "archived" if cond in REF_IDS else "new_run",
                    "n_sources": int(len(lab)), "clean_accuracy": float(np.mean(clean == lab)), "condition_accuracy": float(np.mean(p == lab)),
                    "accuracy_change_pp": float((np.mean(p == lab) - np.mean(clean == lab)) * 100),
                    "flip_rate_pct": float(np.mean(p != clean) * 100),
                    "clean_correct_entailments": int(cc.sum()), "failures": n_fail,
                    "failure_pct": float(100 * n_fail / cc.sum()),
                    "to_neutral": int((fail & (p == 1)).sum()), "to_contradiction": int((fail & (p == 2)).sum()),
                    "neutral_share_pct": float(100 * (fail & (p == 1)).sum() / n_fail) if n_fail else float("nan"),
                    "all_five_denominator": int(common5.sum()), "all_five_failures": int(all5.sum()),
                    "all_five_failure_pct": float(100 * all5.sum() / common5.sum()) if len(five) == 5 else float("nan"),
                    "all_six_denominator": int(common6.sum()), "all_six_failures": int(all6.sum()),
                    "all_six_failure_pct": float(100 * all6.sum() / common6.sum()) if (len(five) == 5 and DEBERTA in avail) else float("nan"),
                })
        # McNemar: on god vs each new control, per model, Holm family of 9
        for m in avail:
            lab, pr = cells[(m, dataset, split)]
            cc = (lab == 0) & (pr["c00_clean"] == 0)
            og_fail = cc & (pr[ON_GOD] != 0)
            fam = []
            for cond in NEW_IDS:
                c_fail = cc & (pr[cond] != 0)
                t = mcnemar_exact(og_fail[cc], c_fail[cc])
                fam.append({"model": m, "model_label": MODEL_LABEL[m], "dataset": dataset, "split": split, "split_short": SPLIT_SHORT[split],
                            "control_id": cond, "control": PHRASE[cond], "group": GROUP[cond], "n_clean_correct_entailments": int(cc.sum()),
                            "on_god_failures": int(og_fail.sum()), "control_failures": int(c_fail.sum()),
                            "on_god_failure_pct": float(100 * og_fail.sum() / cc.sum()), "control_failure_pct": float(100 * c_fail.sum() / cc.sum()),
                            "ratio_on_god_over_control": float(og_fail.sum() / c_fail.sum()) if c_fail.sum() else float("inf"),
                            "only_on_god_fails": t["only_a"], "only_control_fails": t["only_b"], "p_exact": t["p"]})
            adj = holm([f["p_exact"] for f in fam])
            for f, a in zip(fam, adj):
                f["p_holm"] = a
                f["direction"] = "on god worse" if f["on_god_failures"] > f["control_failures"] else ("control worse" if f["control_failures"] > f["on_god_failures"] else "tie")
                f["significant_holm_01"] = bool(a < 0.01)
            mc_rows.extend(fam)

    # ---- write CSVs ----
    def write(path, rs):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
    write(OUT / "results.csv", rows)
    write(OUT / "mcnemar.csv", mc_rows)

    # ---- token lengths ----
    tl = []
    try:
        for m in MODELS:
            tok, hf, c = load_model_and_tokenizer(m, "multi_nli", "cpu")
            for cond in [ON_GOD] + NEW_IDS + ["c18_formal_in_fact"]:
                tl.append({"model": m, "phrase": PHRASE[cond], "subwords_with_leading_space": len(tok(" " + PHRASE[cond], add_special_tokens=False)["input_ids"])})
            del hf
        write(OUT / "token_lengths.csv", tl)
    except Exception as e:  # token lengths are informational
        print("token length table skipped:", e)

    # ---- verdict ----
    R = {(r["model"], r["dataset"], r["split"], r["condition_id"]): r for r in rows}
    verdict = {"gates": {f"{m}/{SPLIT_SHORT[s]}": {"clean_acc_rerun": g["clean_accuracy"], "archived": g["archived_clean_accuracy_full_split"],
                                                     "diff_pp": g["difference_pp_vs_archived_full"], "agreement": g["agreement_with_archived_clean_predictions"], "passed": g["passed"]}
                         for (m, d, s), g in gates.items()},
               "common_denominators": {f"{SPLIT_SHORT[s]}": v for (d, s), v in per_split_common.items()}, "cells": {}}
    for (m, d, s) in cells:
        og = R[(m, d, s, ON_GOD)]["failure_pct"]
        ctrl = {PHRASE[c]: R[(m, d, s, c)]["failure_pct"] for c in NEW_IDS}
        worst = max(ctrl, key=ctrl.get)
        fam = [x for x in mc_rows if x["model"] == m and x["dataset"] == d and x["split"] == s]
        verdict["cells"][f"{m}/{SPLIT_SHORT[s]}"] = {
            "on_god_failure_pct": og, "worst_new_control": worst, "worst_new_control_failure_pct": ctrl[worst],
            "min_ratio_on_god_over_new_controls": og / ctrl[worst] if ctrl[worst] else float("inf"),
            "ratio_vs_heaven_knows": og / ctrl["heaven knows"] if ctrl["heaven knows"] else float("inf"),
            "ratio_vs_lord_knows": og / ctrl["lord knows"] if ctrl["lord knows"] else float("inf"),
            "ratio_vs_in_truth": og / ctrl["in truth"] if ctrl["in truth"] else float("inf"),
            "max_formal_content_over_on_god": max(ctrl[PHRASE[c]] for c in FORMAL_CONTENT) / og if og else float("nan"),
            "max_religious_over_on_god": max(ctrl[PHRASE[c]] for c in RELIGIOUS) / og if og else float("nan"),
            "all_9_holm_p_lt_01_on_god_worse": all(f["p_holm"] < 0.01 and f["direction"] == "on god worse" for f in fam),
            "n_controls_significantly_worse_than_on_god": sum(1 for f in fam if f["p_holm"] < 0.01 and f["direction"] == "control worse"),
            "P1_ratio_ge_2_all_new_controls_and_holm": (og / ctrl[worst] >= 2 if ctrl[worst] else True) and all(f["p_holm"] < 0.01 and f["direction"] == "on god worse" for f in fam),
        }
    # all-five for new controls on MultiNLI
    for (d, s) in SPLITS:
        if per_split_common[(d, s)]["five_complete"]:
            verdict["common_denominators"][SPLIT_SHORT[s]]["all_five_new_controls_pct"] = {PHRASE[c]: R[(FIVE[0], d, s, c)]["all_five_failure_pct"] for c in NEW_IDS}
            verdict["common_denominators"][SPLIT_SHORT[s]]["all_five_on_god_pct"] = R[(FIVE[0], d, s, ON_GOD)]["all_five_failure_pct"]
            verdict["common_denominators"][SPLIT_SHORT[s]]["all_six_new_controls_pct"] = {PHRASE[c]: R[(FIVE[0], d, s, c)]["all_six_failure_pct"] for c in NEW_IDS}
            verdict["common_denominators"][SPLIT_SHORT[s]]["all_six_on_god_pct"] = R[(FIVE[0], d, s, ON_GOD)]["all_six_failure_pct"]
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")

    # ---- results.md tables ----
    L = []
    L.append("# Content-word two-word controls: results\n")
    L.append("Generated by `analyze.py` from `predictions/` (new conditions, this run) and the archived crossed predictions (clean baseline and reference conditions). Preregistration: `PREDICTIONS.md`; control rationale: `CONTROLS.md`. Failure = clean-correct entailment (gold entailment, archived clean prediction entailment) predicted non-entailment under the condition. Neutral share in parentheses.\n")
    L.append("## Sanity gates (rerun clean accuracy vs archived; tolerance 0.5 pp)\n")
    L.append("| Model | Split | Rerun clean acc | Archived | Diff (pp) | Agreement with archived clean predictions | Gate |\n|---|---|---:|---:|---:|---:|---|")
    for (m, d, s), g in sorted(gates.items(), key=lambda kv: (MODELS.index(kv[0][0]), SPLITS.index((kv[0][1], kv[0][2])))):
        L.append(f"| {MODEL_LABEL[m]} | {SPLIT_SHORT[s]} | {g['clean_accuracy']*100:.2f} | {g['archived_clean_accuracy_full_split']*100:.2f} | {g['difference_pp_vs_archived_full']:+.2f} | {g['agreement_with_archived_clean_predictions']*100:.2f}% | {'PASS' if g['passed'] else 'FAIL'} |")
    for (d, s) in SPLITS:
        avail = [m for m in MODELS if (m, d, s) in cells]
        if not avail: continue
        cd = per_split_common[(d, s)]
        L.append(f"\n## {SPLIT_SHORT[s]}\n")
        L.append(f"Common clean-correct entailments: all-five N = {cd['n_five']} (paper denominator {cd['paper_denominator']}), all-six N = {cd['n_six']}.\n")
        L.append("### Failure % on each model's clean-correct entailments (neutral share of failures)\n")
        hdr = "| Condition | Group | " + " | ".join(MODEL_LABEL[m] for m in avail) + " | All-five % | All-six % |"
        L.append(hdr); L.append("|---|---|" + "---:|" * (len(avail) + 2))
        for cond in REF_IDS + NEW_IDS:
            cellsx = []
            for m in avail:
                r = R[(m, d, s, cond)]
                cellsx.append(f"{r['failure_pct']:.1f} ({r['neutral_share_pct']:.0f})" if r["failures"] else "0.0 (-)")
            r0 = R[(avail[0], d, s, cond)]
            a5 = f"{r0['all_five_failure_pct']:.1f}" if cd["five_complete"] else "n/a"
            a6 = f"{r0['all_six_failure_pct']:.1f}" if (cd["five_complete"] and DEBERTA in avail) else "n/a"
            L.append(f"| {PHRASE[cond]} | {GROUP[cond]} | " + " | ".join(cellsx) + f" | {a5} | {a6} |")
        L.append("\n### Paired accuracy change (pp) / gold-free flip rate (%)\n")
        L.append("| Condition | " + " | ".join(MODEL_LABEL[m] for m in avail) + " |"); L.append("|---|" + "---:|" * len(avail))
        for cond in REF_IDS + NEW_IDS:
            L.append(f"| {PHRASE[cond]} | " + " | ".join(f"{R[(m, d, s, cond)]['accuracy_change_pp']:+.2f} / {R[(m, d, s, cond)]['flip_rate_pct']:.1f}" for m in avail) + " |")
        L.append("\n### Exact McNemar, on god vs each new control (failures on identical clean-correct entailments), Holm over 9 tests per model\n")
        L.append("| Model | " + " | ".join(PHRASE[c] for c in NEW_IDS) + " |"); L.append("|---|" + "---:|" * len(NEW_IDS))
        for m in avail:
            cellsx = []
            for c in NEW_IDS:
                f = next(x for x in mc_rows if x["model"] == m and x["dataset"] == d and x["split"] == s and x["control_id"] == c)
                mark = "**" if f["p_holm"] < 0.01 else ""
                arrow = ">" if f["direction"] == "on god worse" else ("<" if f["direction"] == "control worse" else "=")
                ratio = f"{f['ratio_on_god_over_control']:.2f}x" if f["ratio_on_god_over_control"] != float("inf") else "inf"
                cellsx.append(f"{mark}{arrow} {ratio}, p={f['p_holm']:.1e}{mark}")
            L.append(f"| {MODEL_LABEL[m]} | " + " | ".join(cellsx) + " |")
        L.append("\n`>`: on god fails more often than the control; `<`: control fails more often; ratio = on-god failures / control failures; bold = Holm p < .01.")
    L.append("\n## Preregistered verdict (auto-generated numbers; reading in REPORT section below)\n")
    L.append("| Cell | on god fail % | worst new control (fail %) | min ratio | vs heaven knows | vs lord knows | vs in truth | max formal-content / on god | max religious / on god | all 9 Holm p<.01, on god worse | P1 |\n|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
    for k, v in verdict["cells"].items():
        L.append(f"| {k} | {v['on_god_failure_pct']:.1f} | {v['worst_new_control']} ({v['worst_new_control_failure_pct']:.1f}) | {v['min_ratio_on_god_over_new_controls']:.2f} | {v['ratio_vs_heaven_knows']:.2f} | {v['ratio_vs_lord_knows']:.2f} | {v['ratio_vs_in_truth']:.2f} | {v['max_formal_content_over_on_god']:.2f} | {v['max_religious_over_on_god']:.2f} | {v['all_9_holm_p_lt_01_on_god_worse']} | {v['P1_ratio_ge_2_all_new_controls_and_holm']} |")
    for (d, s) in SPLITS:
        cd = verdict["common_denominators"].get(SPLIT_SHORT[s], {})
        if "all_five_new_controls_pct" in cd:
            L.append(f"\nAll-five failure, {SPLIT_SHORT[s]}: on god {cd['all_five_on_god_pct']:.1f}%; new controls " + ", ".join(f"{k} {v:.1f}%" for k, v in cd["all_five_new_controls_pct"].items()) + f". All-six: on god {cd['all_six_on_god_pct']:.1f}%; " + ", ".join(f"{k} {v:.1f}%" for k, v in cd["all_six_new_controls_pct"].items()) + ".")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
