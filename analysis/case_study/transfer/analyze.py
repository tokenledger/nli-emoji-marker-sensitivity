#!/usr/bin/env python3
"""Analysis of the independent-data confirmation (preregistered in PREDICTIONS.md).

Reads PREDICTIONS_DIR/transfer/predictions/<model>__<train>__<dataset>.jsonl.gz (all 12 conditions incl.
clean from the same run).
Writes results.csv (long), mcnemar.csv, verdict.json, results.md.
"""
from __future__ import annotations
import argparse, csv, gzip, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "content_words"))
from nli_inference import MODELS, FIVE, DEBERTA, MODEL_LABEL
from task_b_conditions import CONDITIONS, CONTROL_IDS, ON_GOD, DATASETS  # local module (see run_inference.py)
from stats_utils import mcnemar_exact, holm
import paths

OUT = paths.output("case_study", "transfer")

PHRASE = {c: (p or "clean") for c, p, _ in CONDITIONS}
GROUP = {c: g for c, _, g in CONDITIONS}
NONCLEAN = [c for c, _, _ in CONDITIONS if c != "c00_clean"]
DS_LABEL = {"sick": "SICK test", "anli": "ANLI dev R1-R3"}


def parse(path: Path):
    per = {c: {} for c, _, _ in CONDITIONS}; labels = {}; rounds = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); s = int(r["source_index"])
            per[r["condition_id"]][s] = int(r["prediction"]); labels[s] = int(r["label"]); rounds[s] = r["round"]
    n = max(labels) + 1
    lab = np.array([labels[i] for i in range(n)], dtype=np.int8)
    rnd = np.array([rounds[i] for i in range(n)])
    preds = {}
    for c, m in per.items():
        assert len(m) == n, (path.name, c, len(m), n)
        preds[c] = np.array([m[i] for i in range(n)], dtype=np.int8)
    return lab, rnd, preds


def analyse(train: str, datasets: list[str]):
    cells = {}
    for m in MODELS:
        for d in datasets:
            p = paths.predictions("transfer", "predictions") / f"{m}__{train}__{d}.jsonl.gz"
            if p.is_file() and (paths.predictions("transfer", "manifests") / f"{m}__{train}__{d}.json").is_file():
                cells[(m, d)] = parse(p)
    gates = {m: json.loads((paths.predictions("transfer", "manifests") / f"{m}__{train}__{d}.json").read_text())["checkpoint_gate"] for (m, d) in cells}
    rows, mc, common = [], [], {}
    for d in datasets:
        avail = [m for m in MODELS if (m, d) in cells]
        if not avail: continue
        five = [m for m in FIVE if m in avail]
        lab, rnd, _ = cells[(avail[0], d)]
        c5 = lab == 0
        for m in five: c5 &= cells[(m, d)][2]["c00_clean"] == 0
        c6 = c5 & (cells[(DEBERTA, d)][2]["c00_clean"] == 0) if DEBERTA in avail else c5.copy()
        common[d] = {"n_five": int(c5.sum()), "n_six": int(c6.sum()), "five_complete": len(five) == 5, "n_sources": int(len(lab)),
                     "n_gold_entailment": int((lab == 0).sum())}
        for cond in NONCLEAN:
            a5 = c5.copy()
            for m in five: a5 &= cells[(m, d)][2][cond] != 0
            a6 = a5 & (cells[(DEBERTA, d)][2][cond] != 0) if DEBERTA in avail else a5
            for m in avail:
                lab, rnd, pr = cells[(m, d)]
                clean, p = pr["c00_clean"], pr[cond]
                cc = (lab == 0) & (clean == 0); fail = cc & (p != 0); nf = int(fail.sum())
                rows.append({"train": train, "model": m, "model_label": MODEL_LABEL[m], "dataset": d, "condition_id": cond, "phrase": PHRASE[cond], "group": GROUP[cond],
                             "n_sources": int(len(lab)), "clean_accuracy": float(np.mean(clean == lab)), "condition_accuracy": float(np.mean(p == lab)),
                             "accuracy_change_pp": float((np.mean(p == lab) - np.mean(clean == lab)) * 100), "flip_rate_pct": float(np.mean(p != clean) * 100),
                             "clean_correct_entailments": int(cc.sum()), "failures": nf, "failure_pct": float(100 * nf / cc.sum()) if cc.sum() else float("nan"),
                             "to_neutral": int((fail & (p == 1)).sum()), "to_contradiction": int((fail & (p == 2)).sum()),
                             "neutral_share_pct": float(100 * (fail & (p == 1)).sum() / nf) if nf else float("nan"),
                             "all_five_denominator": int(c5.sum()), "all_five_failures": int(a5.sum()),
                             "all_five_failure_pct": float(100 * a5.sum() / c5.sum()) if (len(five) == 5 and c5.sum()) else float("nan"),
                             "all_six_denominator": int(c6.sum()), "all_six_failures": int(a6.sum()),
                             "all_six_failure_pct": float(100 * a6.sum() / c6.sum()) if (len(five) == 5 and DEBERTA in avail and c6.sum()) else float("nan")})
        for m in avail:
            lab, rnd, pr = cells[(m, d)]
            cc = (lab == 0) & (pr["c00_clean"] == 0); og = cc & (pr[ON_GOD] != 0)
            fam = []
            for cond in NONCLEAN:
                if cond == ON_GOD: continue
                cf = cc & (pr[cond] != 0); t = mcnemar_exact(og[cc], cf[cc])
                fam.append({"train": train, "model": m, "model_label": MODEL_LABEL[m], "dataset": d, "comparator_id": cond, "comparator": PHRASE[cond], "group": GROUP[cond],
                            "is_control": cond in CONTROL_IDS, "n_clean_correct_entailments": int(cc.sum()), "on_god_failures": int(og.sum()), "comparator_failures": int(cf.sum()),
                            "ratio": float(og.sum() / cf.sum()) if cf.sum() else float("inf"), "only_on_god": t["only_a"], "only_comparator": t["only_b"], "p_exact": t["p"]})
            for f, a in zip(fam, holm([f["p_exact"] for f in fam])):
                f["p_holm"] = a; f["direction"] = "on god worse" if f["on_god_failures"] > f["comparator_failures"] else ("comparator worse" if f["comparator_failures"] > f["on_god_failures"] else "tie")
            mc.extend(fam)
    # per-round ANLI clean accuracy and on-god failure
    per_round = []
    if "anli" in datasets:
        for m in [m for m in MODELS if (m, "anli") in cells]:
            lab, rnd, pr = cells[(m, "anli")]
            for r in sorted(set(rnd)):
                sel = rnd == r; cc = sel & (lab == 0) & (pr["c00_clean"] == 0)
                per_round.append({"train": train, "model": m, "round": r, "n": int(sel.sum()), "clean_accuracy": float(np.mean(pr["c00_clean"][sel] == lab[sel])),
                                  "clean_correct_entailments": int(cc.sum()), "on_god_failure_pct": float(100 * (cc & (pr[ON_GOD] != 0)).sum() / cc.sum()) if cc.sum() else float("nan")})
    return cells, gates, rows, mc, common, per_round


def write(path, rs):
    if not rs: return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train", default="mnli"); ap.add_argument("--datasets", nargs="+", default=["sick", "anli"])
    a = ap.parse_args(); train = a.train
    cells, gates, rows, mc, common, per_round = analyse(train, a.datasets)
    suffix = "" if train == "mnli" else f"_{train}"
    write(OUT / f"results{suffix}.csv", rows); write(OUT / f"mcnemar{suffix}.csv", mc); write(OUT / f"anli_per_round{suffix}.csv", per_round)
    R = {(r["model"], r["dataset"], r["condition_id"]): r for r in rows}
    verdict = {"train": train, "gates": gates, "common": common, "cells": {}, "datasets": {}}
    for d in a.datasets:
        avail = [m for m in MODELS if (m, d) in cells]
        q1 = 0
        for m in avail:
            og = R[(m, d, ON_GOD)]; ctrl = {PHRASE[c]: R[(m, d, c)]["failure_pct"] for c in CONTROL_IDS}
            worst = max(ctrl, key=ctrl.get); ratio = og["failure_pct"] / ctrl[worst] if ctrl[worst] else float("inf")
            fam = [x for x in mc if x["model"] == m and x["dataset"] == d and x["is_control"]]
            holm_ok = all(x["p_holm"] < 0.01 and x["direction"] == "on god worse" for x in fam)
            Q1 = ratio >= 3 and holm_ok; Q2 = (og["neutral_share_pct"] >= 80) if og["failures"] else False
            verdict["cells"][f"{m}/{d}"] = {"clean_accuracy": og["clean_accuracy"], "clean_correct_entailments": og["clean_correct_entailments"],
                                            "on_god_failure_pct": og["failure_pct"], "on_god_neutral_share_pct": og["neutral_share_pct"], "worst_control": worst,
                                            "worst_control_failure_pct": ctrl[worst], "ratio_R": ratio, "all_controls_holm_p_lt_01_on_god_worse": holm_ok, "Q1": Q1, "Q2": Q2,
                                            "Q4_deberta_R_lt_2": (ratio < 2) if m == DEBERTA else None}
            if m in FIVE and Q1: q1 += 1
        five_avail = [m for m in FIVE if m in avail]
        dd = {"n_five_models": len(five_avail), "n_Q1_hold": q1}
        if common[d]["five_complete"]:
            og5 = R[(five_avail[0], d, ON_GOD)]["all_five_failure_pct"]; c5 = {PHRASE[c]: R[(five_avail[0], d, c)]["all_five_failure_pct"] for c in CONTROL_IDS}
            mx = max(c5.values()); dd.update({"all_five_on_god_pct": og5, "all_five_controls_pct": c5, "Q3_all_five_ratio": og5 / mx if mx else float("inf"), "Q3": (og5 / mx >= 3) if mx else og5 > 0,
                                              "all_six_on_god_pct": R[(five_avail[0], d, ON_GOD)]["all_six_failure_pct"], "all_six_controls_pct": {PHRASE[c]: R[(five_avail[0], d, c)]["all_six_failure_pct"] for c in CONTROL_IDS}})
        dd["decision"] = "held" if q1 >= 4 else ("failed" if q1 <= 2 else "partial")
        verdict["datasets"][d] = dd
    (OUT / f"verdict{suffix}.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    # markdown
    L = [f"# Independent-data confirmation of the on-god effect: results ({'MultiNLI' if train == 'mnli' else 'SNLI'}-trained seed-42 clean checkpoints)\n",
         "Generated by `analyze.py`. Preregistration: `PREDICTIONS.md`; data: `DATA.md`. Clean predictions come from the same run (condition c00). Failure = clean-correct entailment (gold entailment, clean prediction entailment) predicted non-entailment; neutral share of failures in parentheses.\n",
         "## Checkpoint gates (first 2,000 clean sources of the home split vs archived predictions on the same sources; tolerance 0.5 pp)\n",
         "| Model | Home split | Rerun acc | Archived (same sources) | Diff (pp) | Agreement | Gate |\n|---|---|---:|---:|---:|---:|---|"]
    for m, g in gates.items():
        L.append(f"| {MODEL_LABEL[m]} | {g['home_split']} | {g['clean_accuracy']*100:.2f} | {g['archived_clean_accuracy_same_sources']*100:.2f} | {g['difference_pp']:+.2f} | {g['agreement_with_archived_clean_predictions']*100:.2f}% | {'PASS' if g['passed'] else 'FAIL'} |")
    for d in a.datasets:
        avail = [m for m in MODELS if (m, d) in cells]
        if not avail: continue
        cd = common[d]
        L.append(f"\n## {DS_LABEL[d]}\n\nSources {cd['n_sources']}, gold entailment {cd['n_gold_entailment']}; common clean-correct entailments: all-five N = {cd['n_five']}, all-six N = {cd['n_six']}.\n")
        L.append("Clean accuracy: " + ", ".join(f"{MODEL_LABEL[m]} {R[(m, d, ON_GOD)]['clean_accuracy']*100:.2f}% (clean-correct ent. N={R[(m, d, ON_GOD)]['clean_correct_entailments']})" for m in avail) + ".\n")
        L.append("### Failure % on each model's clean-correct entailments (neutral share)\n")
        L.append("| Condition | Group | " + " | ".join(MODEL_LABEL[m] for m in avail) + " | All-five % | All-six % |"); L.append("|---|---|" + "---:|" * (len(avail) + 2))
        for cond in NONCLEAN:
            cs = []
            for m in avail:
                r = R[(m, d, cond)]; cs.append(f"{r['failure_pct']:.1f} ({r['neutral_share_pct']:.0f})" if r["failures"] else "0.0 (-)")
            r0 = R[(avail[0], d, cond)]
            L.append(f"| {PHRASE[cond]} | {GROUP[cond]} | " + " | ".join(cs) + f" | {r0['all_five_failure_pct']:.1f} | {r0['all_six_failure_pct']:.1f} |")
        L.append("\n### Paired accuracy change (pp) / gold-free flip rate (%)\n")
        L.append("| Condition | " + " | ".join(MODEL_LABEL[m] for m in avail) + " |"); L.append("|---|" + "---:|" * len(avail))
        for cond in NONCLEAN:
            L.append(f"| {PHRASE[cond]} | " + " | ".join(f"{R[(m, d, cond)]['accuracy_change_pp']:+.2f} / {R[(m, d, cond)]['flip_rate_pct']:.1f}" for m in avail) + " |")
        L.append("\n### Exact McNemar, on god vs each comparator on identical clean-correct entailments; Holm over the 10 comparators per model\n")
        comps = [c for c in NONCLEAN if c != ON_GOD]
        L.append("| Model | N | " + " | ".join(PHRASE[c] for c in comps) + " |"); L.append("|---|---:|" + "---:|" * len(comps))
        for m in avail:
            cs = []
            for c in comps:
                f = next(x for x in mc if x["model"] == m and x["dataset"] == d and x["comparator_id"] == c)
                b = "**" if f["p_holm"] < 0.01 else ""; arrow = ">" if f["direction"] == "on god worse" else ("<" if f["direction"] == "comparator worse" else "=")
                cs.append(f"{b}{arrow} {f['ratio']:.2f}x, p={f['p_holm']:.1e}{b}" if f["ratio"] != float("inf") else f"{b}{arrow} inf, p={f['p_holm']:.1e}{b}")
            L.append(f"| {MODEL_LABEL[m]} | {next(x for x in mc if x['model']==m and x['dataset']==d)['n_clean_correct_entailments']} | " + " | ".join(cs) + " |")
        L.append("\n`>`: on god fails more often; ratio = on-god failures / comparator failures; bold = Holm p < .01.")
        L.append("\n### Preregistered endpoints\n")
        L.append("| Model | clean acc | clean-correct ent. N | on god fail % (neutral %) | worst control (fail %) | R = on god / max control | all 5 controls Holm p<.01 | Q1 (R>=3 & Holm) | Q2 (neutral>=80%) |\n|---|---:|---:|---:|---|---:|---|---|---|")
        for m in avail:
            v = verdict["cells"][f"{m}/{d}"]
            L.append(f"| {MODEL_LABEL[m]} | {v['clean_accuracy']*100:.2f} | {v['clean_correct_entailments']} | {v['on_god_failure_pct']:.1f} ({v['on_god_neutral_share_pct']:.0f}) | {v['worst_control']} ({v['worst_control_failure_pct']:.1f}) | {v['ratio_R']:.2f} | {v['all_controls_holm_p_lt_01_on_god_worse']} | {v['Q1']} | {v['Q2']} |")
        dd = verdict["datasets"][d]
        if "all_five_on_god_pct" in dd:
            L.append(f"\nAll-five failure: on god {dd['all_five_on_god_pct']:.1f}% vs controls " + ", ".join(f"{k} {v:.1f}%" for k, v in dd["all_five_controls_pct"].items()) + f" (Q3 ratio {dd['Q3_all_five_ratio']:.1f}x, Q3 {'holds' if dd['Q3'] else 'fails'}). All-six: on god {dd['all_six_on_god_pct']:.1f}% vs " + ", ".join(f"{k} {v:.1f}%" for k, v in dd["all_six_controls_pct"].items()) + ".")
        L.append(f"\n**Decision rule ({d}): Q1 holds for {dd['n_Q1_hold']} of {dd['n_five_models']} RoBERTa/ELECTRA-family encoders -> prediction {dd['decision']}.**")
    if per_round:
        L.append("\n## ANLI per round (clean accuracy; on-god failure on clean-correct entailments)\n")
        L.append("| Model | " + " | ".join(sorted({p['round'] for p in per_round})) + " |"); L.append("|---|" + "---:|" * len({p['round'] for p in per_round}))
        for m in [m for m in MODELS if (m, "anli") in cells]:
            L.append(f"| {MODEL_LABEL[m]} | " + " | ".join(f"acc {p['clean_accuracy']*100:.1f}, N={p['clean_correct_entailments']}, on god {p['on_god_failure_pct']:.1f}%" for p in per_round if p["model"] == m) + " |")
    (OUT / f"results{suffix}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
