#!/usr/bin/env python3
"""Analysis B: commitment-marker contrast.

"honestly" and "seriously" are speaker-commitment (epistemic/emphatic) markers
with the same pragmatic function as "on god". If the "on god" effect were the
model reacting to a pragmatic shift that legitimately changes the NLI label,
the three markers should produce similar failure rates and destinations.

For each model x split, on clean-correct entailments (gold = entailment and the
model's clean prediction = entailment), we compare failure (prediction !=
entailment) under on god vs honestly and vs seriously with an exact McNemar
test on the same sources, and report the neutral-destination share.
The all-five common clean-correct scope is reported as well.

Outputs: commitment_markers.csv, commitment_markers_verdict.csv
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from common import COND_TEXT, DEBERTA, ENT, FIVE, MODELS, NEU, SPLITS, SPLIT_SHORT, load_all, write_csv

import paths

OUT = paths.output("commitment_markers")
ON_GOD = "c05_informal_on_god"
COMPARATORS = ("c16_formal_honestly", "c17_formal_seriously")


def mcnemar(a_fail: np.ndarray, b_fail: np.ndarray) -> dict:
    b = int(np.sum(a_fail & ~b_fail))  # on god fails only
    c = int(np.sum(~a_fail & b_fail))  # comparator fails only
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n else float("nan")
    return {"discordant_on_god_only": b, "discordant_comparator_only": c, "mcnemar_exact_p": p}


def fail_stats(pred: np.ndarray, scope: np.ndarray) -> dict:
    fails = scope & (pred != ENT)
    n_scope = int(scope.sum())
    n_fail = int(fails.sum())
    neutral = int(np.sum(fails & (pred == NEU)))
    return {
        "n_scope": n_scope,
        "n_fail": n_fail,
        "fail_rate": n_fail / n_scope if n_scope else float("nan"),
        "neutral_share_of_failures": neutral / n_fail if n_fail else float("nan"),
        "fails": fails,
    }


def main() -> None:
    cells = load_all()
    rows = []
    verdicts = []
    for dataset, split in SPLITS:
        labels = cells[(MODELS[0], dataset, split)]["labels"]
        gold_ent = labels == ENT
        # all-five common clean-correct entailments
        common = gold_ent.copy()
        for m in FIVE:
            common &= cells[(m, dataset, split)]["preds"]["c00_clean"] == ENT
        for model in MODELS:
            preds = cells[(model, dataset, split)]["preds"]
            scope = gold_ent & (preds["c00_clean"] == ENT)
            og = fail_stats(preds[ON_GOD], scope)
            record = {
                "model": model,
                "dataset": dataset,
                "split": split,
                "n_clean_correct_entailments": og["n_scope"],
                "on_god_fail_pct": round(100 * og["fail_rate"], 2),
                "on_god_neutral_share_pct": round(100 * og["neutral_share_of_failures"], 1),
            }
            summary = {}
            for comp in COMPARATORS:
                name = COND_TEXT[comp]
                cs = fail_stats(preds[comp], scope)
                mc = mcnemar(og["fails"], cs["fails"])
                record[f"{name}_fail_pct"] = round(100 * cs["fail_rate"], 2)
                record[f"{name}_neutral_share_pct"] = round(100 * cs["neutral_share_of_failures"], 1)
                record[f"on_god_minus_{name}_pp"] = round(100 * (og["fail_rate"] - cs["fail_rate"]), 2)
                record[f"on_god_over_{name}_ratio"] = round(og["fail_rate"] / cs["fail_rate"], 2) if cs["fail_rate"] else float("nan")
                record[f"mcnemar_b_on_god_only_vs_{name}"] = mc["discordant_on_god_only"]
                record[f"mcnemar_c_{name}_only"] = mc["discordant_comparator_only"]
                record[f"mcnemar_exact_p_vs_{name}"] = f"{mc['mcnemar_exact_p']:.3g}"
                summary[name] = (og["fail_rate"] - cs["fail_rate"], mc["mcnemar_exact_p"], cs["fail_rate"])
            rows.append(record)
            # verdict: pragmatic-shift account predicts similar effects for all three markers
            d_h, p_h, _ = summary["honestly"]
            d_s, p_s, _ = summary["seriously"]
            ratio_min = min(record["on_god_over_honestly_ratio"], record["on_god_over_seriously_ratio"])
            if d_h > 0 and d_s > 0 and p_h < 0.001 and p_s < 0.001 and ratio_min >= 2:
                verdict = "inconsistent: on god fails at least twice as often as both commitment controls (p<.001 each)"
            elif d_h > 0 and d_s > 0 and p_h < 0.05 and p_s < 0.05:
                verdict = "weakly inconsistent: on god exceeds both controls (p<.05) but by less than a factor of two"
            elif (d_h <= 0 or p_h >= 0.05) and (d_s <= 0 or p_s >= 0.05):
                verdict = "consistent: on god does not exceed either commitment control"
            else:
                verdict = "mixed: on god exceeds one commitment control but not the other"
            verdicts.append(
                {
                    "model": model,
                    "split": SPLIT_SHORT[split],
                    "on_god_fail_pct": record["on_god_fail_pct"],
                    "honestly_fail_pct": record["honestly_fail_pct"],
                    "seriously_fail_pct": record["seriously_fail_pct"],
                    "on_god_neutral_share_pct": record["on_god_neutral_share_pct"],
                    "honestly_neutral_share_pct": record["honestly_neutral_share_pct"],
                    "seriously_neutral_share_pct": record["seriously_neutral_share_pct"],
                    "p_vs_honestly": record["mcnemar_exact_p_vs_honestly"],
                    "p_vs_seriously": record["mcnemar_exact_p_vs_seriously"],
                    "pragmatic_shift_account": verdict,
                }
            )
        # all-five shared failure for the three markers
        n_common = int(common.sum())
        shared = {}
        for cond in (ON_GOD,) + COMPARATORS:
            allfail = common.copy()
            for m in FIVE:
                allfail &= cells[(m, dataset, split)]["preds"][cond] != ENT
            all_neu = common.copy()
            for m in FIVE:
                all_neu &= cells[(m, dataset, split)]["preds"][cond] == NEU
            shared[cond] = (int(allfail.sum()), int(all_neu.sum()))
        rec = {
            "model": "ALL_FIVE_SHARED",
            "dataset": dataset,
            "split": split,
            "n_clean_correct_entailments": n_common,
            "on_god_fail_pct": round(100 * shared[ON_GOD][0] / n_common, 2),
            "on_god_neutral_share_pct": round(100 * shared[ON_GOD][1] / shared[ON_GOD][0], 1) if shared[ON_GOD][0] else float("nan"),
        }
        for comp in COMPARATORS:
            name = COND_TEXT[comp]
            rec[f"{name}_fail_pct"] = round(100 * shared[comp][0] / n_common, 2)
            rec[f"{name}_neutral_share_pct"] = round(100 * shared[comp][1] / shared[comp][0], 1) if shared[comp][0] else float("nan")
        # McNemar on all-five failure sets too
        for comp in COMPARATORS:
            name = COND_TEXT[comp]
            a = common.copy(); b = common.copy()
            for m in FIVE:
                a &= cells[(m, dataset, split)]["preds"][ON_GOD] != ENT
                b &= cells[(m, dataset, split)]["preds"][comp] != ENT
            mc = mcnemar(a, b)
            rec[f"on_god_minus_{name}_pp"] = round(rec["on_god_fail_pct"] - rec[f"{name}_fail_pct"], 2)
            rec[f"on_god_over_{name}_ratio"] = round(rec["on_god_fail_pct"] / rec[f"{name}_fail_pct"], 2) if rec[f"{name}_fail_pct"] else float("nan")
            rec[f"mcnemar_b_on_god_only_vs_{name}"] = mc["discordant_on_god_only"]
            rec[f"mcnemar_c_{name}_only"] = mc["discordant_comparator_only"]
            rec[f"mcnemar_exact_p_vs_{name}"] = f"{mc['mcnemar_exact_p']:.3g}"
        rows.append(rec)
    write_csv(OUT / "commitment_markers.csv", rows)
    write_csv(OUT / "commitment_markers_verdict.csv", verdicts)
    for v in verdicts:
        print(
            f"{v['model']:>16} {v['split']:>7}: on god {v['on_god_fail_pct']:5.1f}% (neu {v['on_god_neutral_share_pct']:5.1f}) "
            f"honestly {v['honestly_fail_pct']:5.1f}% (neu {v['honestly_neutral_share_pct']:5.1f}) "
            f"seriously {v['seriously_fail_pct']:5.1f}% (neu {v['seriously_neutral_share_pct']:5.1f}) "
            f"p={v['p_vs_honestly']}/{v['p_vs_seriously']} -> {v['pragmatic_shift_account']}"
        )
    for r in rows:
        if r["model"] == "ALL_FIVE_SHARED":
            print("ALL_FIVE", r["split"], r["n_clean_correct_entailments"], r["on_god_fail_pct"], r["honestly_fail_pct"], r["seriously_fail_pct"], r["mcnemar_exact_p_vs_honestly"], r["mcnemar_exact_p_vs_seriously"])


if __name__ == "__main__":
    main()
