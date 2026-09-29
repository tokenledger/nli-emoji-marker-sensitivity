#!/usr/bin/env python3
"""Analysis C: "on god" on ChaosNLI high-agreement entailments.

ChaosNLI (Nie, Zhou, Bansal 2020) re-annotated 1,599 MultiNLI dev-matched pairs
with 100 crowd labels each. Its SNLI portion covers SNLI *dev*, not test, so
only the MNLI-m portion intersects our final evaluation split. We join
ChaosNLI to our MultiNLI matched sources by pairID (ChaosNLI ``uid`` is the
MultiNLI pairID) through the release source table, verify the join by exact
premise/hypothesis text after whitespace normalisation, and restrict to
sources whose inherited gold label is entailment and on which >= 80 (also
>= 90) of the 100 ChaosNLI annotators chose entailment.

On these subsets we recompute, for the five encoders and DeBERTa, the failure
rate on clean-correct entailments, the all-five failure rate, and the neutral
share, for on god, no cap, nearby, seriously, honestly, and in fact, next to
the same quantities on the full MNLI-m set.

Data source: the Dropbox link in the ChaosNLI README returns "File Deleted"
(checked 2026-09-24); we use the Hugging Face mirror
``tasksource/chaos-mnli-ambiguity`` (``chaos_mnli.jsonl``, 1,599 rows, 100
labels each), stored by the dataprep module under ``DATA_DIR/external/``.

Outputs: chaosnli_join.csv, chaosnli_subset_results.csv,
         chaosnli_failure_vs_agreement.csv
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crossed"))
from common import COND_TEXT, DEBERTA, ENT, FIVE, MODELS, NEU, load_all, source_rows, write_csv  # noqa: E402
import paths  # noqa: E402

OUT = paths.output("case_study", "chaosnli")
MIRROR = paths.data("external", "chaosnli_mnli.jsonl")
SOURCE_ARROW = paths.data(
    "source", "multi_nli", "final", "validation_matched", "data-00000-of-00001.arrow"
)
CONDS = (
    "c05_informal_on_god",
    "c09_informal_no_cap",
    "c19_random_nearby",
    "c17_formal_seriously",
    "c16_formal_honestly",
    "c18_formal_in_fact",
)
THRESHOLDS = (80, 90)
BOOT_N = 2000
BOOT_SEED = 20260924


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main() -> None:
    chaos = [json.loads(line) for line in MIRROR.open(encoding="utf-8")]
    assert len(chaos) == 1599 and all(sum(r["label_counter"].values()) == 100 for r in chaos)
    by_uid = {r["uid"]: r for r in chaos}
    assert len(by_uid) == 1599

    with pa.memory_map(str(SOURCE_ARROW)) as handle:
        table = ipc.open_stream(handle).read_all()
    pair_ids = table.column("pairID").to_pylist()
    src_rows = source_rows("multi_nli", "validation_matched")
    assert len(src_rows) == len(pair_ids) == table.num_rows
    # source_index i <-> release row i (checked by text + label identity)
    prem = table.column("premise").to_pylist()
    hyp = table.column("hypothesis").to_pylist()
    lab = table.column("label").to_pylist()
    for r in src_rows:
        i = r["source_index"]
        assert prem[i] == r["premise"] and hyp[i] == r["hypothesis"] and lab[i] == r["label"]

    n_src = len(src_rows)
    ent_count = np.full(n_src, -1, dtype=np.int16)
    joined = []
    text_mismatch = 0
    for i, pid in enumerate(pair_ids):
        c = by_uid.get(pid)
        if c is None:
            continue
        if norm(c["premise"]) != norm(prem[i]) or norm(c["hypothesis"]) != norm(hyp[i]):
            text_mismatch += 1
            continue
        ent_count[i] = c["label_counter"].get("e", 0)
        joined.append(
            {
                "source_index": i,
                "pairID": pid,
                "gold_label": lab[i],
                "chaos_e": c["label_counter"].get("e", 0),
                "chaos_n": c["label_counter"].get("n", 0),
                "chaos_c": c["label_counter"].get("c", 0),
                "chaos_majority": c["majority_label"],
                "chaos_old_label": c["old_label"],
                "chaos_entropy": c["entropy"],
            }
        )
    write_csv(OUT / "chaosnli_join.csv", joined)
    join_n = len(joined)
    unmatched_uid = 1599 - join_n - text_mismatch
    print(f"joined {join_n} of 1599 ChaosNLI MNLI-m items by pairID (text mismatches {text_mismatch}, uid not in source {unmatched_uid})")

    cells = load_all()
    labels = cells[(MODELS[0], "multi_nli", "validation_matched")]["labels"]
    assert np.array_equal(labels, np.array(lab, dtype=np.int8))
    gold_ent = labels == ENT
    in_chaos = ent_count >= 0
    print(f"gold-entailment sources in join: {int((gold_ent & in_chaos).sum())}; "
          f"with >=80 human entailment: {int((gold_ent & (ent_count >= 80)).sum())}; >=90: {int((gold_ent & (ent_count >= 90)).sum())}")
    # ChaosNLI majority agrees with gold entailment on:
    maj_e = np.zeros(n_src, dtype=bool)
    for j in joined:
        maj_e[j["source_index"]] = j["chaos_majority"] == "e"
    print(f"ChaosNLI majority = entailment among gold-entailment joined: {int((gold_ent & in_chaos & maj_e).sum())}")

    rng = np.random.default_rng(BOOT_SEED)
    results = []
    subsets = [("full_mnli_m", np.ones(n_src, dtype=bool)), ("chaosnli_joined", in_chaos)]
    subsets += [(f"chaos_e_ge_{t}", ent_count >= t) for t in THRESHOLDS]
    for subset_name, mask in subsets:
        base = gold_ent & mask
        common = base.copy()
        for m in FIVE:
            common &= cells[(m, "multi_nli", "validation_matched")]["preds"]["c00_clean"] == ENT
        for cond in CONDS:
            # per model
            for model in MODELS:
                preds = cells[(model, "multi_nli", "validation_matched")]["preds"]
                scope = base & (preds["c00_clean"] == ENT)
                fails = scope & (preds[cond] != ENT)
                n, k = int(scope.sum()), int(fails.sum())
                neu = int(np.sum(fails & (preds[cond] == NEU)))
                lo, hi = wilson(k, n)
                results.append(
                    {
                        "subset": subset_name,
                        "condition": COND_TEXT[cond],
                        "model": model,
                        "n_clean_correct_entailments": n,
                        "n_fail": k,
                        "fail_pct": round(100 * k / n, 2) if n else float("nan"),
                        "fail_ci95_low_pct": round(100 * lo, 2),
                        "fail_ci95_high_pct": round(100 * hi, 2),
                        "neutral_share_pct": round(100 * neu / k, 1) if k else float("nan"),
                    }
                )
            # all five
            allfail = common.copy()
            allneu = common.copy()
            for m in FIVE:
                p = cells[(m, "multi_nli", "validation_matched")]["preds"][cond]
                allfail &= p != ENT
                allneu &= p == NEU
            n, k = int(common.sum()), int(allfail.sum())
            lo, hi = wilson(k, n)
            results.append(
                {
                    "subset": subset_name,
                    "condition": COND_TEXT[cond],
                    "model": "ALL_FIVE",
                    "n_clean_correct_entailments": n,
                    "n_fail": k,
                    "fail_pct": round(100 * k / n, 2) if n else float("nan"),
                    "fail_ci95_low_pct": round(100 * lo, 2),
                    "fail_ci95_high_pct": round(100 * hi, 2),
                    "neutral_share_pct": round(100 * int(allneu.sum()) / k, 1) if k else float("nan"),
                }
            )
    write_csv(OUT / "chaosnli_subset_results.csv", results)

    # Does the failure set concentrate on items humans dispute? For ALL_FIVE the
    # contrast is all-five-fail vs mixed-or-all-survive (not all-fail vs all-survive).
    # Compare the human entailment count between failed and surviving sources
    # (within the joined, gold-entailment, clean-correct scope).
    agree_rows = []
    for cond in CONDS:
        for model in MODELS + ("ALL_FIVE",):
            if model == "ALL_FIVE":
                scope = gold_ent & in_chaos
                for m in FIVE:
                    scope &= cells[(m, "multi_nli", "validation_matched")]["preds"]["c00_clean"] == ENT
                fails = scope.copy()
                for m in FIVE:
                    fails &= cells[(m, "multi_nli", "validation_matched")]["preds"][cond] != ENT
            else:
                preds = cells[(model, "multi_nli", "validation_matched")]["preds"]
                scope = gold_ent & in_chaos & (preds["c00_clean"] == ENT)
                fails = scope & (preds[cond] != ENT)
            survive = scope & ~fails
            e_fail = ent_count[fails]
            e_surv = ent_count[survive]
            if len(e_fail) and len(e_surv):
                u = mannwhitneyu(e_fail, e_surv, alternative="two-sided").pvalue
            else:
                u = float("nan")
            agree_rows.append(
                {
                    "condition": COND_TEXT[cond],
                    "model": model,
                    "n_fail": int(len(e_fail)),
                    "n_not_failed": int(len(e_surv)),
                    "mean_human_entailment_count_failed": round(float(e_fail.mean()), 1) if len(e_fail) else float("nan"),
                    "mean_human_entailment_count_not_failed": round(float(e_surv.mean()), 1) if len(e_surv) else float("nan"),
                    "share_failed_with_e_ge_80_pct": round(100 * float(np.mean(e_fail >= 80)), 1) if len(e_fail) else float("nan"),
                    "share_not_failed_with_e_ge_80_pct": round(100 * float(np.mean(e_surv >= 80)), 1) if len(e_surv) else float("nan"),
                    "mann_whitney_p": f"{u:.3g}",
                }
            )
    write_csv(OUT / "chaosnli_failure_vs_agreement.csv", agree_rows)

    # Exact composition of the joined common clean-correct scope under on god,
    # all-five failures per human-agreement band, and a paired all-five
    # contrast (on god vs no cap) on the >= 80 subset.
    common_join = gold_ent & in_chaos
    for m in FIVE:
        common_join &= cells[(m, "multi_nli", "validation_matched")]["preds"]["c00_clean"] == ENT
    og_fail_count = np.zeros(n_src, dtype=np.int8)
    for m in FIVE:
        og_fail_count += (cells[(m, "multi_nli", "validation_matched")]["preds"]["c05_informal_on_god"] != ENT).astype(np.int8)
    comp = {
        "n_joined_common_clean_correct": int(common_join.sum()),
        "all_five_fail": int(np.sum(common_join & (og_fail_count == 5))),
        "mixed_1_to_4_fail": int(np.sum(common_join & (og_fail_count >= 1) & (og_fail_count <= 4))),
        "all_five_survive": int(np.sum(common_join & (og_fail_count == 0))),
    }
    band_rows = []
    for band, mask in (("e_lt_80", ent_count < 80), ("e_80_to_89", (ent_count >= 80) & (ent_count < 90)), ("e_ge_80", ent_count >= 80), ("e_ge_90", ent_count >= 90)):
        scope = common_join & mask
        k = int(np.sum(scope & (og_fail_count == 5)))
        n = int(scope.sum())
        band_rows.append({"human_entailment_band": band, "n_common_clean_correct": n, "all_five_on_god_failures": k, "all_five_fail_pct": round(100 * k / n, 1) if n else float("nan")})
    write_csv(OUT / "chaosnli_on_god_by_agreement_band.csv", band_rows)
    print("composition of joined common clean-correct under on god:", comp)
    for r in band_rows:
        print(r)

    # all 23 conditions on the >= 80 subset (all-five failure, five-mean, DeBERTa)
    from common import NONCLEAN
    from scipy.stats import binomtest

    ge80 = gold_ent & (ent_count >= 80)
    common80 = ge80.copy()
    for m in FIVE:
        common80 &= cells[(m, "multi_nli", "validation_matched")]["preds"]["c00_clean"] == ENT
    n80 = int(common80.sum())
    allfail80 = {}
    all_rows = []
    for cond in NONCLEAN:
        af = common80.copy()
        an = common80.copy()
        for m in FIVE:
            p = cells[(m, "multi_nli", "validation_matched")]["preds"][cond]
            af &= p != ENT
            an &= p == NEU
        allfail80[cond] = af
        per_model = []
        for model in MODELS:
            preds = cells[(model, "multi_nli", "validation_matched")]["preds"]
            scope = ge80 & (preds["c00_clean"] == ENT)
            per_model.append((model, float(np.mean(preds[cond][scope] != ENT)), int(scope.sum())))
        five_mean = float(np.mean([v for m, v, _ in per_model if m in FIVE]))
        deb = next(v for m, v, _ in per_model if m == DEBERTA)
        deb_n = next(n for m, _, n in per_model if m == DEBERTA)
        lo, hi = wilson(int(af.sum()), n80)
        all_rows.append(
            {
                "condition_id": cond,
                "inserted_text": COND_TEXT[cond],
                "n_all_five_clean_correct_ge80": n80,
                "all_five_failures": int(af.sum()),
                "all_five_fail_pct": round(100 * int(af.sum()) / n80, 1),
                "all_five_ci95_low_pct": round(100 * lo, 1),
                "all_five_ci95_high_pct": round(100 * hi, 1),
                "all_five_neutral_failures": int(an.sum()),
                "five_encoder_mean_fail_pct": round(100 * five_mean, 1),
                "deberta_fail_pct": round(100 * deb, 1),
                "deberta_n_clean_correct_ge80": deb_n,
            }
        )
    all_rows.sort(key=lambda r: -r["all_five_failures"])
    write_csv(OUT / "chaosnli_ge80_all_conditions.csv", all_rows)
    # paired contrasts on god vs every other condition (all-five failure, same 83 sources)
    pair_rows = []
    og = allfail80["c05_informal_on_god"]
    for cond in NONCLEAN:
        if cond == "c05_informal_on_god":
            continue
        other = allfail80[cond]
        b = int(np.sum(og & ~other))
        c = int(np.sum(~og & other))
        p = binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
        pair_rows.append({"comparator": COND_TEXT[cond], "n": n80, "on_god_all_five_failures": int(og.sum()), "comparator_all_five_failures": int(other.sum()), "discordant_on_god_only": b, "discordant_comparator_only": c, "mcnemar_exact_p": f"{p:.3g}"})
    write_csv(OUT / "chaosnli_ge80_paired_contrasts.csv", pair_rows)
    print(f"\n>= 80 subset, all-five clean-correct n = {n80}")
    for r in all_rows:
        print(f"  {r['inserted_text']:>10} all-five {r['all_five_failures']:>3} ({r['all_five_fail_pct']:5.1f}%) five-mean {r['five_encoder_mean_fail_pct']:5.1f} DeBERTa {r['deberta_fail_pct']:5.1f}")
    for r in pair_rows:
        if r["comparator"] in ("no cap", "nearby", "seriously", "honestly", "in fact", "deadass", "real talk"):
            print("  paired:", r)

    print()
    print("subset | condition | ALL_FIVE fail% [n] | DeBERTa fail% [n] | five-model mean fail% | on-god neutral share (all-five)")
    for subset_name, _ in subsets:
        for cond in CONDS:
            sub = [r for r in results if r["subset"] == subset_name and r["condition"] == COND_TEXT[cond]]
            a5 = next(r for r in sub if r["model"] == "ALL_FIVE")
            de = next(r for r in sub if r["model"] == DEBERTA)
            five = [r for r in sub if r["model"] in FIVE]
            print(
                f"{subset_name:>16} | {COND_TEXT[cond]:>9} | {a5['fail_pct']:5.1f} [{a5['n_clean_correct_entailments']}] "
                f"({a5['fail_ci95_low_pct']:.1f}-{a5['fail_ci95_high_pct']:.1f}) | {de['fail_pct']:5.1f} [{de['n_clean_correct_entailments']}] | "
                f"{np.mean([r['fail_pct'] for r in five]):5.1f} | neu {a5['neutral_share_pct']}"
            )
    print()
    for r in agree_rows:
        if r["condition"] in ("on god", "no cap", "seriously") and r["model"] in ("ALL_FIVE", DEBERTA, "roberta"):
            print(r)


if __name__ == "__main__":
    main()
