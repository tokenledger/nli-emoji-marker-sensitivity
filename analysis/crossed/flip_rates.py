#!/usr/bin/env python3
"""Analysis A: gold-free decision-flip rates.

For every model (five encoders + DeBERTa) x non-clean condition (23) x split (3),
the flip rate is the fraction of all sources whose prediction under the
condition differs from the same model's clean prediction. It uses no gold label.
A source bootstrap (2,000 resamples, seed 20260924, one shared resample index
set per split so every cell is compared on the same draws) gives 95% CIs.

Outputs (OUTPUT_DIR/flip_rates/):
  flip_rates_long.csv    one row per model x split x condition
  flip_rates_table.csv   paper-shaped table: five-encoder mean, worst cell,
                         DeBERTa, next to the paired accuracy change
  flip_ratio_by_model.csv informal-vs-control flip ratio per model
"""

from __future__ import annotations

import numpy as np

from common import (
    CONTROLS,
    COND_TEXT,
    COND_TYPE,
    DEBERTA,
    FIVE,
    GROUP,
    INFORMAL,
    MODELS,
    NONCLEAN,
    SPLITS,
    SPLIT_SHORT,
    load_all,
    write_csv,
)

import paths

OUT = paths.output("flip_rates")
BOOT_N = 2000
BOOT_SEED = 20260924


def bootstrap_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(BOOT_N, n))


def main() -> None:
    cells = load_all()
    long_rows = []
    # flip[(model, split, cond)] and acc change
    flip = {}
    dacc = {}
    for dataset, split in SPLITS:
        n = len(cells[(MODELS[0], dataset, split)]["labels"])
        # one resample set per split, derived from the frozen seed and the split name
        seed = BOOT_SEED + sum(ord(ch) for ch in f"{dataset}/{split}")
        idx = bootstrap_indices(n, seed)
        for model in MODELS:
            cell = cells[(model, dataset, split)]
            labels = cell["labels"]
            clean = cell["preds"]["c00_clean"]
            clean_acc = float(np.mean(clean == labels))
            flips = np.stack([cell["preds"][c] != clean for c in NONCLEAN]).astype(np.float32)  # 23 x N
            point = flips.mean(axis=1)
            # bootstrap: mean over resampled sources for each replicate
            boot = np.empty((len(NONCLEAN), BOOT_N), dtype=np.float32)
            for b in range(BOOT_N):
                boot[:, b] = flips[:, idx[b]].mean(axis=1)
            lo = np.percentile(boot, 2.5, axis=1)
            hi = np.percentile(boot, 97.5, axis=1)
            for k, cond in enumerate(NONCLEAN):
                pred = cell["preds"][cond]
                acc = float(np.mean(pred == labels))
                # flip destinations among flipped sources
                moved = pred != clean
                dest = {
                    "to_entailment": float(np.mean(pred[moved] == 0)) if moved.any() else float("nan"),
                    "to_neutral": float(np.mean(pred[moved] == 1)) if moved.any() else float("nan"),
                    "to_contradiction": float(np.mean(pred[moved] == 2)) if moved.any() else float("nan"),
                }
                flip[(model, split, cond)] = float(point[k])
                dacc[(model, split, cond)] = (acc - clean_acc) * 100
                long_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "split": split,
                        "condition_id": cond,
                        "inserted_text": COND_TEXT[cond],
                        "condition_type": COND_TYPE[cond],
                        "n_sources": n,
                        "flip_rate": round(float(point[k]), 6),
                        "flip_ci_low": round(float(lo[k]), 6),
                        "flip_ci_high": round(float(hi[k]), 6),
                        "flipped_sources": int(moved.sum()),
                        "clean_accuracy": round(clean_acc, 6),
                        "condition_accuracy": round(acc, 6),
                        "paired_accuracy_change_pp": round((acc - clean_acc) * 100, 4),
                        "flip_share_to_entailment": round(dest["to_entailment"], 6),
                        "flip_share_to_neutral": round(dest["to_neutral"], 6),
                        "flip_share_to_contradiction": round(dest["to_contradiction"], 6),
                        "bootstrap_replicates": BOOT_N,
                        "bootstrap_seed": seed,
                    }
                )
    write_csv(OUT / "flip_rates_long.csv", long_rows)

    # paper-shaped table
    table = []
    for cond in NONCLEAN:
        five_cells = [(m, s) for m in FIVE for _, s in SPLITS]
        five_flips = [flip[(m, s, cond)] for m, s in five_cells]
        worst_i = int(np.argmax(five_flips))
        deb = {s: flip[(DEBERTA, s, cond)] for _, s in SPLITS}
        table.append(
            {
                "group": GROUP[COND_TYPE[cond]],
                "condition_id": cond,
                "inserted_text": COND_TEXT[cond],
                "five_mean_acc_change_pp": round(float(np.mean([dacc[(m, s, cond)] for m, s in five_cells])), 2),
                "five_worst_acc_change_pp": round(float(np.min([dacc[(m, s, cond)] for m, s in five_cells])), 2),
                "five_mean_flip_pct": round(100 * float(np.mean(five_flips)), 1),
                "five_worst_flip_pct": round(100 * float(np.max(five_flips)), 1),
                "five_worst_flip_cell": f"{five_cells[worst_i][0]}/{SPLIT_SHORT[five_cells[worst_i][1]]}",
                "five_flip_pct_snli": round(100 * float(np.mean([flip[(m, 'test', cond)] for m in FIVE])), 1),
                "five_flip_pct_mnli_m": round(100 * float(np.mean([flip[(m, 'validation_matched', cond)] for m in FIVE])), 1),
                "five_flip_pct_mnli_mm": round(100 * float(np.mean([flip[(m, 'validation_mismatched', cond)] for m in FIVE])), 1),
                "deberta_mean_acc_change_pp": round(float(np.mean([dacc[(DEBERTA, s, cond)] for _, s in SPLITS])), 2),
                "deberta_mean_flip_pct": round(100 * float(np.mean(list(deb.values()))), 1),
                "deberta_flip_pct_snli": round(100 * deb["test"], 1),
                "deberta_flip_pct_mnli_m": round(100 * deb["validation_matched"], 1),
                "deberta_flip_pct_mnli_mm": round(100 * deb["validation_mismatched"], 1),
            }
        )
    order = {"Informal markers": 0, "Component words": 1, "Formal controls": 2, "Random controls": 3}
    table.sort(key=lambda r: (order[r["group"]], -r["five_mean_flip_pct"]))
    write_csv(OUT / "flip_rates_table.csv", table)

    # informal-vs-control ratio per model
    ratio_rows = []
    for model in MODELS:
        for _, split in SPLITS + (("all", "all"),):
            splits = [s for _, s in SPLITS] if split == "all" else [split]
            inf = float(np.mean([flip[(model, s, c)] for s in splits for c in INFORMAL]))
            ctl = float(np.mean([flip[(model, s, c)] for s in splits for c in CONTROLS]))
            og = float(np.mean([flip[(model, s, "c05_informal_on_god")] for s in splits]))
            ctl_max = max(float(np.mean([flip[(model, s, c)] for s in splits])) for c in CONTROLS)
            ctl_max_c = max(CONTROLS, key=lambda c: float(np.mean([flip[(model, s, c)] for s in splits])))
            ratio_rows.append(
                {
                    "model": model,
                    "split": split,
                    "informal_mean_flip_pct": round(100 * inf, 2),
                    "control_mean_flip_pct": round(100 * ctl, 2),
                    "informal_over_control_ratio": round(inf / ctl, 3),
                    "on_god_flip_pct": round(100 * og, 2),
                    "on_god_over_control_mean_ratio": round(og / ctl, 3),
                    "strongest_control": COND_TEXT[ctl_max_c],
                    "strongest_control_flip_pct": round(100 * ctl_max, 2),
                    "on_god_over_strongest_control_ratio": round(og / ctl_max, 3),
                }
            )
    write_csv(OUT / "flip_ratio_by_model.csv", ratio_rows)

    # rank agreement between flip rate and accuracy loss across the 23 conditions
    from scipy.stats import spearmanr

    five_cells = [(m, s) for m in FIVE for _, s in SPLITS]
    five_loss = {c: -float(np.mean([dacc[(m, s, c)] for m, s in five_cells])) for c in NONCLEAN}
    five_flip = {c: float(np.mean([flip[(m, s, c)] for m, s in five_cells])) for c in NONCLEAN}
    deb_loss = {c: -float(np.mean([dacc[(DEBERTA, s, c)] for _, s in SPLITS])) for c in NONCLEAN}
    deb_flip = {c: float(np.mean([flip[(DEBERTA, s, c)] for _, s in SPLITS])) for c in NONCLEAN}

    def inversions(loss: dict, fl: dict) -> list[str]:
        out = []
        for i, c in enumerate(NONCLEAN):
            for d in NONCLEAN[i + 1:]:
                if (loss[c] - loss[d]) * (fl[c] - fl[d]) < 0:
                    out.append(f"{COND_TEXT[c]} vs {COND_TEXT[d]}")
        return out

    rank_rows = []
    for name, loss, fl in (("five_encoder_mean", five_loss, five_flip), ("deberta_mean", deb_loss, deb_flip)):
        rho, p = spearmanr([loss[c] for c in NONCLEAN], [fl[c] for c in NONCLEAN])
        inv = inversions(loss, fl)
        top_loss = sorted(NONCLEAN, key=lambda c: -loss[c])[:4]
        top_flip = sorted(NONCLEAN, key=lambda c: -fl[c])[:4]
        rank_rows.append(
            {
                "scope": name,
                "n_conditions": len(NONCLEAN),
                "spearman_rho_flip_vs_accuracy_loss": round(float(rho), 4),
                "spearman_p": f"{p:.3g}",
                "pairwise_rank_inversions": len(inv),
                "pairwise_comparisons": len(NONCLEAN) * (len(NONCLEAN) - 1) // 2,
                "inverted_pairs": "; ".join(inv),
                "top4_by_accuracy_loss": ", ".join(COND_TEXT[c] for c in top_loss),
                "top4_by_flip_rate": ", ".join(COND_TEXT[c] for c in top_flip),
            }
        )
    write_csv(OUT / "flip_rank_agreement.csv", rank_rows)
    for r in rank_rows:
        print(r)

    # console summary
    print("condition | five mean flip% (worst) | DeBERTa flip% | five mean dAcc | DeBERTa dAcc")
    for r in table:
        print(
            f"{r['inserted_text']:>10} | {r['five_mean_flip_pct']:5.1f} ({r['five_worst_flip_pct']:5.1f} {r['five_worst_flip_cell']}) "
            f"| {r['deberta_mean_flip_pct']:5.1f} | {r['five_mean_acc_change_pp']:7.2f} | {r['deberta_mean_acc_change_pp']:7.2f}"
        )
    print()
    for r in ratio_rows:
        if r["split"] == "all":
            print(r)


if __name__ == "__main__":
    main()
