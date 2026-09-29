"""Build Figures 1 and 2 (per-phrase effects).

Run through the Makefile of the analysis module, after ``make crossed`` and
``make content-words``:
    make figures

Inputs (result files below OUTPUT_DIR; nothing is re-estimated):
  - flip_rates/flip_rates_table.csv
      23 crossed conditions: five-encoder mean paired accuracy change over the
      15 model-split cells (E23/E35) and the controlled DeBERTa three-split mean.
  - content_words/results.csv
      9 preregistered content-word phrases (E39), one row per model x split.
      The five-encoder mean is taken over the 15 five-encoder cells (this
      reproduces the E39 summary table); the DeBERTa value is the mean of its
      three split rows (computed here for plotting only).
  - crossed/cofailure_results_v2.csv
      all-five failure on the common clean-correct entailments (E23), MNLI-m and
      MNLI-mm, for the 23 crossed conditions. For the 9 new phrases the same
      quantity is read from results.csv (same denominators 2,518 / 2,544).

Outputs (grayscale, LNCS text width):
  - figures/phrase_effects.pdf: paired accuracy change, (a) the 23
    crossed conditions, (b) the 9 content-word phrases.
  - figures/shared_failure.pdf: all-five failure on MultiNLI, (a) markers
    and controls, (b) content-word phrases.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

FLIP = paths.result("flip_rates", "flip_rates_table.csv")
CW = paths.result("content_words", "results.csv")
COF = paths.result("crossed", "cofailure_results_v2.csv")
OUT_ACC = paths.output("figures") / "phrase_effects.pdf"
OUT_AF = paths.output("figures") / "shared_failure.pdf"

FIVE = {"electra", "roberta", "roberta_base", "timelm", "bertweet"}
LOCATIVES = {"nearby", "outside"}

GROUPS = [
    ("Informal markers", "Informal markers"),
    ("Component words", "Component words"),
    ("Formal controls", "Formal phrases"),
    ("Random controls", "Other appended words"),
    ("Content-word", "Content-word phrases (prereg.)"),
]


def mean(xs):
    return sum(xs) / len(xs)


def load():
    rows = {}  # phrase -> dict
    for r in csv.DictReader(open(FLIP)):
        rows[r["inserted_text"]] = {
            "group": r["group"],
            "cid": r["condition_id"],
            "five": float(r["five_mean_acc_change_pp"]),
            "deb": float(r["deberta_mean_acc_change_pp"]),
        }
    # all-five failure on MultiNLI for the crossed conditions
    cof = defaultdict(dict)
    for r in csv.DictReader(open(COF)):
        if (
            r["record_type"] == "condition_summary"
            and r["scope"] == "five_model_common_clean_correct_entailment"
            and r["dataset"] == "multi_nli"
        ):
            key = "m" if "matched" in r["split"] and "mis" not in r["split"] else "mm"
            cof[r["condition_id"]][key] = 100 * float(r["observed_joint_rate"])
    for p, d in rows.items():
        d["af_m"] = cof[d["cid"]]["m"]
        d["af_mm"] = cof[d["cid"]]["mm"]
    # content-word phrases
    five = defaultdict(list)
    deb = defaultdict(list)
    af = defaultdict(dict)
    for r in csv.DictReader(open(CW)):
        if not r["condition_id"].startswith(("c24", "c25", "c26", "c27", "c28",
                                            "c29", "c30", "c31", "c32")):
            continue
        p = r["phrase"]
        ch = float(r["accuracy_change_pp"])
        if r["model"] in FIVE:
            five[p].append(ch)
        elif r["model"].startswith("deberta"):
            deb[p].append(ch)
        if r["split_short"] == "MNLI-m":
            af[p]["m"] = float(r["all_five_failure_pct"])
        elif r["split_short"] == "MNLI-mm":
            af[p]["mm"] = float(r["all_five_failure_pct"])
    for p in five:
        assert len(five[p]) == 15 and len(deb[p]) == 3, p
        rows[p] = {
            "group": "Content-word",
            "five": mean(five[p]),
            "deb": mean(deb[p]),
            "af_m": af[p]["m"],
            "af_mm": af[p]["mm"],
        }
    return rows


CROSSED = GROUPS[:4]
CONTENT = GROUPS[4:]
NO_COMPONENTS = [g for g in GROUPS[:4] if g[0] != "Component words"]


def layout(rows, groups):
    """Rows top to bottom, most damaging first; group headers only when a
    panel shows more than one group (the panel title names a single group)."""
    order = []
    for gkey, glabel in groups:
        members = [(p, d) for p, d in rows.items() if d["group"] == gkey]
        members.sort(key=lambda x: x[1]["five"])
        if len(groups) > 1:
            order.append((glabel, None))
        order.extend(members)
    return order


def style_axis(ax, order):
    n = len(order)
    ys, labels = [], []
    for i, (lab, d) in enumerate(order):
        y = n - 1 - i
        if d is None:
            ax.axhline(y, color="0.75", lw=0.4, zorder=1)
            ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7, fontweight="bold")
            continue
        ys.append(y)
        labels.append(lab + (" (loc.)" if lab in LOCATIVES else ""))
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, n - 0.2)
    ax.grid(axis="x", color="0.92", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return {lab: n - 1 - i for i, (lab, d) in enumerate(order) if d is not None}


def two_panel(n_left, n_right):
    """Left and right panels with equal row height, right panel top-aligned."""
    fig = plt.figure(figsize=(4.8, 0.11 * n_left + 0.5))
    gs = fig.add_gridspec(n_left, 2, width_ratios=[1, 1], wspace=0.75)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:n_right, 1])
    return fig, ax1, ax2


def main():
    rows = load()
    assert len(rows) == 32, len(rows)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7.3,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.0,
        "pdf.fonttype": 42,
    })
    ctrl = [d["five"] for d in rows.values()
            if d["group"] in ("Formal controls", "Random controls")]

    # Figure 1: paired accuracy change.
    left, right = layout(rows, CROSSED), layout(rows, CONTENT)
    fig, ax1, ax2 = two_panel(len(left), len(right))
    for ax, order in ((ax1, left), (ax2, right)):
        pos = style_axis(ax, order)
        ax.axvspan(min(ctrl), max(ctrl), color="0.88", zorder=0, lw=0)
        ax.axvline(0, color="0.5", lw=0.5, zorder=1)
        for lab, d in order:
            if d is None:
                continue
            ax.plot(d["five"], pos[lab], "o", ms=4, color="black", zorder=3)
            ax.plot(d["deb"], pos[lab], "o", ms=4, mfc="white", mec="black",
                    mew=0.7, zorder=4)
        ax.set_xlim(-33, 1)
        ax.set_xlabel("Paired accuracy change (pp)")
    ax1.set_title("(a) Crossed study", fontsize=8, loc="left")
    ax2.set_title("(b) Content-word phrases", fontsize=8, loc="left")
    h = [
        plt.Line2D([], [], ls="", marker="o", ms=4, color="black",
                   label="five-encoder mean"),
        plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white",
                   mec="black", label="DeBERTa-v3-base"),
        plt.Rectangle((0, 0), 1, 1, color="0.88", label="control range"),
    ]
    ax2.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.02, -0.4),
               frameon=False, fontsize=7.4, handletextpad=0.4)
    fig.savefig(OUT_ACC, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT_ACC)

    # Figure 2: all-five failure on MultiNLI.
    left, right = layout(rows, NO_COMPONENTS), layout(rows, CONTENT)
    fig, ax1, ax2 = two_panel(len(left), len(right))
    for ax, order in ((ax1, left), (ax2, right)):
        pos = style_axis(ax, order)
        for lab, d in order:
            if d is None:
                continue
            ax.plot(d["af_m"], pos[lab], "s", ms=3.4, color="black", zorder=3)
            ax.plot(d["af_mm"], pos[lab], "^", ms=3.8, mfc="white",
                    mec="black", mew=0.7, zorder=4)
        ax.set_xlim(-2, 88)
        ax.set_xlabel("All-five failure (%)")
    ax1.set_title("(a) Markers and controls", fontsize=8, loc="left")
    ax2.set_title("(b) Content-word phrases", fontsize=8, loc="left")
    h = [
        plt.Line2D([], [], ls="", marker="s", ms=3.4, color="black",
                   label="MultiNLI matched"),
        plt.Line2D([], [], ls="", marker="^", ms=3.8, mfc="white",
                   mec="black", label="MultiNLI mismatched"),
    ]
    ax2.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.02, -0.4),
               frameon=False, fontsize=7.4, handletextpad=0.4)
    fig.savefig(OUT_AF, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT_AF)


if __name__ == "__main__":
    main()
