"""Content-word coding of the 32 appended phrases against flip rate (E43).

Read-only analysis of flip rates already on disk; no model is run. Run through
the Makefile of the analysis module, after ``make crossed``:

    make content-words PREDICTIONS_DIR=... DATA_DIR=...

Coding rule (strict): a phrase is content-bearing if at least one of its words
is an English noun, adjective, or lexical verb in its primary dictionary sense.
Abbreviations and respellings (fr, frfr, tbh, ngl, istg, deadass),
prepositions, determiners, pronouns, the negator "no", and adverbs are not.
Two alternative codings test the two judgment calls: adverbs counted as content
words, and "deadass" counted as content-bearing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

OUT = paths.output("content_word_coding")
CROSSED = paths.result("flip_rates", "flip_rates_long.csv")
CONTENT = paths.result("content_words", "results.csv")
FIVE = ["electra", "roberta", "roberta_base", "timelm", "bertweet"]

# phrase: (family, strict coding, content words under the strict rule)
PHRASES = {
    "on god": ("informal marker", 1, "god"), "no cap": ("informal marker", 1, "cap"),
    "real talk": ("informal marker", 1, "real, talk"), "deadass": ("informal marker", 0, ""),
    "frfr": ("informal marker", 0, ""), "ngl": ("informal marker", 0, ""),
    "tbh": ("informal marker", 0, ""), "istg": ("informal marker", 0, ""),
    "fr": ("informal marker", 0, ""),
    "no": ("component word", 0, ""), "god": ("component word", 1, "god"),
    "talk": ("component word", 1, "talk"), "real": ("component word", 1, "real"),
    "cap": ("component word", 1, "cap"), "on": ("component word", 0, ""),
    "seriously": ("formal control", 0, ""), "honestly": ("formal control", 0, ""),
    "in fact": ("formal control", 1, "fact"),
    "nearby": ("other control", 0, ""), "outside": ("other control", 0, ""),
    "in the": ("other control", 0, ""), "with it": ("other control", 0, ""),
    "at this": ("other control", 0, ""),
    "in truth": ("content-word phrase", 1, "truth"), "in essence": ("content-word phrase", 1, "essence"),
    "in short": ("content-word phrase", 1, "short"), "in reality": ("content-word phrase", 1, "reality"),
    "heaven knows": ("content-word phrase", 1, "heaven, knows"),
    "lord knows": ("content-word phrase", 1, "lord, knows"),
    "by heaven": ("content-word phrase", 1, "heaven"),
    "good heavens": ("content-word phrase", 1, "good, heavens"),
    "for real": ("content-word phrase", 1, "real"),
}
ADVERBS = {"seriously", "honestly", "nearby", "outside"}
PERMUTATIONS, SEED = 20000, 20260927


def cell_rates():
    a = pd.read_csv(CROSSED)
    b = pd.read_csv(CONTENT)
    b = b[b.source == "new_run"]
    a = a.rename(columns={"inserted_text": "phrase", "paired_accuracy_change_pp": "accuracy_change_pp"})
    a["flip_rate_pct"] = 100 * a.flip_rate
    keep = ["model", "split", "phrase", "flip_rate_pct", "accuracy_change_pp"]
    return pd.concat([a[keep], b[keep]], ignore_index=True)


def permutation_p(values, coding, rng):
    observed = values[coding == 1].mean() - values[coding == 0].mean()
    draws = np.array([(lambda c: values[c == 1].mean() - values[c == 0].mean())(rng.permutation(coding))
                      for _ in range(PERMUTATIONS)])
    return observed, (np.sum(draws >= observed) + 1) / (PERMUTATIONS + 1)


def main():
    rng = np.random.default_rng(SEED)
    cells = cell_rates()
    assert set(cells.phrase) == set(PHRASES), set(cells.phrase) ^ set(PHRASES)
    five = cells[cells.model.isin(FIVE)].groupby("phrase")[["flip_rate_pct", "accuracy_change_pp"]].mean()
    deb = cells[cells.model == "deberta_v3_base"].groupby("phrase")[["flip_rate_pct", "accuracy_change_pp"]].mean()
    t = pd.DataFrame.from_dict(PHRASES, orient="index", columns=["family", "content_strict", "content_words"])
    t["content_adverbs"] = [1 if p in ADVERBS else c for p, c in t.content_strict.items()]
    t["content_deadass"] = [1 if p == "deadass" else c for p, c in t.content_strict.items()]
    t["words"] = [len(p.split()) for p in t.index]
    t["five_flip_pct"], t["five_acc_change_pp"] = five.flip_rate_pct, five.accuracy_change_pp
    t["deberta_flip_pct"], t["deberta_acc_change_pp"] = deb.flip_rate_pct, deb.accuracy_change_pp
    t = t.sort_values("five_flip_pct", ascending=False)
    t.round(2).to_csv(OUT / "phrase_coding.csv", index_label="phrase")

    subsets = {"all 32 phrases": t, "26 phrases (no component words)": t[t.family != "component word"],
               "two-word phrases only": t[t.words == 2]}
    rows = []
    for name, s in subsets.items():
        for coding in ["content_strict", "content_adverbs", "content_deadass"]:
            for outcome in ["five_flip_pct", "deberta_flip_pct"]:
                v, c = s[outcome].to_numpy(), s[coding].to_numpy()
                diff, p = permutation_p(v, c, rng)
                u = mannwhitneyu(v[c == 1], v[c == 0], alternative="greater")
                rows.append((name, coding, outcome, int(c.sum()), int((1 - c).sum()),
                             v[c == 1].mean(), v[c == 0].mean(), np.median(v[c == 1]), np.median(v[c == 0]),
                             diff, p, u.statistic / (c.sum() * (1 - c).sum()), u.pvalue))
    r = pd.DataFrame(rows, columns=["subset", "coding", "outcome", "n_content", "n_other", "mean_content",
                                    "mean_other", "median_content", "median_other", "mean_difference",
                                    "permutation_p_one_sided", "prob_content_exceeds_other", "mann_whitney_p_one_sided"])
    r.round(4).to_csv(OUT / "contrasts.csv", index=False)
    by_length = pd.concat({c: t.groupby(["words", c])[["five_flip_pct", "deberta_flip_pct"]]
                           .agg(["mean", "median", "min", "max", "size"]).rename_axis(["words", "content"])
                           for c in ["content_strict", "content_adverbs", "content_deadass"]}, names=["coding"])
    by_length.round(2).to_csv(OUT / "by_length.csv")
    rho = spearmanr(t.five_flip_pct, t.five_acc_change_pp)

    lines = ["# Content-word coding of 32 appended phrases (E43)", "",
             "Generated by `content_word_coding.py`. Flip rate: five-encoder mean over 15 model-split cells;",
             "DeBERTa-v3-base mean over three splits. Seed 42.", "",
             f"Spearman rho between five-encoder flip rate and accuracy change over 32 phrases: {rho.statistic:.3f}.", "",
             "## Phrases", "", t.round(2).to_markdown(), "", "## Contrasts", "", r.round(4).to_markdown(index=False), "",
             "## By number of words", "", by_length.round(2).to_markdown(), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
    print(t.round(2).drop(columns=["content_words"]).to_string())
    print(r.round(3).to_string())


if __name__ == "__main__":
    main()
