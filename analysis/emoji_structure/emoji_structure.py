"""Emoji loss by replacement structure and by noun (E42).

Read-only analysis of seed-42 clean-baseline predictions already on disk; no
model is run. Run through the Makefile of the analysis module:

    make emoji-structure PREDICTIONS_DIR=... DATA_DIR=...

Structure of an emoji pair (at most two replacements per pair):
  same_noun_both   one noun, replaced once in the premise and once in the hypothesis
  premise_only     one noun, replaced in the premise only
  hypothesis_only  one noun, replaced in the hypothesis only
  two_nouns_split  two different nouns, one in each sentence
  two_nouns_one    two different nouns, both in the same sentence
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

OUT = paths.output("emoji_structure")
RELEASE = str(paths.data("eval_sets"))
BACKUP = str(paths.predictions("fold_study")) + "/"
DEBERTA = str(paths.predictions("deberta")) + "/"
PREDICTIONS = {
    "ELECTRA-small": BACKUP + "publication_results/{d}/seed_42/electra_baseline_final_{s}",
    "RoBERTa-base": BACKUP + "_colab/publication_results_roberta_base42/{d}/seed_42/roberta_base_baseline_final_{s}",
    "RoBERTa-large": BACKUP + "_colab/publication_results_roberta42_{t}/{d}/seed_42/roberta_baseline_final_{s}",
    "TimeLM-21": BACKUP + "publication_results/{d}/seed_42/timelm_baseline_final_{s}",
    "BERTweet": BACKUP + "publication_results/{d}/seed_42/bertweet_baseline_final_{s}",
    "DeBERTa-v3-base": DEBERTA + "{d}/deberta_v3_base_baseline_final_{s}",
}
FIVE = [m for m in PREDICTIONS if m != "DeBERTa-v3-base"]
SPLITS = [("snli", "test", "snli"), ("multi_nli", "validation_matched", "mnli"),
          ("multi_nli", "validation_mismatched", "mnli")]
STRUCTURES = ["same_noun_both", "premise_only", "hypothesis_only", "two_nouns_split", "two_nouns_one"]
BOOTSTRAP, PERMUTATIONS, SEED = 2000, 2000, 20260927


def structure(premise_nouns, hypothesis_nouns):
    nouns = set(premise_nouns) | set(hypothesis_nouns)
    if len(nouns) == 1:
        if premise_nouns and hypothesis_nouns:
            return "same_noun_both"
        return "premise_only" if premise_nouns else "hypothesis_only"
    return "two_nouns_split" if premise_nouns and hypothesis_nouns else "two_nouns_one"


def load():
    rows = []
    for d, s, t in SPLITS:
        data = Dataset.from_file(f"{RELEASE}/{d}/final/{s}/emoji_raw/data-00000-of-00001.arrow")
        meta = {}
        for r in data:
            events = r["transform_metadata"]["events"]
            p = [e["source"].lower() for e in (events["premise"] or [])]
            h = [e["source"].lower() for e in (events["hypothesis"] or [])]
            meta[r["source_index"]] = ("+".join(sorted(set(p + h))), structure(p, h))
        for model, template in PREDICTIONS.items():
            j = json.load(open(template.format(d=d, s=s, t=t) + "/predictions.json"))
            e, c = j["emoji_raw"], j["clean_emoji_raw"]
            assert e["source_indices"] == c["source_indices"] and e["labels"] == c["labels"]
            for i, src in enumerate(e["source_indices"]):
                noun, struct = meta[src]
                clean_ok = c["predictions"][i] == c["labels"][i]
                emoji_ok = e["predictions"][i] == e["labels"][i]
                rows.append((model, d, s, src, noun, struct, 100.0 * (emoji_ok - clean_ok),
                             100.0 * (e["predictions"][i] != c["predictions"][i])))
    return pd.DataFrame(rows, columns=["model", "dataset", "split", "source_index", "noun",
                                       "structure", "change_pp", "flip_pct"])


def summarize(frame, keys, rng):
    out = []
    for key, g in frame.groupby(keys, observed=True):
        v = g.change_pp.to_numpy()
        boot = v[rng.integers(0, len(v), (BOOTSTRAP, len(v)))].mean(axis=1)
        key = key if isinstance(key, tuple) else (key,)
        out.append((*key, len(v), v.mean(), *np.percentile(boot, [2.5, 97.5]), g.flip_pct.mean()))
    return pd.DataFrame(out, columns=[*keys, "pairs", "change_pp", "ci_low", "ci_high", "flip_pct"])


def heterogeneity(frame, group, rng):
    """Permutation p for equal mean change across the levels of `group`."""
    v, g = frame.change_pp.to_numpy(), frame[group].to_numpy()

    def stat(labels):
        q = pd.Series(v).groupby(labels).agg(["mean", "size"])
        return float(((q["mean"] - v.mean()) ** 2 * q["size"]).sum())

    observed = stat(g)
    exceed = sum(stat(rng.permutation(g)) >= observed for _ in range(PERMUTATIONS))
    return (exceed + 1) / (PERMUTATIONS + 1)


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    df["structure"] = pd.Categorical(df.structure, STRUCTURES)

    overall = summarize(df, ["model", "split"], rng)
    overall.to_csv(OUT / "overall_check.csv", index=False)

    by_structure = summarize(df, ["split", "structure", "model"], rng)
    by_structure.to_csv(OUT / "structure_by_model.csv", index=False)

    single = df[df.structure.isin(STRUCTURES[:3])]
    by_noun = summarize(single[single.split == "test"], ["noun", "model"], rng)
    by_noun.to_csv(OUT / "noun_by_model_snli.csv", index=False)
    noun_structure = summarize(single[single.split == "test"], ["noun", "structure", "model"], rng)
    noun_structure.to_csv(OUT / "noun_by_structure_snli.csv", index=False)

    tests = []
    snli = df[df.split == "test"]
    for model in PREDICTIONS:
        m = snli[snli.model == model]
        ms = m[m.structure.isin(STRUCTURES[:3])]
        tests.append((model, "structure (5 levels), all SNLI pairs", len(m), heterogeneity(m, "structure", rng)))
        tests.append((model, "noun, single-noun SNLI pairs", len(ms), heterogeneity(ms, "noun", rng)))
        aligned = ms[ms.structure == "same_noun_both"]
        tests.append((model, "noun, same_noun_both SNLI pairs", len(aligned), heterogeneity(aligned, "noun", rng)))
    tests = pd.DataFrame(tests, columns=["model", "factor", "pairs", "permutation_p"])
    tests.to_csv(OUT / "heterogeneity.csv", index=False)

    pd.set_option("display.width", 250)
    lines = ["# Emoji loss by replacement structure and noun (E42)", "",
             "Generated by `emoji_structure.py`. Seed-42 clean baselines; paired accuracy change in pp",
             f"(emoji minus paired clean); 95% source-bootstrap intervals ({BOOTSTRAP} resamples) are in the CSVs.", ""]

    def table(frame, index, title, value="change_pp"):
        wide = frame.pivot_table(index=index, columns="model", values=value, observed=True)[list(PREDICTIONS)]
        wide.insert(0, "five_mean", wide[FIVE].mean(axis=1))
        counts = frame[frame.model == "BERTweet"].set_index(index).pairs
        wide.insert(0, "pairs", counts)
        lines.extend([f"## {title}", "", wide.round(2).to_markdown(), ""])
        return wide

    table(overall, "split", "A. Overall (reproduces E01 and E32)")
    for split in ["test", "validation_matched", "validation_mismatched"]:
        table(by_structure[by_structure.split == split], "structure", f"B. By structure, {split}")
    table(by_structure[by_structure.split == "test"], "structure", "C. Flip rate (%) by structure, SNLI test", "flip_pct")
    nouns = table(by_noun, "noun", "D. By noun, single-noun SNLI test pairs")
    lines[-2] = nouns.sort_values("pairs", ascending=False).round(2).to_markdown()
    aligned = noun_structure[noun_structure.structure == "same_noun_both"]
    wide = table(aligned, "noun", "E. By noun within same_noun_both, SNLI test")
    lines[-2] = wide.sort_values("pairs", ascending=False).round(2).to_markdown()
    lines.extend(["## F. Permutation tests of equal mean change (SNLI test)", "",
                  tests.round(4).to_markdown(index=False), ""])
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
