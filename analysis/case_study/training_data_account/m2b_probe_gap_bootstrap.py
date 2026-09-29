#!/usr/bin/env python3
"""M2b: paired source-bootstrap CIs for the MultiNLI hypothesis-only probe's neutral-shift gaps
(on god minus in fact, on god minus nearby, on god minus god)."""
import gzip, json, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402
OUT = paths.output("case_study", "training_data_account")
REL = paths.data("source")
EVAL = paths.data("crossed")
SEED = 20260924; B = 5000
train = load_from_disk(str(REL / "multi_nli/training"))
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
X = vec.fit_transform(train["hypothesis"]); clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(X, np.array(train["label"]))
CONDS = {"on god": "c05_informal_on_god", "god": "c13_component_god", "on": "c12_component_on", "nearby": "c19_random_nearby",
         "in fact": "c18_formal_in_fact", "in the": "c21_random_in_the", "seriously": "c17_formal_seriously"}
out = []
for split in ("validation_matched", "validation_mismatched"):
    by = defaultdict(dict)
    with gzip.open(EVAL / f"multi_nli__final__{split}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line); by[r["condition_id"]][r["source_index"]] = (r["hypothesis"], r["label"])
    idx = sorted(by["c00_clean"]); y = np.array([by["c00_clean"][i][1] for i in idx])
    pc = clf.predict(vec.transform([by["c00_clean"][i][0] for i in idx]))
    ent = [i for i, p, l in zip(idx, pc, y) if l == 0 and p == 0]
    neu = {k: (clf.predict(vec.transform([by[c][i][0] for i in ent])) == 1).astype(int) for k, c in CONDS.items()}
    rng = np.random.RandomState(SEED); n = len(ent)
    for k in neu: print(split, k, f"{neu[k].mean()*100:.2f}%")
    for a, b in (("on god", "in fact"), ("on god", "nearby"), ("on god", "god"), ("on god", "in the"), ("god", "nearby")):
        d = neu[a] - neu[b]; obs = d.mean() * 100
        boots = np.array([d[rng.randint(0, n, n)].mean() * 100 for _ in range(B)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(dict(split=split, n_sources=n, contrast=f"{a} minus {b}", gap_pp=obs, ci_low=lo, ci_high=hi, bootstrap_reps=B, seed=SEED))
        print(out[-1])
with open(OUT / "m2b_probe_gap_bootstrap.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
