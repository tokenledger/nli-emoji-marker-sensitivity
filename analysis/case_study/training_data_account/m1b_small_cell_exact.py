#!/usr/bin/env python3
"""M1b: exact multinomial goodness-of-fit (vs the dataset's overall label distribution) for small cells,
and distinct-premise counts for premise "god" rows."""
import re, csv
from itertools import product
from math import factorial
from pathlib import Path
from datasets import load_from_disk
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402
OUT = paths.output("case_study", "training_data_account")
REL = paths.data("source")

def exact_multinomial_p(obs, probs):
    n = sum(obs)
    def pmf(c):
        p = factorial(n)
        for k, q in zip(c, probs):
            p = p * q ** k / factorial(k)
        return p
    p_obs = pmf(obs); total = 0.0
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = (a, b, n - a - b)
            pc = pmf(c)
            if pc <= p_obs * (1 + 1e-12):
                total += pc
    return total

rows = []
for name in ("multi_nli", "snli"):
    ds = load_from_disk(str(REL / f"{name}/training"))
    lab = ds["label"]; n = len(lab)
    probs = [sum(1 for l in lab if l == i) / n for i in range(3)]
    for field in ("hypothesis", "premise"):
        texts = ds[field]
        for term, pat in (("on god", r"\bon\s+god\b"), ("my god (not oh my god)", None), ("oh my god", r"\boh\s+my\s+god\b"),
                          ("no cap", r"\bno\s+cap\b"), ("god (any)", r"\bgod\b")):
            if pat is None:
                r1 = re.compile(r"\bmy\s+god\b", re.I); r2 = re.compile(r"\boh\s+my\s+god\b", re.I)
                idx = [i for i, t in enumerate(texts) if r1.search(t) and not r2.search(t)]
            else:
                r1 = re.compile(pat, re.I); idx = [i for i, t in enumerate(texts) if r1.search(t)]
            obs = [sum(1 for i in idx if lab[i] == k) for k in range(3)]
            p = exact_multinomial_p(obs, probs) if 0 < sum(obs) <= 60 else float("nan")
            rows.append(dict(dataset=name, field=field, term=term, n_rows=len(idx), n_distinct_text=len({texts[i] for i in idx}),
                             entailment=obs[0], neutral=obs[1], contradiction=obs[2], exact_multinomial_p=p))
            print(rows[-1])
with open(OUT / "m1b_small_cell_exact.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
