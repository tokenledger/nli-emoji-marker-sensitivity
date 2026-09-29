#!/usr/bin/env python3
"""M1: training-data counts of "god" and the other suffix words by gold label (and genre)."""
import re, csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import chi2_contingency
from datasets import load_from_disk

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402
OUT = paths.output("case_study", "training_data_account")
REL = paths.data("source")
LABELS = ["entailment", "neutral", "contradiction"]

# god sub-patterns (mutually exclusive, applied in order)
GOD_PATTERNS = [
    ("oh my god", re.compile(r"\boh\s+my\s+god\b", re.I)),
    ("my god (not oh my god)", re.compile(r"\bmy\s+god\b", re.I)),
    ("god elsewhere", re.compile(r"\bgod\b", re.I)),
]
GOD_ANY = re.compile(r"\bgod\b", re.I)

MARKERS = ["fr", "tbh", "ngl", "real talk", "on god", "istg", "frfr", "deadass", "no cap"]
COMPONENTS = ["real", "talk", "on", "god", "no", "cap"]
CONTROLS = ["honestly", "seriously", "in fact", "nearby", "outside", "in the", "with it", "at this"]
ALL_TERMS = MARKERS + COMPONENTS + CONTROLS

def term_re(t):
    return re.compile(r"\b" + r"\s+".join(map(re.escape, t.split())) + r"\b", re.I)

def dist_row(counter, total):
    return [counter.get(i, 0) for i in range(3)]

def chi_vs_overall(obs, overall):
    """Chi-square of obs (3 counts) vs the rest of the dataset."""
    rest = [o - x for o, x in zip(overall, obs)]
    if sum(obs) == 0:
        return float("nan"), float("nan"), float("nan")
    tab = np.array([obs, rest])
    chi2, p, dof, _ = chi2_contingency(tab, correction=False)
    g, pg, _, _ = chi2_contingency(tab, correction=False, lambda_="log-likelihood")
    return chi2, p, g

def analyze(name, ds, has_genre):
    hyp = ds["hypothesis"]; prem = ds["premise"]; lab = ds["label"]
    genre = ds["genre"] if has_genre else ["all"] * len(lab)
    n = len(lab)
    overall = Counter(lab)
    overall_row = dist_row(overall, n)
    out = []
    def add(field, term, mask_counter_by_genre):
        for g, c in sorted(mask_counter_by_genre.items()):
            obs = dist_row(c, sum(c.values()))
            tot = sum(obs)
            if g == "ALL" and term != "<overall>":
                chi2, p, G = chi_vs_overall(obs, overall_row)
            else:
                chi2 = p = G = float("nan")
            out.append(dict(dataset=name, field=field, term=term, genre=g, n=tot,
                            entailment=obs[0], neutral=obs[1], contradiction=obs[2],
                            p_ent=obs[0]/tot if tot else float("nan"),
                            p_neu=obs[1]/tot if tot else float("nan"),
                            p_con=obs[2]/tot if tot else float("nan"),
                            chi2_vs_rest=chi2, p_value=p, G_stat=G))
    # overall
    add("hypothesis", "<overall>", {"ALL": overall} if not has_genre else
        {"ALL": overall, **{g: Counter(l for l, gg in zip(lab, genre) if gg == g) for g in set(genre)}})
    for field, texts in (("hypothesis", hyp), ("premise", prem)):
        # god sub-patterns
        sub = {k: {} for k, _ in GOD_PATTERNS}
        anyc = {}
        for t, l, g in zip(texts, lab, genre):
            if not GOD_ANY.search(t):
                continue
            for key in ("ALL", g) if has_genre else ("ALL",):
                anyc.setdefault(key, Counter())[l] += 1
            for k, pat in GOD_PATTERNS:
                if pat.search(t):
                    for key in ("ALL", g) if has_genre else ("ALL",):
                        sub[k].setdefault(key, Counter())[l] += 1
                    break
        add(field, "god (any)", anyc)
        for k, _ in GOD_PATTERNS:
            add(field, k, sub[k])
        # other terms
        for term in ALL_TERMS:
            if term == "god":
                continue
            pat = term_re(term)
            c = {}
            for t, l, g in zip(texts, lab, genre):
                if pat.search(t):
                    for key in ("ALL", g) if has_genre else ("ALL",):
                        c.setdefault(key, Counter())[l] += 1
            if not c:
                c = {"ALL": Counter()}
            add(field, term, c)
    return out

rows = []
mnli = load_from_disk(str(REL / "multi_nli/training"))
rows += analyze("multi_nli", mnli, True)
snli = load_from_disk(str(REL / "snli/training"))
rows += analyze("snli", snli, False)
with open(OUT / "m1_training_counts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# example MultiNLI hypotheses with god, by label (for the report)
ex = []
for t, l, g in zip(mnli["hypothesis"], mnli["label"], mnli["genre"]):
    if GOD_ANY.search(t):
        ex.append(dict(genre=g, label=LABELS[l], hypothesis=t))
with open(OUT / "m1_mnli_god_hypotheses.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["genre", "label", "hypothesis"]); w.writeheader(); w.writerows(ex)
ex = []
for t, l in zip(snli["hypothesis"], snli["label"]):
    if GOD_ANY.search(t):
        ex.append(dict(label=LABELS[l], hypothesis=t))
with open(OUT / "m1_snli_god_hypotheses.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label", "hypothesis"]); w.writeheader(); w.writerows(ex)

# print summary
for r in rows:
    if r["genre"] == "ALL" and r["field"] == "hypothesis":
        print(f'{r["dataset"]:9s} {r["term"]:26s} n={r["n"]:6d}  E/N/C={r["entailment"]}/{r["neutral"]}/{r["contradiction"]}  pN={r["p_neu"]:.3f}  chi2={r["chi2_vs_rest"]:.1f} p={r["p_value"]:.2e}')
print("--- god (any) in hypothesis by genre, MultiNLI")
for r in rows:
    if r["dataset"] == "multi_nli" and r["field"] == "hypothesis" and r["term"] in ("god (any)", "<overall>") and r["genre"] != "ALL":
        print(f'{r["term"]:12s} {r["genre"]:12s} n={r["n"]:6d} E/N/C={r["entailment"]}/{r["neutral"]}/{r["contradiction"]} pN={r["p_neu"]:.3f}')
print("--- premise god (any)")
for r in rows:
    if r["genre"] == "ALL" and r["field"] == "premise" and r["term"].startswith("god"):
        print(r["dataset"], r["term"], r["n"], r["entailment"], r["neutral"], r["contradiction"], f'{r["p_neu"]:.3f}', f'{r["p_value"]:.2e}')
