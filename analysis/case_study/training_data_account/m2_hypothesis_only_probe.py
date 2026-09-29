#!/usr/bin/env python3
"""M2: hypothesis-only TF-IDF (1-2 gram) logistic regression on SNLI and MultiNLI training;
predicted-label shift on the crossed evaluation's clean-correct entailments under each suffix."""
import gzip, json, csv, re
from collections import Counter, defaultdict
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
REG = Path(__file__).resolve().parents[3] / "dataprep" / "CONDITION_REGISTRY.csv"
SEED = 20260924
conds = list(csv.DictReader(open(REG)))
LAB = ["entailment", "neutral", "contradiction"]

def fit(name, train, dev):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True, sublinear_tf=True,
                          token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(train["hypothesis"]); y = np.array(train["label"])
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    clf.fit(X, y)
    acc_train = clf.score(X, y)
    acc_dev = clf.score(vec.transform(dev["hypothesis"]), np.array(dev["label"]))
    print(f"{name}: hypothesis-only train acc {acc_train:.4f}, dev acc {acc_dev:.4f}, n_train={X.shape[0]}, n_feat={X.shape[1]}")
    return vec, clf, acc_dev

def coef_rows(name, vec, clf, terms):
    vocab = vec.vocabulary_; out = []
    for t in terms:
        i = vocab.get(t)
        if i is None:
            out.append(dict(dataset=name, feature=t, in_vocab=False, coef_ent=None, coef_neu=None, coef_con=None, idf=None)); continue
        out.append(dict(dataset=name, feature=t, in_vocab=True, coef_ent=clf.coef_[0, i], coef_neu=clf.coef_[1, i], coef_con=clf.coef_[2, i], idf=vec.idf_[i]))
    return out

def read_eval(path):
    by = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            by[r["condition_id"]][r["source_index"]] = (r["hypothesis"], r["label"])
    return by

rows, coefs, accs = [], [], []
for name, tr_path, dev_path, evals in (
    ("multi_nli", REL / "multi_nli/training", REL / "multi_nli/development/train_holdout",
     [("validation_matched", EVAL / "multi_nli__final__validation_matched.jsonl.gz"),
      ("validation_mismatched", EVAL / "multi_nli__final__validation_mismatched.jsonl.gz")]),
    ("snli", REL / "snli/training", REL / "snli/development/validation",
     [("test", EVAL / "snli__final__test.jsonl.gz")]),
):
    train = load_from_disk(str(tr_path))
    try:
        dev = load_from_disk(str(dev_path))
    except Exception:
        dev = None
    if dev is None:
        idx = np.random.RandomState(SEED).permutation(len(train)); dev = train.select(idx[:10000]); train = train.select(idx[10000:])
        print(f"{name}: no development split found, holding out 10000 rows")
    vec, clf, acc_dev = fit(name, train, dev)
    accs.append(dict(dataset=name, dev_accuracy=acc_dev, n_train=len(train), n_dev=len(dev)))
    coefs += coef_rows(name, vec, clf, ["god", "on god", "my god", "oh my", "on", "no cap", "cap", "no", "nearby", "seriously",
                                        "honestly", "in fact", "outside", "deadass", "real talk", "in the", "with it", "at this", "fact"])
    for split, ep in evals:
        by = read_eval(ep)
        clean = by["c00_clean"]
        idx = sorted(clean)
        H = [clean[i][0] for i in idx]; y = np.array([clean[i][1] for i in idx])
        pc = clf.predict(vec.transform(H))
        clean_acc = (pc == y).mean()
        ent_ok = [i for i, p, l in zip(idx, pc, y) if l == 0 and p == 0]
        print(f"{name}/{split}: probe clean acc {clean_acc:.4f}; clean-correct entailments {len(ent_ok)}/{(y==0).sum()}")
        for c in conds:
            cid = c["condition_id"]
            if cid == "c00_clean":
                continue
            Hc = [by[cid][i][0] for i in ent_ok]
            pcnd = clf.predict(vec.transform(Hc))
            cnt = Counter(pcnd)
            # also on all sources: accuracy change
            Hall = [by[cid][i][0] for i in idx]
            acc_c = (clf.predict(vec.transform(Hall)) == y).mean()
            rows.append(dict(dataset=name, split=split, condition=cid, inserted_text=c["inserted_text"],
                             condition_type=c["condition_type"], n_clean_correct_ent=len(ent_ok),
                             to_neutral=cnt.get(1, 0), to_contradiction=cnt.get(2, 0),
                             frac_to_neutral=cnt.get(1, 0) / len(ent_ok), frac_fail=(cnt.get(1, 0) + cnt.get(2, 0)) / len(ent_ok),
                             probe_clean_acc=clean_acc, probe_cond_acc=acc_c, acc_change_pp=100 * (acc_c - clean_acc)))
with open(OUT / "m2_probe_shifts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(OUT / "m2_probe_coefficients.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(coefs[0].keys())); w.writeheader(); w.writerows(coefs)
with open(OUT / "m2_probe_accuracy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(accs[0].keys())); w.writeheader(); w.writerows(accs)
for r in rows:
    print(f'{r["dataset"]:9s} {r["split"]:22s} {r["inserted_text"]:10s} toNeu={r["frac_to_neutral"]:.3f} fail={r["frac_fail"]:.3f} dAcc={r["acc_change_pp"]:+.2f}')
for c in coefs:
    print(c)
