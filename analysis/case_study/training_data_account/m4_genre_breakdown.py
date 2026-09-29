#!/usr/bin/env python3
"""M4: on-god failure on MultiNLI-matched clean-correct entailments by genre and premise-"god" for six encoders."""
import gzip, json, re, csv
from collections import defaultdict, Counter
from pathlib import Path
from datasets import load_from_disk
from scipy.stats import chi2_contingency

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402
OUT = paths.output("case_study", "training_data_account")
V1 = paths.predictions("crossed", "workspace", "predictions")
V2 = paths.predictions("crossed", "bertweet_corrected")
DEB = paths.predictions("crossed", "deberta")
SRC = paths.data("source", "multi_nli", "final")
GOD = re.compile(r"\bgod\b", re.I)
CONDS = ["c05_informal_on_god", "c13_component_god", "c12_component_on", "c09_informal_no_cap",
         "c19_random_nearby", "c17_formal_seriously", "c18_formal_in_fact"]

def find(d, prefix):
    m = sorted(p for p in d.glob(prefix + "*.jsonl.gz"))
    assert len(m) == 1, (d, prefix, m)
    return m[0]

def load(path, conds):
    preds = {c: {} for c in ["c00_clean"] + conds}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            c = r["condition_id"]
            if c in preds:
                preds[c][int(r["source_index"])] = (int(r["prediction"]), int(r["label"]))
    return preds

rows = []
summary = []
for split in ("validation_matched", "validation_mismatched"):
    src = load_from_disk(str(SRC / split))
    genre = src["genre"]; prem = src["premise"]; hyp = src["hypothesis"]
    models = {
        "electra": find(V1, f"electra__multi_nli__{split}__"),
        "roberta_base": find(V1, f"roberta_base__multi_nli__{split}__"),
        "roberta": find(V1, f"roberta__multi_nli__{split}__"),
        "timelm": find(V1, f"timelm__multi_nli__{split}__"),
        "bertweet": find(V2, f"bertweet__multi_nli__{split}__tokenizerfix__"),
        "deberta_v3_base": find(DEB, f"deberta_v3_base_controlled_mnli_seed42__multi_nli__{split}"),
    }
    allpreds = {m: load(p, CONDS) for m, p in models.items()}
    # sanity: label matches source
    for m, pr in allpreds.items():
        for i, (p, l) in list(pr["c00_clean"].items())[:200]:
            assert l == src[i]["label"], (m, i)
    five = ["electra", "roberta_base", "roberta", "timelm", "bertweet"]
    for cond in CONDS:
        for m in list(models) + ["all_five"]:
            by = defaultdict(lambda: [0, 0])   # key -> [n, fail]
            byp = defaultdict(lambda: [0, 0])
            n_src = len(genre)
            for i in range(n_src):
                if m == "all_five":
                    if not all(allpreds[k]["c00_clean"][i][0] == 0 == allpreds[k]["c00_clean"][i][1] for k in five):
                        continue
                    fail = all(allpreds[k][cond][i][0] != 0 for k in five)
                else:
                    p0, l = allpreds[m]["c00_clean"][i]
                    if not (l == 0 and p0 == 0):
                        continue
                    fail = allpreds[m][cond][i][0] != 0
                g = genre[i]
                by[g][0] += 1; by[g][1] += fail
                by["ALL"][0] += 1; by["ALL"][1] += fail
                key = "premise_has_god" if GOD.search(prem[i]) else "premise_no_god"
                byp[key][0] += 1; byp[key][1] += fail
            for g, (n, fl) in sorted(by.items()):
                rows.append(dict(split=split, condition=cond, model=m, breakdown="genre", key=g, n=n, fail=fl, fail_rate=fl / n if n else float("nan")))
            for k, (n, fl) in sorted(byp.items()):
                rows.append(dict(split=split, condition=cond, model=m, breakdown="premise_god", key=k, n=n, fail=fl, fail_rate=fl / n if n else float("nan")))
            # chi-square genre heterogeneity
            gs = [g for g in by if g != "ALL"]
            tab = [[by[g][1], by[g][0] - by[g][1]] for g in gs]
            try:
                chi2, p, _, _ = chi2_contingency(tab, correction=False)
            except Exception:
                chi2 = p = float("nan")
            summary.append(dict(split=split, condition=cond, model=m, genre_chi2=chi2, genre_p=p,
                                **{f"rate_{g}": by[g][1] / by[g][0] for g in gs},
                                rate_all=by["ALL"][1] / by["ALL"][0], n_all=by["ALL"][0]))
with open(OUT / "m4_genre_breakdown.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
keys = sorted({k for s in summary for k in s}, key=lambda k: (k not in ("split", "condition", "model"), k))
with open(OUT / "m4_genre_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
for s in summary:
    if s["condition"] in ("c05_informal_on_god", "c13_component_god", "c19_random_nearby", "c17_formal_seriously"):
        print(s["split"][11:], s["condition"][4:], f'{s["model"]:16s}', " ".join(f'{k[5:]}={v:.3f}' for k, v in s.items() if k.startswith("rate_")), f'chi2p={s["genre_p"]:.1e}')
print("--- premise god, on god")
for r in rows:
    if r["condition"] == "c05_informal_on_god" and r["breakdown"] == "premise_god":
        print(r["split"][11:], f'{r["model"]:16s}', r["key"], r["n"], r["fail"], f'{r["fail_rate"]:.3f}')
