#!/usr/bin/env python3
"""M4b: the clean-correct MultiNLI entailments whose premise contains "god": source ids, genre,
distinct premises, and each encoder's outcome under on god and control suffixes."""
import gzip, json, re, csv
from pathlib import Path
from datasets import load_from_disk
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402
OUT = paths.output("case_study", "training_data_account")
V1 = paths.predictions("crossed", "workspace", "predictions")
V2 = paths.predictions("crossed", "bertweet_corrected")
DEB = paths.predictions("crossed", "deberta")
SRC = paths.data("source", "multi_nli", "final")
GOD = re.compile(r"\bgod\b", re.I)
CONDS = {"on god": "c05_informal_on_god", "god": "c13_component_god", "nearby": "c19_random_nearby",
         "seriously": "c17_formal_seriously", "in fact": "c18_formal_in_fact", "no cap": "c09_informal_no_cap", "outside": "c20_random_outside"}
def find(d, prefix):
    m = sorted(d.glob(prefix + "*.jsonl.gz")); assert len(m) == 1; return m[0]
rows = []
for split in ("validation_matched", "validation_mismatched"):
    src = load_from_disk(str(SRC / split))
    cand = {i for i, (p, l) in enumerate(zip(src["premise"], src["label"])) if l == 0 and GOD.search(p)}
    models = {"electra": find(V1, f"electra__multi_nli__{split}__"), "roberta_base": find(V1, f"roberta_base__multi_nli__{split}__"),
              "roberta": find(V1, f"roberta__multi_nli__{split}__"), "timelm": find(V1, f"timelm__multi_nli__{split}__"),
              "bertweet": find(V2, f"bertweet__multi_nli__{split}__tokenizerfix__"),
              "deberta_v3_base": find(DEB, f"deberta_v3_base_controlled_mnli_seed42__multi_nli__{split}")}
    want = {"c00_clean", *CONDS.values()}
    pred = {m: {} for m in models}
    for m, p in models.items():
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["condition_id"] in want and int(r["source_index"]) in cand:
                    pred[m][(r["condition_id"], int(r["source_index"]))] = int(r["prediction"])
    for i in sorted(cand):
        for m in models:
            if pred[m][("c00_clean", i)] != 0:
                continue
            rows.append(dict(split=split, source_index=i, pairID=src[i]["pairID"], genre=src[i]["genre"], model=m,
                             premise=src[i]["premise"], hypothesis=src[i]["hypothesis"],
                             **{f"pred_{k}": pred[m][(c, i)] for k, c in CONDS.items()}))
with open(OUT / "m4b_premise_god_sources.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import pandas as pd
d = pd.DataFrame(rows)
for split, g in d.groupby("split"):
    ids = sorted(g.source_index.unique()); print(split, "sources:", ids, "distinct premises:", g.drop_duplicates("source_index").premise.nunique(),
          "genres:", g.drop_duplicates("source_index").genre.value_counts().to_dict())
    for m, gm in g.groupby("model"):
        print(" ", f"{m:16s}", "n=", len(gm), " ".join(f"{k}:fail={(gm[f'pred_{k}']!=0).sum()}" for k in CONDS))
