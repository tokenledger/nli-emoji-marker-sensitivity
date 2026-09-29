#!/usr/bin/env python3
"""M3: how the six tokenizers segment the suffixes at hypothesis end."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402
OUT = paths.output("case_study", "training_data_account")
from transformers import AutoTokenizer
import training_runtime
DEBERTA_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
SPECS = [
    ("electra", "google/electra-small-discriminator", dict(use_fast=True)),
    ("roberta_base", "roberta-base", dict(use_fast=True)),
    ("roberta", "roberta-large", dict(use_fast=True)),
    ("timelm", "cardiffnlp/twitter-roberta-base-2021-124m", dict(use_fast=True)),
    ("bertweet", "vinai/bertweet-base", dict(use_fast=False, normalization=False)),
    ("deberta_v3_base", "microsoft/deberta-v3-base", None),
]
BASE = "Animal is outdoors."
SUFFIXES = ["fr", "tbh", "ngl", "real talk", "on god", "istg", "frfr", "deadass", "no cap", "real", "talk", "on", "god", "no", "cap", "honestly", "seriously", "in fact", "nearby", "outside", "in the", "with it", "at this", "God"]
rows = []
for name, repo, kw in SPECS:
    if kw is None:
        tok = training_runtime.load_tokenizer(repo, DEBERTA_REVISION, backend="fast", normalization="spm_byte_fallback")
    else:
        tok = AutoTokenizer.from_pretrained(repo, **kw)
    base_ids = tok(BASE, add_special_tokens=False)["input_ids"]
    for suf in SUFFIXES:
        text = BASE + " " + suf
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert ids[:len(base_ids)] == base_ids, (name, suf, ids, base_ids)
        suf_ids = ids[len(base_ids):]
        pieces = tok.convert_ids_to_tokens(suf_ids)
        # is "god" a single whole-word piece?
        god_single = None
        if "god" in suf.lower().split():
            wid = tok(" god" if name != "bertweet" else "god", add_special_tokens=False)["input_ids"]
            god_single = (len(wid) == 1)
        rows.append(dict(model=name, tokenizer=repo, suffix=suf, n_pieces=len(suf_ids),
                         piece_ids=" ".join(map(str, suf_ids)), pieces=" ".join(pieces),
                         god_single_piece=god_single))
        print(f"{name:16s} {suf!r:12s} n={len(suf_ids)} pieces={pieces} ids={suf_ids} god_single={god_single}")
with open(OUT / "m3_tokenization.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
