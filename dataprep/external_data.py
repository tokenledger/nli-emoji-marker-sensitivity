"""Fetch the external datasets of the case study: ChaosNLI, SICK, and ANLI.

SICK and ANLI are frozen as JSON lines with labels mapped to the convention
0 = entailment, 1 = neutral, 2 = contradiction. The SICK and ANLI part follows
``download_data.py`` of the transfer experiment. ChaosNLI is read from the
Hugging Face mirror ``tasksource/chaos-mnli-ambiguity``.
"""

from __future__ import annotations

import collections
import hashlib
import json
import shutil
from pathlib import Path

FROZEN = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
CHAOSNLI_REPOSITORY = 'tasksource/chaos-mnli-ambiguity'
CHAOSNLI_FILE = 'chaos_mnli.jsonl'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(out: Path) -> None:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    notes = ['# External datasets', '']

    mirror = hf_hub_download(CHAOSNLI_REPOSITORY, CHAOSNLI_FILE, repo_type='dataset')
    chaos = out / 'chaosnli_mnli.jsonl'
    shutil.copyfile(mirror, chaos)
    rows = [json.loads(line) for line in chaos.open(encoding='utf-8')]
    notes.append(
        f'## ChaosNLI (MultiNLI part)\n\nHugging Face `{CHAOSNLI_REPOSITORY}` '
        f'(`{CHAOSNLI_FILE}`), {len(rows)} rows. File sha256 `{sha(chaos)}`.\n'
    )

    # The canonical `sick` repo is a script dataset (unsupported by datasets 5.x); use the Hub's
    # auto-converted parquet of the same repo (revision refs/convert/parquet, default/test).
    parquet = hf_hub_download('sick', 'default/test/0000.parquet', repo_type='dataset', revision='refs/convert/parquet')
    sick = load_dataset('parquet', data_files={'test': parquet}, split='test')
    names = sick.features['label'].names if hasattr(sick.features['label'], 'names') else ['entailment', 'neutral', 'contradiction']
    notes.append(f"## SICK test\n\nHugging Face `sick` (parquet conversion `refs/convert/parquet`, `default/test/0000.parquet`, sha256 `{sha(Path(parquet))}`), {len(sick)} rows. Label feature: {sick.features['label']}; names used: {names}.")
    name_to_frozen = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    mapping = {i: name_to_frozen[n] for i, n in enumerate(names)}
    notes.append(f'Mapping HF id -> frozen id: {mapping} (frozen: {FROZEN}).')
    rows = []
    for i, ex in enumerate(sick):
        rows.append({'dataset': 'sick', 'split': 'test', 'source_index': i, 'orig_id': str(ex.get('id', i)),
                     'premise': ex['sentence_A'], 'hypothesis': ex['sentence_B'],
                     'label': mapping[int(ex['label'])], 'hf_label': int(ex['label']), 'round': 'test'})
    p = out / 'sick_test.jsonl'
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    dist = collections.Counter(r['label'] for r in rows)
    notes.append(f'Label distribution (frozen ids): {dict(sorted(dist.items()))}. File sha256 `{sha(p)}`.\n')

    rows = []
    per_round = {}
    for r in ('dev_r1', 'dev_r2', 'dev_r3'):
        ds = load_dataset('facebook/anli', split=r)
        names = ds.features['label'].names
        mapping = {i: name_to_frozen[n] for i, n in enumerate(names)}
        per_round[r] = (len(ds), names, mapping)
        for ex in ds:
            rows.append({'dataset': 'anli', 'split': 'dev', 'source_index': len(rows), 'orig_id': ex['uid'],
                         'premise': ex['premise'], 'hypothesis': ex['hypothesis'],
                         'label': mapping[int(ex['label'])], 'hf_label': int(ex['label']), 'round': r})
    p = out / 'anli_dev.jsonl'
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    dist = collections.Counter(r['label'] for r in rows)
    notes.append(f'## ANLI dev R1+R2+R3\n\nHugging Face `facebook/anli`, splits dev_r1/dev_r2/dev_r3, {len(rows)} rows pooled.')
    for r, (n, names, mapping) in per_round.items():
        notes.append(f'- {r}: {n} rows, HF label names {names}, mapping {mapping}')
    notes.append(f'\nLabel distribution (frozen ids): {dict(sorted(dist.items()))}. File sha256 `{sha(p)}`.\n')
    (out / 'DATA.md').write_text('\n'.join(notes) + '\n')
    print('\n'.join(notes))
