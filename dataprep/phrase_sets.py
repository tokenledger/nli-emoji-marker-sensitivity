"""Build the crossed-study conditions and the content-word phrases.

Both sets append a phrase to the end of every hypothesis of the three final
splits. The crossed study has 24 conditions per source pair
(``CONDITION_REGISTRY.csv``); the content-word set has nine phrases.

The row format and the construction of each crossed row follow the builder
used for the paper (``build_eval.py`` of the crossed evaluation). Checks that
bound that builder to one machine's files are replaced by ``verify.py``, which
compares logical checksums with ``expected_checksums.json``.

The two sets follow the rules that produced the predictions of the paper, and
these rules differ for a hypothesis that ends in whitespace. The crossed
builder inserts no second space before the phrase. The content-word inference
appended a space and the phrase to the unchanged hypothesis, so such a
hypothesis is followed by two spaces. No SNLI test hypothesis ends in
whitespace; 1,291 MultiNLI matched and 1,348 mismatched hypotheses do.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from helpers import load_frozen_source_split
from transforms import (FORMAL_MARKER_CONTROLS, HELD_OUT_MARKERS,
                        RANDOM_PHRASE_CONTROLS, TRAIN_MARKERS,
                        apply_marker_with_metadata, invert_from_provenance)

MODULE_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = MODULE_ROOT / 'CONDITION_REGISTRY.csv'
CONTENT_WORD_CONDITIONS = (
    ('c24_formal_content_in_truth', 'in truth', 'formal_content_noun'),
    ('c25_formal_content_in_essence', 'in essence', 'formal_content_noun'),
    ('c26_formal_content_in_short', 'in short', 'formal_content_noun'),
    ('c27_formal_content_in_reality', 'in reality', 'formal_content_noun'),
    ('c28_religious_heaven_knows', 'heaven knows', 'religious_noun_commitment'),
    ('c29_religious_lord_knows', 'lord knows', 'religious_noun_commitment'),
    ('c30_religious_by_heaven', 'by heaven', 'religious_noun_commitment'),
    ('c31_religious_good_heavens', 'good heavens', 'religious_noun_commitment'),
    ('c32_informal_content_for_real', 'for real', 'informal_content_marker'),
)


def canonical_json(value: Mapping) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def suite_id(dataset: str, split: str) -> str:
    return f'{dataset}:final:{split}'


def artifact_name(dataset: str, split: str) -> str:
    return suite_id(dataset, split).replace(':', '__') + '.jsonl.gz'


def load_registry() -> list[dict]:
    with REGISTRY_PATH.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise ValueError(f'Condition registry must have exactly 24 rows, found {len(rows)}')
    expected_ids = [f'c{index:02d}_' for index in range(24)]
    if any(not row['condition_id'].startswith(expected_ids[index]) for index, row in enumerate(rows)):
        raise ValueError('Condition registry indices/IDs are not in frozen order')
    if len({row['condition_id'] for row in rows}) != 24:
        raise ValueError('Condition IDs are not unique')
    for index, row in enumerate(rows):
        row['condition_index'] = int(row['condition_index'])
        row['inserted_token_count'] = int(row['inserted_token_count'])
        row['synthetic_lexical_ablation'] = row['synthetic_lexical_ablation'] == 'true'
        row['new_manual_validation'] = row['new_manual_validation'] == 'true'
        if row['condition_index'] != index:
            raise ValueError('Condition indices must be contiguous from zero')
        expected_tokens = len(row['inserted_text'].split()) if row['inserted_text'] else 0
        if row['inserted_token_count'] != expected_tokens:
            raise ValueError(f"Token count mismatch for {row['condition_id']}")
        if row['new_manual_validation']:
            raise ValueError('No new condition may claim manual validation')
    by_type = Counter(row['condition_type'] for row in rows)
    if by_type != Counter({
        'clean': 1,
        'informal_single_token': 6,
        'informal_full_phrase': 3,
        'component_ablation': 6,
        'formal_control': 3,
        'random_control': 5,
    }):
        raise ValueError(f'Unexpected condition-family counts: {dict(by_type)}')
    expected_informal = list(TRAIN_MARKERS) + list(HELD_OUT_MARKERS)
    observed_informal = [row['inserted_text'] for row in rows if row['condition_type'].startswith('informal_')]
    if observed_informal != expected_informal:
        raise ValueError('Informal conditions do not exactly match the frozen config lexicons')
    if [r['inserted_text'] for r in rows if r['condition_type'] == 'formal_control'] != list(FORMAL_MARKER_CONTROLS):
        raise ValueError('Formal controls do not exactly match the frozen config lexicon')
    if [r['inserted_text'] for r in rows if r['condition_type'] == 'random_control'] != list(RANDOM_PHRASE_CONTROLS):
        raise ValueError('Random controls do not exactly match the frozen config lexicon')
    return rows


def content_word_registry() -> list[dict]:
    return [
        {
            'condition_index': 24 + position,
            'condition_id': condition_id,
            'condition_type': group,
            'inserted_text': phrase,
            'inserted_token_count': len(phrase.split()),
            'edit_side': 'hypothesis',
            'placement': 'suffix',
            'synthetic_lexical_ablation': False,
            'new_manual_validation': False,
        }
        for position, (condition_id, phrase, group) in enumerate(CONTENT_WORD_CONDITIONS)
    ]


def load_source(data_dir: Path, dataset: str, split: str) -> list[dict]:
    # Access only the final split's text and production label columns.
    source = load_frozen_source_split(data_dir, dataset, 'final', split)
    source = source.select_columns(['premise', 'hypothesis', 'label'])
    rows = []
    for source_index in range(len(source)):
        item = source[source_index]
        row = {
            'source_index': source_index,
            'premise': item['premise'],
            'hypothesis': item['hypothesis'],
            'label': int(item['label']),
        }
        if not isinstance(row['premise'], str) or not row['premise']:
            raise ValueError(f'Empty/non-string premise at {dataset}/{split}:{source_index}')
        if not isinstance(row['hypothesis'], str) or not row['hypothesis']:
            raise ValueError(f'Empty/non-string hypothesis at {dataset}/{split}:{source_index}')
        if row['label'] not in {0, 1, 2}:
            raise ValueError(f'Invalid label at {dataset}/{split}:{source_index}')
        rows.append(row)
    return rows


def transformed_row(source: Mapping, dataset: str, split: str, condition: Mapping,
                    registry_name: str | None) -> dict:
    text = source['hypothesis']
    event = None
    if condition['condition_type'] == 'clean':
        hypothesis = text
    elif registry_name is None:
        hypothesis = f"{text} {condition['inserted_text']}"
    else:
        hypothesis, events = apply_marker_with_metadata(
            text,
            prob=1.0,
            marker=condition['inserted_text'],
            marker_choices=(condition['inserted_text'],),
            registry_name=registry_name,
            placement='suffix',
            seed=0,
        )
        if len(events) != 1:
            raise ValueError('Deterministic suffix transformation did not emit exactly one event')
        event = events[0].to_dict()
        if invert_from_provenance(hypothesis, [event]) != text:
            raise ValueError('Suffix provenance failed exact inverse reconstruction')
        inserted = event['replacement']
        if inserted.lstrip() != condition['inserted_text'] or inserted.strip() != condition['inserted_text']:
            raise ValueError('Recorded event does not contain the exact frozen inserted text')
        if event['placement'] != 'suffix' or event['source'] != '':
            raise ValueError('Transformation is not a pure suffix insertion')
    return {
        'schema_version': 1,
        'suite_id': suite_id(dataset, split),
        'dataset': dataset,
        'split': split,
        'source_index': source['source_index'],
        'condition_index': condition['condition_index'],
        'condition_id': condition['condition_id'],
        'condition_type': condition['condition_type'],
        'inserted_text': condition['inserted_text'],
        'inserted_token_count': condition['inserted_token_count'],
        'edit_side': condition['edit_side'],
        'placement': condition['placement'],
        'synthetic_lexical_ablation': condition['synthetic_lexical_ablation'],
        'new_manual_validation': False,
        'premise': source['premise'],
        'hypothesis': hypothesis,
        'label': source['label'],
        'transformation_event': event,
    }


def write_suite(output: Path, source_rows: list[dict], dataset: str, split: str,
                conditions: list[dict], registry_name: str | None) -> dict:
    matrix_digest = hashlib.sha256()
    condition_digests = {row['condition_id']: hashlib.sha256() for row in conditions}
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='\n') as handle:
                for condition in conditions:
                    for source in source_rows:
                        row = transformed_row(source, dataset, split, condition, registry_name)
                        encoded = (canonical_json(row) + '\n')
                        handle.write(encoded)
                        matrix_digest.update(encoded.encode('utf-8'))
                        condition_digests[condition['condition_id']].update(
                            encoded.encode('utf-8')
                        )
    temporary.replace(output)
    return {
        'suite_id': suite_id(dataset, split),
        'dataset': dataset,
        'split': split,
        'artifact': output.name,
        'source_examples': len(source_rows),
        'condition_count': len(conditions),
        'matrix_rows': len(source_rows) * len(conditions),
        'logical_checksum_sha256': matrix_digest.hexdigest(),
        'condition_logical_checksums': {
            key: digest.hexdigest() for key, digest in condition_digests.items()
        },
    }


def build(data_dir: Path, suites: list[tuple[str, str]], *, directory: str,
          conditions: list[dict], registry_name: str | None) -> None:
    if not suites:
        raise SystemExit('No final split matches the requested DATASETS and SPLITS')
    target = data_dir / directory
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / 'manifest.json'
    manifest = (
        json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest_path.is_file() else {'schema_version': 1, 'suites': {}}
    )
    manifest['condition_ids'] = [row['condition_id'] for row in conditions]
    manifest['placement'] = 'hypothesis_suffix'
    for dataset, split in suites:
        source_rows = load_source(data_dir, dataset, split)
        record = write_suite(
            target / artifact_name(dataset, split), source_rows, dataset, split,
            conditions, registry_name,
        )
        manifest['suites'][record['suite_id']] = record
        print(f"[{directory}] {record['suite_id']}: {record['matrix_rows']:,} rows")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )


def build_crossed(data_dir: Path, suites: list[tuple[str, str]]) -> None:
    build(data_dir, suites, directory='crossed', conditions=load_registry(),
          registry_name='phrase_component_counterfactuals_v1')


def build_content_words(data_dir: Path, suites: list[tuple[str, str]]) -> None:
    build(data_dir, suites, directory='content_words',
          conditions=content_word_registry(), registry_name=None)
