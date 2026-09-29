"""Checks on the generated data.

Every check reads the generated rows; none trusts a stored audit. A check whose
input was not generated (for example after a limited ``make modify``) is
reported as SKIP and does not count as a failure.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

from helpers import (load_frozen_source_split, logical_dataset_checksum)
from transforms import invert_from_provenance

import phrase_sets

MODULE_ROOT = Path(__file__).resolve().parent
EXPECTED_PATH = MODULE_ROOT / 'expected_checksums.json'
ELIGIBLE_EMOJI_PAIRS = {
    ('snli', 'test'): 6430,
    ('multi_nli', 'validation_matched'): 522,
    ('multi_nli', 'validation_mismatched'): 574,
}
FINAL_SUITES = tuple(ELIGIBLE_EMOJI_PAIRS)


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def record(self, name: str, passed: bool | None, detail: str) -> None:
        status = 'SKIP' if passed is None else ('PASS' if passed else 'FAIL')
        if passed is False:
            self.failed += 1
        print(f'[{status}] {name}: {detail}')


def load_variant(data_dir: Path, manifest: dict, dataset: str, split: str, variant: str):
    from datasets import load_from_disk

    entry = manifest['datasets'].get(dataset, {}).get('splits', {}).get(f'final:{split}')
    if entry is None or variant not in entry['variants']:
        return None
    return load_from_disk(str(data_dir / 'eval_sets' / entry['variants'][variant]['path']))


def check_source(data_dir: Path, expected: dict, report: Report) -> None:
    manifest = json.loads((data_dir / 'source_manifest.json').read_text(encoding='utf-8'))
    for dataset, reference in expected['source'].items():
        entry = manifest['datasets'][dataset]
        observed = {
            'training': entry['training'],
            'development': entry['development'],
            **{f'final/{name}': value for name, value in entry['final'].items()},
        }
        wanted = {
            'training': reference['training'],
            'development': reference['development'],
            **{f'final/{name}': value for name, value in reference['final'].items()},
        }
        for name, value in wanted.items():
            same = (
                observed[name]['examples'] == value['examples']
                and observed[name]['checksum_sha256'] == value['checksum_sha256']
            )
            report.record(
                f'source {dataset}/{name}', same,
                f"{observed[name]['examples']:,} pairs, checksum "
                f"{'matches' if same else 'differs from'} the paper's data",
            )
    for dataset, split in FINAL_SUITES:
        rows = load_frozen_source_split(data_dir, dataset, 'final', split)
        report.record(
            f'source rows {dataset}/{split}', True,
            f'{len(rows):,} rows reread; stored checksum reproduced',
        )


def check_eval_checksums(data_dir: Path, manifest: dict, expected: dict,
                         report: Report) -> None:
    from datasets import load_from_disk

    for dataset, entry in manifest['datasets'].items():
        for split_key, split in entry['splits'].items():
            reference = expected['eval_sets'][f'{dataset}/{split_key}']['variants']
            generated = {
                name: value for name, value in split['variants'].items()
                if 'alias_of' not in value
            }
            mismatched = []
            for name, value in generated.items():
                rows = load_from_disk(str(data_dir / 'eval_sets' / value['path']))
                if (
                    len(rows) != reference[name]['examples']
                    or logical_dataset_checksum(rows) != reference[name]['checksum_sha256']
                ):
                    mismatched.append(name)
            report.record(
                f'evaluation sets {dataset}/{split_key}', not mismatched,
                f'{len(generated)} of {len(reference)} conditions generated; '
                + (f'differ from the paper\'s data: {mismatched}' if mismatched
                   else 'all reproduce the paper\'s checksums'),
            )


def check_emoji(data_dir: Path, manifest: dict, report: Report) -> None:
    for (dataset, split), eligible in ELIGIBLE_EMOJI_PAIRS.items():
        rows = load_variant(data_dir, manifest, dataset, split, 'emoji_raw')
        if rows is None:
            report.record(f'emoji {dataset}/{split}', None, 'emoji_raw was not generated')
            continue
        report.record(
            f'eligible emoji pairs {dataset}/{split}', len(rows) == eligible,
            f'{len(rows):,} pairs, expected {eligible:,}',
        )
        source = load_frozen_source_split(data_dir, dataset, 'final', split)
        failures = 0
        replacements = Counter()
        for row in rows:
            clean = source[int(row['source_index'])]
            events = row['transform_metadata']['events']
            count = 0
            for field in ('premise', 'hypothesis'):
                field_events = events[field] or []
                count += len(field_events)
                if invert_from_provenance(row[field], field_events) != clean[field]:
                    failures += 1
            replacements[count] += 1
            if row['label'] != clean['label']:
                failures += 1
        report.record(
            f'exact inversion of emoji substitution {dataset}/{split}', failures == 0,
            f'{failures} of {len(rows):,} pairs fail to reproduce the paired clean text',
        )
        report.record(
            f'at most two replacements {dataset}/{split}',
            set(replacements) <= {1, 2},
            f'replacements per pair: {dict(sorted(replacements.items()))}',
        )


def check_markers(data_dir: Path, manifest: dict, config: dict, report: Report) -> None:
    for fold in config['transformations']['marker_folds']:
        overlap = set(fold['train']) & set(fold['test'])
        report.record(
            f"marker pools {fold['id']}", not overlap,
            f"training {fold['train']}, held out {fold['test']}, overlap {sorted(overlap)}",
        )
        for dataset, split in FINAL_SUITES:
            name = f"marker_unseen_{fold['id']}_hypothesis_suffix"
            rows = load_variant(data_dir, manifest, dataset, split, name)
            if rows is None:
                report.record(f'{name} {dataset}/{split}', None, 'not generated')
                continue
            counts: Counter = Counter()
            for row in rows:
                counts[(row['transform_metadata']['marker'], int(row['label']))] += 1
            markers = {marker for marker, _ in counts}
            leaked = markers & set(fold['train'])
            report.record(
                f'no held-out marker in the training pool, {name} {dataset}/{split}',
                not leaked and markers == set(fold['test']),
                f'markers used {sorted(markers)}, shared with training pool {sorted(leaked)}',
            )
            spread = max(
                max(counts[(marker, label)] for marker in markers)
                - min(counts[(marker, label)] for marker in markers)
                for label in (0, 1, 2)
            )
            report.record(
                f'label balance of marker assignment, {name} {dataset}/{split}',
                spread <= 1,
                f'largest difference between markers within a label: {spread} pair(s)',
            )


def read_rows(path: Path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)


def check_crossed(data_dir: Path, expected: dict, report: Report) -> None:
    import hashlib

    for dataset, split in FINAL_SUITES:
        suite = phrase_sets.suite_id(dataset, split)
        path = data_dir / 'crossed' / phrase_sets.artifact_name(dataset, split)
        if not path.is_file():
            report.record(f'crossed study {suite}', None, 'not generated')
            continue
        digest = hashlib.sha256()
        conditions: Counter = Counter()
        for row in read_rows(path):
            digest.update((phrase_sets.canonical_json(row) + '\n').encode('utf-8'))
            conditions[row['condition_id']] += 1
        reference = expected['crossed'][suite]
        complete = (
            len(conditions) == 24
            and set(conditions.values()) == {reference['source_examples']}
        )
        report.record(
            f'crossed study {suite}: 24 conditions on every source pair', complete,
            f'{len(conditions)} conditions, {sum(conditions.values()):,} rows',
        )
        report.record(
            f'crossed study {suite}: checksum',
            digest.hexdigest() == reference['logical_checksum_sha256'],
            'rows reproduce the evaluation data of the paper'
            if digest.hexdigest() == reference['logical_checksum_sha256']
            else 'rows differ from the evaluation data of the paper',
        )


def check_content_words(data_dir: Path, report: Report) -> None:
    phrases = {cid: phrase for cid, phrase, _ in phrase_sets.CONTENT_WORD_CONDITIONS}
    for dataset, split in FINAL_SUITES:
        suite = phrase_sets.suite_id(dataset, split)
        path = data_dir / 'content_words' / phrase_sets.artifact_name(dataset, split)
        if not path.is_file():
            report.record(f'content-word phrases {suite}', None, 'not generated')
            continue
        source = load_frozen_source_split(data_dir, dataset, 'final', split)
        conditions: Counter = Counter()
        wrong = 0
        for row in read_rows(path):
            clean = source[int(row['source_index'])]
            conditions[row['condition_id']] += 1
            expected_text = f"{clean['hypothesis']} {phrases[row['condition_id']]}"
            if (
                row['hypothesis'] != expected_text
                or row['premise'] != clean['premise']
                or row['label'] != clean['label']
            ):
                wrong += 1
        report.record(
            f'content-word phrases {suite}',
            wrong == 0 and set(conditions) == set(phrases)
            and set(conditions.values()) == {len(source)},
            f'{len(conditions)} phrases, {sum(conditions.values()):,} rows, '
            f'{wrong} rows differ from "hypothesis + space + phrase"',
        )


def run(data_dir: Path, config: dict) -> int:
    expected = json.loads(EXPECTED_PATH.read_text(encoding='utf-8'))
    report = Report()
    check_source(data_dir, expected, report)
    manifest_path = data_dir / 'eval_sets' / 'dataset_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        check_eval_checksums(data_dir, manifest, expected, report)
        check_emoji(data_dir, manifest, report)
        check_markers(data_dir, manifest, config, report)
    else:
        report.record('fold-study evaluation sets', None, 'eval_sets/ was not generated')
    check_crossed(data_dir, expected, report)
    check_content_words(data_dir, report)
    print(f'\n{report.failed} check(s) failed')
    return report.failed
