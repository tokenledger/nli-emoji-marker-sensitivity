"""Regression tests for frozen source rows and compact paired-clean views."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import datasets


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from helpers import (load_eval_suite, load_frozen_development_data,  # noqa: E402
                     load_frozen_training_data,
                     logical_dataset_checksum,
                     paired_clean_result_from_original,
                     source_dataset_identity,
                     verify_frozen_evaluation_data)
from prepare_eval_sets import paired_clean_records  # noqa: E402
from prepare_eval_sets import (  # noqa: E402
    _run_condition_tasks,
    distribute_worker_budget as distribute_eval_workers,
    resolve_worker_budget as resolve_eval_worker_budget,
)
from prepare_data_release import (  # noqa: E402
    apply_source_exclusions,
    distribute_worker_budget as distribute_source_workers,
    resolve_worker_budget as resolve_source_worker_budget,
    select_release_splits,
)
from experiment_config import load_experiment_config  # noqa: E402


ROWS = [
    {'premise': 'p0', 'hypothesis': 'h0', 'label': 0},
    {'premise': 'p1', 'hypothesis': 'h1', 'label': 1},
    {'premise': 'p2', 'hypothesis': 'h2', 'label': 2},
]


class DataReleaseTests(unittest.TestCase):
    def test_configured_source_exclusion_removes_only_declared_rows(self):
        prefix = 'Al and Tipper Gore have helped by saying'
        rows = datasets.Dataset.from_list([
            {'premise': prefix + ' one', 'hypothesis': 'h0', 'label': 0},
            {'premise': 'Unrelated premise', 'hypothesis': 'h1', 'label': 1},
            {'premise': prefix + ' two', 'hypothesis': 'h2', 'label': 2},
        ])
        final = datasets.Dataset.from_list([
            {'premise': prefix + ' three', 'hypothesis': 'h3', 'label': 1},
            {'premise': 'Another premise', 'hypothesis': 'h4', 'label': 0},
        ])
        rule = {
            'id': 'fixture', 'field': 'premise', 'match': 'prefix',
            'value': prefix, 'expected_total_matches': 3, 'reason': 'fixture',
        }
        filtered, audit = apply_source_exclusions(
            datasets.DatasetDict({'train': rows, 'validation': final}), [rule]
        )
        self.assertEqual(len(filtered['train']), 1)
        self.assertEqual(len(filtered['validation']), 1)
        self.assertEqual(audit['excluded_examples_total'], 3)
        self.assertEqual(
            audit['excluded_examples_by_split'], {'train': 2, 'validation': 1}
        )
        self.assertFalse(any(
            row['premise'].startswith(prefix)
            for split in filtered.values() for row in split
        ))

    def test_source_exclusion_aborts_when_expected_count_drifts(self):
        data = datasets.DatasetDict({'train': datasets.Dataset.from_list(ROWS)})
        rule = {
            'id': 'fixture', 'field': 'premise', 'match': 'prefix',
            'value': 'p', 'expected_total_matches': 2, 'reason': 'fixture',
        }
        with self.assertRaisesRegex(ValueError, 'expected 2 matches, found 3'):
            apply_source_exclusions(data, [rule])

    def test_post_selection_exclusion_preserves_all_other_split_membership(self):
        prefix = 'Al and Tipper Gore have helped by saying'
        base_rows = [
            {
                'row_id': index,
                'premise': f'premise {index}',
                'hypothesis': f'hypothesis {index}',
                'label': index % 3,
            }
            for index in range(90)
        ]
        label_feature = datasets.ClassLabel(num_classes=3)
        finals = datasets.Dataset.from_list([
            {'row_id': 1000, 'premise': 'final', 'hypothesis': 'final', 'label': 0}
        ]).cast_column('label', label_feature)
        base_training = datasets.Dataset.from_list(base_rows).cast_column(
            'label', label_feature
        )
        unmodified = datasets.DatasetDict({
            'train': base_training,
            'validation_matched': finals,
            'validation_mismatched': finals,
        })
        baseline_train, baseline_development = select_release_splits(
            unmodified,
            'multi_nli',
            {
                'train_split': 'train',
                'development_fraction': 0.05,
                'development_seed': 42,
                'final_splits': ['validation_matched', 'validation_mismatched'],
            },
        )[:2]
        removed_ids = set(baseline_train['row_id'][:3])
        changed_rows = [
            {
                **row,
                'premise': (
                    f'{prefix} row {row["row_id"]}'
                    if row['row_id'] in removed_ids else row['premise']
                ),
            }
            for row in base_rows
        ]
        changed_training = datasets.Dataset.from_list(changed_rows).cast_column(
            'label', label_feature
        )
        changed = datasets.DatasetDict({
            'train': changed_training,
            'validation_matched': finals,
            'validation_mismatched': finals,
        })
        protocol = {
            'train_split': 'train',
            'development_fraction': 0.05,
            'development_seed': 42,
            'final_splits': ['validation_matched', 'validation_mismatched'],
            'source_exclusion_stage': 'after_train_development_selection',
            'source_exclusions': [{
                'id': 'fixture', 'field': 'premise', 'match': 'prefix',
                'value': prefix, 'expected_total_matches': 3,
                'expected_matches_by_split': {
                    'training': 3, 'development': 0,
                    'validation_matched': 0, 'validation_mismatched': 0,
                },
                'reason': 'fixture',
            }],
        }
        training, development, observed_finals, audit = select_release_splits(
            changed, 'multi_nli', protocol,
        )
        self.assertEqual(
            list(development['row_id']), list(baseline_development['row_id'])
        )
        self.assertEqual(
            list(training['row_id']),
            [value for value in baseline_train['row_id'] if value not in removed_ids],
        )
        self.assertEqual(audit['excluded_examples_by_split'], {'training': 3})
        self.assertEqual(list(observed_finals['validation_matched']['row_id']), [1000])

    def test_post_selection_exclusion_rejects_split_membership_drift(self):
        prefix = 'target'
        data = datasets.DatasetDict({
            'training': datasets.Dataset.from_list([
                {'premise': prefix, 'hypothesis': 'h', 'label': 0}
            ]),
            'development': datasets.Dataset.from_list([
                {'premise': 'clean', 'hypothesis': 'h', 'label': 1}
            ]),
        })
        rule = {
            'id': 'fixture', 'field': 'premise', 'match': 'prefix',
            'value': prefix, 'expected_total_matches': 1,
            'expected_matches_by_split': {'training': 0, 'development': 1},
            'reason': 'fixture',
        }
        with self.assertRaisesRegex(ValueError, 'split counts changed'):
            apply_source_exclusions(data, [rule])

    def test_environment_cannot_bypass_evaluation_checksum_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_root = root / 'eval_sets'
            base = eval_root / 'snli/final/test'
            original = datasets.Dataset.from_list(ROWS)
            indices = [2, 0]
            expected = datasets.Dataset.from_list([
                {**ROWS[source_index], 'source_index': source_index}
                for source_index in indices
            ])
            tampered = datasets.Dataset.from_list([
                {
                    **ROWS[source_index],
                    'premise': (
                        'tampered' if source_index == 2
                        else ROWS[source_index]['premise']
                    ),
                    'source_index': source_index,
                }
                for source_index in indices
            ])
            original.save_to_disk(str(base / 'original'))
            tampered.save_to_disk(str(base / 'noise'))
            clean = datasets.Dataset.from_list([
                {**ROWS[source_index], 'source_index': source_index}
                for source_index in indices
            ])
            manifest = {
                'schema_version': 4,
                'config_sha256': 'config',
                'seed': 42,
                'label_policies': {
                    'preserve': {'label_changing': False},
                },
                'datasets': {'snli': {
                    'default_development_split': 'validation',
                    'default_final_split': 'test',
                    'splits': {'final:test': {
                        'base_path': 'snli/final/test',
                        'role': 'final',
                        'source_split': 'test',
                        'source_checksum_sha256': logical_dataset_checksum(original),
                        'variants': {'noise': {
                            'path': 'snli/final/test/noise',
                            'paired_clean_storage': 'source_index_view',
                            'checksum_sha256': logical_dataset_checksum(expected),
                            'paired_clean_checksum_sha256': logical_dataset_checksum(clean),
                            'label_policy': 'preserve',
                            'label_changing': False,
                            'evaluation_family': 'fixture',
                        }},
                    }},
                }},
            }
            manifest_path = eval_root / 'dataset_manifest.json'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            original_path = base / 'original'
            original_path.chmod(0o555)
            try:
                with patch.dict(
                    os.environ,
                    {'NLI_VERIFIED_EVAL_MANIFEST_SHA256': manifest_sha},
                ):
                    for load in (
                        lambda: load_eval_suite(
                            eval_root / 'snli', ['noise'],
                            split_role='final', split_name='test',
                        ),
                        lambda: verify_frozen_evaluation_data(root, ['snli']),
                    ):
                        with self.subTest(loader=load):
                            with self.assertRaisesRegex(
                                ValueError, 'Transformed checksum mismatch'
                            ):
                                load()
            finally:
                original_path.chmod(0o755)

    def test_auto_worker_budget_uses_all_cpus_and_distributes_exactly(self):
        with patch('prepare_data_release.os.cpu_count', return_value=16), \
             patch('prepare_eval_sets.os.cpu_count', return_value=16):
            self.assertEqual(resolve_source_worker_budget(0), 16)
            self.assertEqual(resolve_eval_worker_budget(0), 16)
        self.assertEqual(distribute_source_workers(16, 2), [8, 8])
        self.assertEqual(distribute_eval_workers(16, 5), [4, 3, 3, 3, 3])

    def test_parallel_condition_generation_matches_sequential_results(self):
        config = load_experiment_config()
        source_rows = [
            {
                'premise': f'A man sees a dog and a cat number {index}.',
                'hypothesis': 'A woman holds a camera.',
                'label': index % 3,
            }
            for index in range(12)
        ]
        specs = [
            {'id': 'slang', 'kind': 'slang'},
            {'id': 'emoji:emoji_raw', 'kind': 'emoji',
             'condition': 'emoji_raw'},
            {'id': 'emoji:emoji_gloss', 'kind': 'emoji',
             'condition': 'emoji_gloss'},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def context(path):
                path.mkdir()
                return {
                    'source_rows': source_rows,
                    'split_path': str(path),
                    'dataset_name': 'fixture',
                    'split_role': 'development',
                    'split_name': 'validation',
                    'seed': 42,
                    'transformations': config['transformations'],
                }

            sequential = _run_condition_tasks(
                context(root / 'sequential'), specs, workers=1
            )
            try:
                parallel = _run_condition_tasks(
                    context(root / 'parallel'), specs, workers=2
                )
            except PermissionError as error:
                self.skipTest(
                    f'process semaphores are unavailable in this sandbox: {error}'
                )
            self.assertEqual(sequential, parallel)

    def test_paired_clean_predictions_are_indexed_from_original(self):
        original = {
            'predictions': [2, 1, 0], 'labels': [0, 1, 2],
            'source_indices': [0, 1, 2],
            'logits': [[0.0], [1.0], [2.0]],
        }
        clean = datasets.Dataset.from_list([
            {'premise': 'p2', 'hypothesis': 'h2', 'label': 2, 'source_index': 2},
            {'premise': 'p0', 'hypothesis': 'h0', 'label': 0, 'source_index': 0},
        ])
        selected = paired_clean_result_from_original(
            original, clean, save_logits=True
        )
        self.assertEqual(selected['predictions'], [0, 2])
        self.assertEqual(selected['labels'], [2, 0])
        self.assertEqual(selected['source_indices'], [2, 0])
        self.assertEqual(selected['logits'], [[2.0], [0.0]])
        self.assertEqual(selected['prediction_source'], 'indexed_from_original')

    def test_training_loader_returns_manifest_verified_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = datasets.Dataset.from_list(ROWS)
            development = datasets.Dataset.from_list(ROWS[:2])
            training.save_to_disk(str(root / 'source/snli/training'))
            development.save_to_disk(str(root / 'source/snli/development/validation'))
            manifest = {
                'schema_version': 1, 'config_sha256': 'config',
                'datasets': {'snli': {
                    'huggingface_repository': 'stanfordnlp/snli',
                    'huggingface_revision': 'commit',
                    'training': {
                        'path': 'source/snli/training', 'examples': 3,
                        'checksum_sha256': logical_dataset_checksum(training),
                    },
                    'development': {
                        'path': 'source/snli/development/validation', 'examples': 2,
                        'checksum_sha256': logical_dataset_checksum(development),
                    },
                }},
            }
            (root / 'source_manifest.json').write_text(json.dumps(manifest))
            observed_train, observed_dev = load_frozen_training_data(root, 'snli')
            self.assertEqual(observed_train[:], training[:])
            self.assertEqual(observed_dev[:], development[:])
            identity = source_dataset_identity(root, 'snli')
            self.assertEqual(identity['huggingface_revision'], 'commit')
            self.assertEqual(identity['source_exclusions']['total_matches'], 0)
            self.assertEqual(identity['training_checksum_sha256'],
                             logical_dataset_checksum(training))

            with patch('datasets.load_from_disk', wraps=datasets.load_from_disk) as loader:
                development_only = load_frozen_development_data(root, 'snli')
            self.assertEqual(development_only[:], development[:])
            loader.assert_called_once_with(
                str(root / 'source/snli/development/validation')
            )

            manifest['datasets']['snli']['training']['checksum_sha256'] = '0' * 64
            manifest_path = root / 'source_manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            with patch.dict(
                os.environ,
                {'NLI_VERIFIED_SOURCE_MANIFEST_SHA256': manifest_sha},
            ):
                with self.assertRaisesRegex(ValueError, 'training checksum changed'):
                    load_frozen_training_data(root, 'snli')

    def test_paired_clean_is_reconstructed_from_source_indices_without_duplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / 'snli/final/test'
            original = datasets.Dataset.from_list(ROWS)
            transformed_rows = [
                {'premise': 'changed2', 'hypothesis': 'h2', 'label': 2,
                 'source_index': 2},
                {'premise': 'changed0', 'hypothesis': 'h0', 'label': 0,
                 'source_index': 0},
            ]
            transformed = datasets.Dataset.from_list(transformed_rows)
            original.save_to_disk(str(base / 'original'))
            transformed.save_to_disk(str(base / 'noise'))
            clean_rows = paired_clean_records(ROWS, transformed_rows)
            manifest = {
                'schema_version': 3, 'config_sha256': 'config', 'seed': 42,
                'datasets': {'snli': {
                    'default_development_split': 'validation',
                    'default_final_split': 'test',
                    'splits': {'final:test': {
                        'base_path': 'snli/final/test', 'role': 'final',
                        'source_split': 'test',
                        'source_checksum_sha256': logical_dataset_checksum(original),
                        'variants': {'noise': {
                            'path': 'snli/final/test/noise',
                            'paired_clean_storage': 'source_index_view',
                            'checksum_sha256': logical_dataset_checksum(transformed),
                            'paired_clean_checksum_sha256':
                                logical_dataset_checksum(clean_rows),
                        }},
                    }},
                }},
            }
            (root / 'dataset_manifest.json').write_text(json.dumps(manifest))
            suite = load_eval_suite(root / 'snli', variants=['noise'])
            self.assertEqual(list(suite['clean_noise']['source_index']), [2, 0])
            self.assertEqual(list(suite['clean_noise']['premise']), ['p2', 'p0'])
            self.assertFalse((base / 'clean_noise').exists())


if __name__ == '__main__':
    unittest.main()
