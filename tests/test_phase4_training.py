"""Acceptance tests for implementation-plan Phase 4."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import run_all_v2  # noqa: E402
import train_hybrid  # noqa: E402
from experiment_config import (config_hash, load_experiment_config,  # noqa: E402
                               model_config)
from helpers import (limit_eval_suite, paired_clean_result,  # noqa: E402
                     paired_clean_result_from_original, smoke_nli_records)
from on_the_fly_training import (  # noqa: E402
    AuditedCleanDataset,
    OnTheFlyMarkerDataset,
    PRESENTATION_SEED_MIXER,
    prepare_epoch_tokenized_dataset,
    presentation_seed,
    validate_training_audit,
    validate_training_world_size,
)
from training_metadata import (  # noqa: E402
    SYSTEM_STAGES,
    planned_optimizer_steps,
    system_stage_metadata,
    training_run_metadata,
    validate_training_run_metadata,
)
from training_runtime import (mixed_precision_kwargs,  # noqa: E402
                              tokenization_cache_operation)


class RecordingTokenizer:
    def __call__(self, premise, hypothesis, **kwargs):
        del kwargs
        return {"input_ids": [premise, hypothesis], "attention_mask": [1, 1]}


SOURCES = [
    {"premise": "A man sees a dog.", "hypothesis": "A dog moves.", "label": 0},
    {"premise": "A woman has a camera.", "hypothesis": "A woman waits.", "label": 1},
]


class ComputeMatchingTests(unittest.TestCase):
    def test_presentation_seed_hash_prevents_shifted_run_streams(self):
        self.assertEqual(
            load_experiment_config()['training']['augmentation_seed_mixer'],
            PRESENTATION_SEED_MIXER,
        )
        self.assertEqual(presentation_seed(13, 1, 108), presentation_seed(13, 1, 108))
        self.assertNotEqual(
            presentation_seed(13, 1, 108),
            presentation_seed(21, 1, 100),
        )

    def test_epoch_preparation_uses_disk_cache_when_no_cache_is_supplied(self):
        class BatchTokenizer:
            name_or_path = 'fixture-tokenizer'

            def __call__(self, premise, hypothesis, **kwargs):
                del kwargs
                if isinstance(premise, list):
                    return {
                        'input_ids': [[len(first), len(second)]
                                      for first, second in zip(premise, hypothesis)],
                        'attention_mask': [[1, 1] for _ in premise],
                    }
                return {'input_ids': [len(premise), len(hypothesis)],
                        'attention_mask': [1, 1]}

        prepared = prepare_epoch_tokenized_dataset(
            Dataset.from_list(SOURCES), BatchTokenizer(), 32, epochs=2,
            augmentation_probability=0.5,
            marker_placements=('hypothesis_suffix',), seed=13,
        )
        self.assertIsNotNone(prepared._cache_owner)
        self.assertTrue(Path(prepared._cache_owner.name).is_dir())
        self.assertTrue(all(epoch.cache_files for epoch in prepared.epochs))
        prepared.close()

    def test_on_the_fly_dataset_honors_model_input_preprocessing(self):
        dataset = OnTheFlyMarkerDataset(
            [{"premise": "@alice sees https://example.test", "hypothesis": "x",
              "label": 0}],
            RecordingTokenizer(), 32, augmentation_probability=0.0,
            marker_placements=("hypothesis_suffix",), seed=4,
            input_preprocessing='timelm',
        )
        encoded = dataset[0]
        self.assertEqual(encoded['input_ids'][0], '@user sees http')
        self.assertEqual(encoded['label'], 0)
        self.assertNotIn('labels', encoded)

    def test_paired_clean_condition_fails_if_original_is_not_first(self):
        clean = Dataset.from_list([
            {'premise': 'p', 'hypothesis': 'h', 'label': 0, 'source_index': 0}
        ])
        with self.assertRaisesRegex(ValueError, 'evaluated first'):
            paired_clean_result({}, 'clean_noise', clean)

    def test_tokenization_cache_identity_covers_the_full_input_contract(self):
        base = tokenization_cache_operation(
            'model', 'revision', 128, 'evaluation', tokenizer_backend='fast',
            tokenizer_normalization='default', input_preprocessing='none',
        )
        changed = {
            tokenization_cache_operation(
                'model', 'revision', 128, 'evaluation', tokenizer_backend=backend,
                tokenizer_normalization=normalization,
                input_preprocessing=preprocessing,
            )
            for backend, normalization, preprocessing in (
                ('slow', 'default', 'none'),
                ('fast', 'disabled', 'none'),
                ('fast', 'default', 'timelm'),
            )
        }
        self.assertNotIn(base, changed)
        self.assertEqual(len(changed), 3)

    def test_auto_precision_prefers_bf16_when_supported(self):
        supported = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True, is_bf16_supported=lambda: True,
        ))
        fallback = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True, is_bf16_supported=lambda: False,
        ))
        self.assertEqual(
            mixed_precision_kwargs(supported, 'auto'), {'fp16': False, 'bf16': True}
        )
        self.assertEqual(
            mixed_precision_kwargs(fallback, 'auto'), {'fp16': True, 'bf16': False}
        )

    def test_hybrid_checkpoint_selection_tokenizes_frozen_development_rows(self):
        train = Dataset.from_list([
            {'premise': 'TRAIN', 'hypothesis': 'train', 'label': 0}
        ])
        development = Dataset.from_list([
            {'premise': 'DEVELOPMENT', 'hypothesis': 'dev', 'label': 1}
        ])
        final = Dataset.from_list([
            {'premise': 'FINAL', 'hypothesis': 'final', 'label': 2}
        ])

        class StopBeforeModelLoad(Exception):
            pass

        argv = [
            'train_hybrid.py', '--smoke_test', '--data_release', '/release',
            '--eval_sets_dir', '/eval', '--smoke_train_samples', '1',
            '--smoke_eval_samples', '1',
        ]
        with patch.object(sys, 'argv', argv), \
             patch.object(train_hybrid, 'load_frozen_training_data',
                          return_value=(train, development)), \
             patch.object(train_hybrid, 'load_eval_suite',
                          return_value={'original': final}), \
             patch.object(train_hybrid, 'load_tokenizer',
                          return_value=SimpleNamespace(name_or_path='fixture')), \
             patch.object(train_hybrid, 'prepare_epoch_tokenized_dataset',
                          return_value=object()), \
             patch.object(train_hybrid, 'map_dataset_locally',
                          side_effect=lambda dataset, *args, **kwargs: dataset) as mapper, \
             patch.object(
                 train_hybrid.AutoModelForSequenceClassification, 'from_pretrained',
                 side_effect=StopBeforeModelLoad,
             ):
            with self.assertRaises(StopBeforeModelLoad):
                train_hybrid.main()
        checkpoint_selection_rows = mapper.call_args_list[-1].args[0]
        self.assertEqual(checkpoint_selection_rows[0]['premise'], 'DEVELOPMENT')

    def test_limited_fixed_eval_retains_original_rows_needed_by_pairs(self):
        original = Dataset.from_list([
            {'premise': str(index), 'hypothesis': 'x', 'label': index % 3}
            for index in range(10)
        ])
        transformed = Dataset.from_list([
            {'premise': 'emoji', 'hypothesis': 'x', 'label': index % 3,
             'source_index': index}
            for index in (3, 7, 9)
        ])
        clean = Dataset.from_list([
            {'premise': str(index), 'hypothesis': 'x', 'label': index % 3,
             'source_index': index}
            for index in (3, 7, 9)
        ])
        limited = limit_eval_suite({
            'original': original, 'emoji_raw': transformed,
            'clean_emoji_raw': clean,
        }, 2)
        self.assertEqual(list(limited['emoji_raw']['source_index']), [3, 7])
        self.assertEqual(list(limited['clean_emoji_raw']['source_index']), [3, 7])
        self.assertEqual(list(limited['original']['source_index']), [3, 7])
        paired = paired_clean_result_from_original({
            'predictions': [0, 1],
            'labels': [0, 1],
            'source_indices': list(limited['original']['source_index']),
        }, limited['clean_emoji_raw'])
        self.assertEqual(paired['source_indices'], [3, 7])
        self.assertEqual(paired['prediction_source'], 'indexed_from_original')

    def test_smoke_records_are_deterministic_balanced_and_local(self):
        first = smoke_nli_records(6)
        second = smoke_nli_records(6)
        self.assertEqual(first, second)
        self.assertEqual(first["label"], [0, 1, 2, 0, 1, 2])
        self.assertEqual(len(first["premise"]), 6)

    def test_primary_training_systems_have_identical_planned_step_budget(self):
        budgets = {
            system: planned_optimizer_steps(101, 8, 2, 3)
            for system in (
                "clean_baseline", "marker_augmentation",
                "stage_matched_hybrid", "compute_matched_clean_control",
            )
        }
        self.assertEqual(len(set(budgets.values())), 1)
        self.assertEqual(next(iter(budgets.values())), 21)

    def test_clean_control_uses_same_path_without_transforming_examples(self):
        dataset = OnTheFlyMarkerDataset(
            SOURCES, RecordingTokenizer(), 32,
            augmentation_probability=0.0,
            marker_placements=("hypothesis_suffix",), seed=4,
        )
        for epoch in range(3):
            dataset.set_epoch(epoch)
            for index in range(len(dataset)):
                encoded = dataset[index]
                self.assertEqual(encoded["input_ids"], [
                    SOURCES[index]["premise"], SOURCES[index]["hypothesis"]
                ])
        audit = dataset.audit(3)
        validate_training_audit(audit)
        self.assertEqual(audit["transformed_presentations"], 0)
        self.assertEqual(audit["unique_presentations"], len(SOURCES) * 3)

    def test_duplicate_example_epoch_presentation_fails_the_audit(self):
        dataset = OnTheFlyMarkerDataset(
            SOURCES, RecordingTokenizer(), 32,
            augmentation_probability=0.5,
            marker_placements=("hypothesis_suffix",), seed=4,
        )
        dataset[0]
        dataset[0]
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            validate_training_audit(dataset.audit(1))

    def test_clean_baseline_audit_is_observed(self):
        dataset = AuditedCleanDataset([
            {"input_ids": [1], "attention_mask": [1], "labels": 0},
            {"input_ids": [2], "attention_mask": [1], "labels": 1},
        ])
        for index in range(len(dataset)):
            dataset[index]
        audit = dataset.audit(1)
        validate_training_audit(audit)
        self.assertTrue(audit["observed"])
        self.assertEqual(audit["total_fetches"], 2)

    def test_distributed_training_is_rejected_before_audit(self):
        validate_training_world_size(1)
        with self.assertRaisesRegex(RuntimeError, "world_size=1"):
            validate_training_world_size(2)

    def test_training_metadata_embeds_audit_and_enforces_step_budget(self):
        audit = {
            "policy": "one presentation", "dataset_examples": 10,
            "completed_epochs": 1, "expected_presentations": 10,
            "unique_presentations": 10, "total_fetches": 10,
            "duplicate_presentations": 0, "missing_presentations": 0,
            "transformed_presentations": 4, "observed": True,
        }
        metadata = training_run_metadata(
            system_id="marker_augmentation", train_examples=10, epochs=1,
            batch_size=2, gradient_accumulation_steps=1, warmup_ratio=0.0,
            learning_rate=1e-5, actual_optimizer_steps=5, runtime_seconds=1.0,
            checkpoint_selection="fixture", presentation_audit=audit,
        )
        validate_training_run_metadata(metadata)
        self.assertEqual(
            metadata["training_presentations"]["transformed_presentations"], 4
        )
        metadata["actual_optimizer_steps"] = 4
        with self.assertRaisesRegex(RuntimeError, "Optimizer-step budget mismatch"):
            validate_training_run_metadata(metadata)

    def test_frozen_checkpoint_policy_is_forwarded_to_training_cli(self):
        experiment = load_experiment_config()
        cfg = model_config(experiment, "electra")
        common = run_all_v2.build_common(
            cfg, experiment["training"], "snli", 42, False, Path("/eval")
        )
        flag_index = common.index("--checkpoint_selection")
        self.assertEqual(
            common[flag_index + 1], experiment["training"]["checkpoint_selection"]
        )

    def test_model_input_contract_and_tiny_smoke_sizes_are_forwarded(self):
        experiment = load_experiment_config()
        cfg = model_config(experiment, 'bertweet')
        common = run_all_v2.build_common(
            cfg, experiment['training'], 'snli', 42, True, None,
            smoke_train_samples=100, smoke_eval_samples=30,
        )
        self.assertEqual(common[common.index('--tokenizer_backend') + 1], 'slow')
        self.assertEqual(
            common[common.index('--tokenizer_normalization') + 1], 'disabled'
        )
        self.assertEqual(common[common.index('--smoke_train_samples') + 1], '100')
        self.assertEqual(common[common.index('--smoke_eval_samples') + 1], '30')
        self.assertEqual(common[common.index('--mixed_precision') + 1], 'auto')


class StageSeparationTests(unittest.TestCase):
    def test_exactly_six_named_systems_are_declared(self):
        config = load_experiment_config()
        self.assertEqual(set(config["systems"]), set(SYSTEM_STAGES))

    def test_hybrid_never_deletes_markers(self):
        metadata = system_stage_metadata("stage_matched_hybrid")
        self.assertEqual(
            metadata["transformations"]["training"],
            ["deterministic_epoch_marker_augmentation"],
        )
        self.assertEqual(
            metadata["transformations"]["inference"],
            ["bijective_emoji_normalization"],
        )
        self.assertNotIn(
            "known_marker_deletion",
            metadata["transformations"]["training"]
            + metadata["transformations"]["inference"],
        )

    def test_marker_deletion_is_explicitly_an_oracle(self):
        metadata = system_stage_metadata("known_marker_deletion_oracle")
        self.assertEqual(metadata["classification"], "oracle")
        self.assertNotEqual(metadata["system_id"], "emoji_normalization")

    def test_manifest_records_train_and_inference_stages(self):
        experiment = load_experiment_config()
        cfg = model_config(experiment, "electra")
        with tempfile.TemporaryDirectory() as tmp:
            output = run_all_v2.materialize_manifest(
                tmp, "hybrid_fold_1", cfg, experiment,
                REPOSITORY_ROOT / "configs" / "experiment_v6.json",
                "snli", 42, "test", "final", ["emoji"],
                system_id="stage_matched_hybrid",
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
        overrides = manifest["runtime_overrides"]
        self.assertEqual(overrides["system_id"], "stage_matched_hybrid")
        self.assertEqual(
            overrides["transformations"]["inference"],
            ["bijective_emoji_normalization"],
        )

    def test_reusing_manifest_preserves_original_creation_timestamp(self):
        experiment = load_experiment_config()
        cfg = model_config(experiment, 'electra')
        with tempfile.TemporaryDirectory() as tmp:
            first = run_all_v2.materialize_manifest(
                tmp, 'baseline', cfg, experiment,
                REPOSITORY_ROOT / 'configs/experiment_v6.json',
                'snli', 42, 'test', 'final', ['emoji'],
                system_id='clean_baseline',
            )
            before = json.loads((first / 'run_manifest.json').read_text())
            second = run_all_v2.materialize_manifest(
                tmp, 'baseline', cfg, experiment,
                REPOSITORY_ROOT / 'configs/experiment_v6.json',
                'snli', 42, 'test', 'final', ['emoji'],
                system_id='clean_baseline',
            )
            after = json.loads((second / 'run_manifest.json').read_text())
        self.assertEqual(before['created_at_utc'], after['created_at_utc'])


class SeedRunnerTests(unittest.TestCase):
    def test_frozen_models_have_model_specific_sweeps_and_required_seeds(self):
        config = load_experiment_config()
        electra = config["models"]["electra"]
        roberta = config["models"]["roberta"]
        self.assertGreaterEqual(len(electra["seeds"]), 5)
        self.assertGreaterEqual(len(roberta["seeds"]), 3)
        self.assertNotEqual(
            electra["learning_rate_sweep"], roberta["learning_rate_sweep"]
        )

    def test_all_route_dispatches_all_six_systems(self):
        experiment = load_experiment_config()
        cfg = model_config(experiment, "electra")
        with patch.object(run_all_v2, "step_baseline") as baseline, \
             patch.object(run_all_v2, "step_augmentation") as augmentation, \
             patch.object(run_all_v2, "step_preprocessing") as preprocessing, \
             patch.object(run_all_v2, "step_hybrid") as hybrid, \
             patch.object(run_all_v2, "step_clean_control") as clean_control, \
             patch.object(run_all_v2, "step_stats"), \
             patch.object(run_all_v2, "validate_smoke_matrix"):
            run_all_v2.run_step(
                "all", cfg, experiment, Path("config.json"), Path("/tmp/out"),
                "snli", 42, True, None, "test", "fold_1", ["emoji"], "final",
            )
        baseline.assert_called_once()
        augmentation.assert_called_once()
        hybrid.assert_called_once()
        clean_control.assert_called_once()
        self.assertEqual(preprocessing.call_count, 2)
        self.assertEqual(
            preprocessing.call_args_list[1].kwargs["mode"], "marker_oracle"
        )

    def test_failed_run_is_persisted_in_status_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                run_all_v2.subprocess, "run",
                return_value=SimpleNamespace(returncode=7),
            ):
                with self.assertRaises(SystemExit):
                    run_all_v2.run(["fixture", "--flag"], "Fixture", tmp)
            with patch.object(
                run_all_v2.subprocess, "run",
                return_value=SimpleNamespace(returncode=0),
            ):
                run_all_v2.run(["fixture", "--flag"], "Fixture", tmp)
            status = json.loads((Path(tmp) / "run_status.json").read_text())
            attempts = [json.loads(line) for line in
                        (Path(tmp) / "run_attempts.jsonl").read_text().splitlines()]
        self.assertEqual(status["status"], "completed")
        self.assertEqual([attempt["status"] for attempt in attempts], ["failed", "completed"])
        self.assertEqual(attempts[0]["exit_code"], 7)
        self.assertEqual(status["command"], ["fixture", "--flag"])

    def test_fold_independent_baseline_uses_one_all_variant_identity(self):
        experiment = load_experiment_config()
        cfg = model_config(experiment, "electra")
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(run_all_v2, "run"):
            for fold in ("fold_1", "fold_2"):
                run_all_v2.step_baseline(
                    cfg, tmp, [], experiment,
                    REPOSITORY_ROOT / "configs" / "experiment_v6.json",
                    "snli", 42, "test", "final", ["all"], fold,
                )
            paths = sorted(Path(tmp).glob("electra_baseline*_final_test"))
        self.assertEqual(
            [path.name for path in paths],
            ["electra_baseline_final_test"],
        )

    def test_secondary_split_reuses_primary_checkpoint_without_training(self):
        experiment = load_experiment_config()
        cfg = dict(
            model_config(experiment, 'electra'),
            run_mode='production',
            reuse_trained_from_split='validation_matched',
            eval_sets_dir=Path('/eval'),
            preprocessing_workers=4,
            model_revision='model-sha',
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'electra_baseline_final_validation_matched'
            (source / 'final_model').mkdir(parents=True)
            (source / 'final_model/config.json').write_text('{}')
            (source / 'run_status.json').write_text(json.dumps({
                'status': 'completed', 'exit_code': 0,
            }))
            (source / 'results.json').write_text(json.dumps({
                'system_id': 'clean_baseline',
                'run_metadata': {'run_origin': 'frozen_config'},
            }))
            (source / 'predictions.json').write_text('{}')
            (source / 'predictions_manifest.json').write_text('{}')
            (source / 'run_manifest.json').write_text(json.dumps({
                'config_sha256': config_hash(experiment),
                'runtime_overrides': {
                    'system_id': 'clean_baseline', 'dataset': 'multi_nli',
                    'seed': 13, 'learning_rate': cfg['lr'],
                    'run_mode': 'production',
                    'reported_evaluation_role': 'final',
                    'reported_evaluation_split': 'validation_matched',
                    'evaluated_variants': ['all'],
                    'model_revision': 'model-sha',
                    'tokenizer_backend': 'fast',
                    'tokenizer_normalization': 'default',
                    'input_preprocessing': 'none',
                    'mixed_precision': 'auto',
                    'trained_checkpoint_split': 'validation_matched',
                },
            }))
            with patch.object(run_all_v2, 'run') as run:
                run_all_v2.step_baseline(
                    cfg, tmp, [], experiment,
                    REPOSITORY_ROOT / 'configs/experiment_v6.json',
                    'multi_nli', 13, 'validation_mismatched', 'final',
                    ['all'], 'fold_1',
                )
            command = run.call_args.args[0]
            self.assertIn('src/evaluate_checkpoint.py', command)
            self.assertEqual(
                command[command.index('--source_run_dir') + 1], str(source)
            )
            self.assertNotIn('src/train_baseline.py', command)

    def test_reused_checkpoint_validation_covers_full_runtime_contract(self):
        experiment = load_experiment_config()
        cfg = dict(
            model_config(experiment, 'electra'),
            run_mode='production',
            reuse_trained_from_split='validation_matched',
            eval_sets_dir=Path('/eval'),
            data_release_id='release-id',
            source_manifest_sha256='source-sha',
            model_revision='model-sha',
            tokenizer_backend='slow',
            tokenizer_normalization='disabled',
            input_preprocessing='timelm',
            mixed_precision='bf16',
        )
        with patch.object(run_all_v2, 'validate_completed_run') as validate, \
             patch.object(run_all_v2, 'run'):
            reused = run_all_v2.evaluate_reused_checkpoint(
                cfg=cfg,
                p='/runs',
                approach='baseline',
                source_approach='baseline',
                system_id='clean_baseline',
                source_system_id='clean_baseline',
                experiment=experiment,
                dataset='multi_nli',
                seed=13,
                output_dir=Path('/out'),
                eval_split='validation_mismatched',
                eval_role='final',
                eval_variants=['all'],
                eval_sets_dir=Path('/eval'),
            )
        self.assertTrue(reused)
        expected = validate.call_args.kwargs['expected_manifest']
        self.assertEqual(expected['data_release_id'], 'release-id')
        self.assertEqual(expected['source_manifest_sha256'], 'source-sha')
        self.assertEqual(expected['model_revision'], 'model-sha')
        self.assertEqual(expected['tokenizer_backend'], 'slow')
        self.assertEqual(expected['tokenizer_normalization'], 'disabled')
        self.assertEqual(expected['input_preprocessing'], 'timelm')
        self.assertEqual(expected['mixed_precision'], 'bf16')
        self.assertEqual(
            expected['trained_checkpoint_split'], 'validation_matched'
        )

    def test_missing_baseline_is_recorded_as_blocked(self):
        experiment = load_experiment_config()
        cfg = model_config(experiment, "electra")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                run_all_v2.step_preprocessing(
                    cfg, experiment["training"], tmp, "snli", 42, True, None,
                    experiment, REPOSITORY_ROOT / "configs" / "experiment_v6.json",
                    "test", ["emoji"], "final", marker_fold="fold_2",
                )
            status_path = (
                Path(tmp) / "electra_preprocessing_final_test" / "run_status.json"
            )
            status = json.loads(status_path.read_text())
            attempts = status_path.with_name("run_attempts.jsonl").read_text().splitlines()
        self.assertEqual(status["status"], "blocked")
        self.assertEqual(len(attempts), 1)

    def test_failed_baseline_checkpoint_is_not_reused(self):
        experiment = load_experiment_config()
        cfg = dict(model_config(experiment, "electra"), run_mode="smoke")
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "electra_baseline_final_test"
            (baseline / "final_model").mkdir(parents=True)
            (baseline / "final_model" / "config.json").write_text("{}")
            (baseline / "run_status.json").write_text(
                json.dumps({"status": "failed", "exit_code": 1})
            )
            (baseline / "run_manifest.json").write_text("{}")
            (baseline / "results.json").write_text(json.dumps({
                "system_id": "clean_baseline",
                "run_metadata": {"run_origin": "frozen_config"},
            }))
            (baseline / "predictions.json").write_text("{}")
            with self.assertRaises(SystemExit):
                run_all_v2.step_preprocessing(
                    cfg, experiment["training"], tmp, "snli", 42, True, None,
                    experiment, REPOSITORY_ROOT / "configs" / "experiment_v6.json",
                    "test", ["all"], "final",
                )
            status = json.loads(
                (Path(tmp) / "electra_preprocessing_final_test" / "run_status.json").read_text()
            )
        self.assertEqual(status["status"], "blocked")
        self.assertIn("status='failed'", status["error"])

    def test_smoke_and_production_scopes_are_distinct(self):
        experiment = load_experiment_config()
        cfg = model_config(experiment, "electra")
        with patch.object(run_all_v2, "step_baseline") as baseline:
            run_all_v2.run_step(
                "baseline", cfg, experiment, Path("config.json"), Path("/runs"),
                "snli", 42, True, None, "test", "fold_1", ["all"], "final",
            )
            smoke_out = baseline.call_args.args[1]
            run_all_v2.run_step(
                "baseline", cfg, experiment, Path("config.json"), Path("/runs"),
                "snli", 42, False, Path("/eval"), "test", "fold_1", ["all"], "final",
            )
            production_out = baseline.call_args.args[1]
        self.assertEqual(smoke_out, "/runs/smoke")
        self.assertEqual(production_out, "/runs")

    def test_direct_principal_system_is_labeled_exploratory(self):
        metadata = training_run_metadata(
            system_id="marker_augmentation", train_examples=10, epochs=1,
            batch_size=2, gradient_accumulation_steps=1, warmup_ratio=0.0,
            learning_rate=1e-5, actual_optimizer_steps=5, runtime_seconds=1.0,
            checkpoint_selection="fixture", frozen_run=False,
        )
        self.assertEqual(metadata["classification"], "exploratory")
        self.assertEqual(metadata["run_origin"], "direct_override")


if __name__ == "__main__":
    unittest.main()


class SentencePieceByteFallbackTests(unittest.TestCase):
    """Contract tests for the additive ``spm_byte_fallback`` tokenizer normalization."""

    DEBERTA = "microsoft/deberta-v3-base"
    REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"

    @classmethod
    def _deberta_snapshot(cls):
        from huggingface_hub import constants

        snapshot = (
            Path(constants.HF_HUB_CACHE)
            / "models--microsoft--deberta-v3-base" / "snapshots" / cls.REVISION
        )
        required = {"spm.model", "tokenizer_config.json"}
        if snapshot.is_dir() and required <= {p.name for p in snapshot.iterdir()}:
            return snapshot
        return None

    def test_value_is_accepted_everywhere_and_changes_the_cache_key(self):
        from copy import deepcopy

        from experiment_config import validate_config
        from training_runtime import (
            TOKENIZER_NORMALIZATION, model_source_metadata,
            tokenization_cache_operation,
        )

        self.assertIn("spm_byte_fallback", TOKENIZER_NORMALIZATION)
        config = deepcopy(load_experiment_config())
        config["models"]["roberta_base"]["tokenizer_normalization"] = "spm_byte_fallback"
        validate_config(config)  # must not raise
        keys = {
            tokenization_cache_operation(
                "m", "r", 128, "training", tokenizer_backend="fast",
                tokenizer_normalization=value, input_preprocessing="none",
            )
            for value in ("default", "spm_byte_fallback")
        }
        self.assertEqual(len(keys), 2)
        metadata = model_source_metadata(
            "m", "r", tokenizer_backend="fast",
            tokenizer_normalization="spm_byte_fallback", input_preprocessing="none",
        )
        self.assertEqual(metadata["tokenizer_normalization"], "spm_byte_fallback")
        # The extension config and its CLI consumers carry the value through.
        extension = load_experiment_config(
            REPOSITORY_ROOT / "configs" / "experiment_v6_deberta_ext.json"
        )
        self.assertEqual(
            extension["models"]["deberta_v3_base"]["tokenizer_normalization"],
            "spm_byte_fallback",
        )
        for script in (
            "train_baseline", "train_preprocessing", "evaluate_checkpoint",
            "train_hybrid", "train_augmentation",
        ):
            source = (SOURCE_ROOT / f"{script}.py").read_text(encoding="utf-8")
            self.assertIn("'spm_byte_fallback'", source, script)

    def test_non_unigram_tokenizer_is_rejected_loudly(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        from training_runtime import load_tokenizer

        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        with tempfile.TemporaryDirectory() as tmp:
            PreTrainedTokenizerFast(
                tokenizer_object=backend, unk_token="[UNK]"
            ).save_pretrained(tmp)
            # Unchanged behaviour for existing values.
            load_tokenizer(tmp, None, backend="fast", normalization="default")
            with self.assertRaisesRegex(RuntimeError, "Unigram"):
                load_tokenizer(tmp, None, backend="fast", normalization="spm_byte_fallback")

    def test_deberta_byte_fallback_gives_distinct_emoji_and_survives_save_reload(self):
        import os

        snapshot = self._deberta_snapshot()
        if snapshot is None:
            self.skipTest("microsoft/deberta-v3-base snapshot is not in the local HF cache")
        from training_runtime import load_tokenizer, unigram_byte_fallback_state

        emoji = load_experiment_config()["transformations"]["main_emoji_map"]
        previous = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            default = load_tokenizer(
                self.DEBERTA, self.REVISION, backend="fast", normalization="default"
            )
            patched = load_tokenizer(
                self.DEBERTA, self.REVISION, backend="fast",
                normalization="spm_byte_fallback",
            )
            self.assertIs(unigram_byte_fallback_state(default), False)
            self.assertIs(unigram_byte_fallback_state(patched), True)
            self.assertEqual(len(default), len(patched))
            unk = patched.unk_token_id
            sequences = {
                key: tuple(patched(value, add_special_tokens=False)["input_ids"])
                for key, value in emoji.items()
            }
            self.assertEqual(len(set(sequences.values())), 16)
            self.assertFalse(any(unk in seq for seq in sequences.values()))
            # The default contract really does collapse emoji, so the new value matters.
            default_sequences = {
                tuple(default(value, add_special_tokens=False)["input_ids"])
                for value in emoji.values()
            }
            self.assertLess(len(default_sequences), 16)

            probes = list(emoji.values()) + [
                "A 👩 with a 🐕 rides a 🚲. no cap",
                "Two 🧒 pet a 🐈 and a 🐎 frfr",
            ]
            before = [patched(p, add_special_tokens=False)["input_ids"] for p in probes]
            pair_before = patched("A 👤 sits.", "A 🐈 sleeps.")["input_ids"]
            with tempfile.TemporaryDirectory() as tmp:
                patched.save_pretrained(tmp)
                saved = json.loads((Path(tmp) / "tokenizer.json").read_text())["model"]
                self.assertEqual(saved["type"], "Unigram")
                self.assertIs(saved.get("byte_fallback"), True)
                reloaded = load_tokenizer(
                    tmp, None, backend="fast", normalization="spm_byte_fallback"
                )
                self.assertIs(unigram_byte_fallback_state(reloaded), True)
                after = [reloaded(p, add_special_tokens=False)["input_ids"] for p in probes]
                self.assertEqual(before, after)
                self.assertEqual(pair_before, reloaded("A 👤 sits.", "A 🐈 sleeps.")["input_ids"])
        finally:
            if previous is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous


class Float32MasterWeightTests(unittest.TestCase):
    """Fine-tuning must always instantiate fp32 master weights (transformers 5 defaults to dtype='auto')."""

    DEBERTA = "microsoft/deberta-v3-base"
    REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"

    def test_model_kwargs_pin_fp32_and_tokenizer_kwargs_do_not(self):
        import torch

        from training_runtime import (
            assert_float32_parameters, model_pretrained_kwargs, pretrained_kwargs,
            model_source_metadata,
        )

        self.assertEqual(
            model_pretrained_kwargs("sha"), {"revision": "sha", "dtype": torch.float32}
        )
        self.assertEqual(model_pretrained_kwargs(None), {"dtype": torch.float32})
        self.assertNotIn("dtype", pretrained_kwargs("sha"))
        # Trainers and evaluators all go through the fp32 helper and the guard.
        for script in (
            "train_baseline", "train_augmentation", "train_hybrid",
            "train_preprocessing", "evaluate_checkpoint",
        ):
            source = (SOURCE_ROOT / f"{script}.py").read_text(encoding="utf-8")
            self.assertIn("model_pretrained_kwargs(", source, script)
            self.assertIn("assert_float32_parameters(", source, script)
            self.assertNotIn("**pretrained_kwargs(args.model_revision)\n    )", source, script)
        # The guard fires on fp16 parameters and reports the dtype.
        half = torch.nn.Linear(2, 2).to(torch.float16)
        with self.assertRaisesRegex(RuntimeError, "non-fp32"):
            assert_float32_parameters(half, "half")
        self.assertEqual(assert_float32_parameters(torch.nn.Linear(2, 2)), "torch.float32")
        metadata = model_source_metadata("m", "r", parameter_dtype="torch.float32")
        self.assertEqual(metadata["parameter_dtype"], "torch.float32")
        self.assertNotIn("parameter_dtype", model_source_metadata("m", "r"))

    def test_deberta_v3_base_loads_with_fp32_parameters(self):
        import os

        from huggingface_hub import constants

        snapshot = (
            Path(constants.HF_HUB_CACHE)
            / "models--microsoft--deberta-v3-base" / "snapshots" / self.REVISION
        )
        if not (snapshot / "pytorch_model.bin").is_file():
            self.skipTest("microsoft/deberta-v3-base weights are not in the local HF cache")
        import torch
        from transformers import AutoModelForSequenceClassification

        from training_runtime import (
            assert_float32_parameters, model_pretrained_kwargs, pretrained_kwargs,
        )

        previous = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.DEBERTA, num_labels=3, **model_pretrained_kwargs(self.REVISION)
            )
            self.assertEqual(assert_float32_parameters(model, self.DEBERTA), "torch.float32")
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            self.assertEqual(model.classifier.weight.dtype, torch.float32)
            # Regression witness: the old construction path yields fp16 parameters.
            legacy = AutoModelForSequenceClassification.from_pretrained(
                self.DEBERTA, num_labels=3, **pretrained_kwargs(self.REVISION)
            )
            if next(legacy.parameters()).dtype != torch.float32:
                with self.assertRaises(RuntimeError):
                    assert_float32_parameters(legacy, self.DEBERTA)
        finally:
            if previous is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous
