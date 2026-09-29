"""Hard acceptance tests for implementation-plan Phase 2."""

from __future__ import annotations

import json
from types import SimpleNamespace
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pipeline_audit import (audit_pipeline_identity,  # noqa: E402
                            reconstruct_with_preprocessing)
from helpers import robustness_drops  # noqa: E402
from prepare_eval_sets import (  # noqa: E402
    atomic_output_directory,
    dataset_checksum,
    generate_audited_records,
    paired_clean_records,
    validate_principal_audit,
)
from on_the_fly_training import (OnTheFlyMarkerDataset,  # noqa: E402
                                 validate_training_audit)
from statistical_tests import (_align_to_reference, _holm_adjust,
                               run_tests)  # noqa: E402
from run_all_v2 import scoped_run_dir  # noqa: E402
import run_all_v2  # noqa: E402
from preprocessing import (  # noqa: E402
    KNOWN_MARKERS,
    audit_marker_removal,
    normalize_emoji,
    remove_markers,
)
from transforms import (  # noqa: E402
    HELD_OUT_MARKERS,
    TRAIN_MARKERS,
    canonicalize_whitespace,
    transform_example,
    apply_training_condition,
    validate_transform_condition,
)


class DeterministicTokenizer:
    """Tiny pair tokenizer implementing the interface used by the audit."""

    def __call__(self, premise, hypothesis, *, max_length, truncation, padding,
                 return_tensors):
        del truncation, padding, return_tensors
        tokens = [101]
        tokens.extend(sum(ord(char) for char in word) % 997 + 3 for word in premise.split())
        tokens.append(102)
        tokens.extend(sum(ord(char) for char in word) % 997 + 3 for word in hypothesis.split())
        tokens.append(102)
        tokens = tokens[:max_length]
        mask = [1] * len(tokens)
        tokens += [0] * (max_length - len(tokens))
        mask += [0] * (max_length - len(mask))
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


class DeterministicModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, **kwargs):
        del kwargs
        values = (input_ids * attention_mask).float()
        logits = torch.stack(
            (values.sum(dim=1), values.mean(dim=1), values.max(dim=1).values), dim=1
        )
        return SimpleNamespace(logits=logits)


SOURCES = [
    {"premise": "A man sees a dog.", "hypothesis": "A dog sees a man.", "label": 0},
    {"premise": "A woman holds a camera.", "hypothesis": "A woman has a camera.", "label": 1},
    {"premise": "A cat is near a car.", "hypothesis": "A bird is near a car.", "label": 2},
]


class PipelineInvariantTests(unittest.TestCase):
    def test_1_every_known_marker_is_removed_by_oracle(self):
        for marker in KNOWN_MARKERS:
            with self.subTest(marker=marker):
                clean = "A dog runs quickly."
                noisy = f"{marker} {clean} {marker}"
                self.assertEqual(remove_markers(noisy), clean)
                report = audit_marker_removal(noisy, clean)
                self.assertEqual(report["marker_removal_recall"], 1.0)
                self.assertTrue(report["exact_reconstruction"])

    def test_2_preprocessed_noisy_text_equals_clean_text(self):
        clean = "A man sees a dog."
        for marker in KNOWN_MARKERS:
            reconstructed = remove_markers(f"  {marker}\t{clean}\n")
            self.assertEqual(
                canonicalize_whitespace(reconstructed), canonicalize_whitespace(clean)
            )
        emoji = transform_example(
            {"premise": clean, "hypothesis": clean}, "emoji", seed=4
        )
        self.assertEqual(normalize_emoji(emoji["premise"]), clean)
        self.assertEqual(normalize_emoji(emoji["hypothesis"]), clean)

    def test_3_preprocessed_token_ids_equal_clean_token_ids(self):
        noisy = [transform_example(row, "combined", seed=20 + index,
                                   noise_prob=1.0, include_metadata=True)
                 for index, row in enumerate(SOURCES)]
        report = audit_pipeline_identity(
            SOURCES, noisy, tokenizer=DeterministicTokenizer(), max_length=32,
            reconstruct=reconstruct_with_preprocessing("combined"),
        )
        self.assertEqual(report["stages"]["input_ids_equal"]["rate"], 1.0)
        self.assertEqual(report["stages"]["attention_mask_equal"]["rate"], 1.0)

    def test_4_clean_and_reconstructed_logits_match(self):
        noisy = [transform_example(row, "combined", seed=30 + index,
                                   noise_prob=1.0, include_metadata=True)
                 for index, row in enumerate(SOURCES)]
        report = audit_pipeline_identity(
            SOURCES,
            noisy,
            tokenizer=DeterministicTokenizer(),
            model=DeterministicModel(),
            reconstruct=reconstruct_with_preprocessing("combined"),
            max_length=32,
            logit_atol=0.0,
            raise_on_failure=True,
        )
        for stage in ("text_equal", "input_ids_equal", "attention_mask_equal",
                      "logits_equal"):
            self.assertEqual(report["stages"][stage]["rate"], 1.0)

    def test_model_audit_fails_hard_on_actual_preprocessing_mismatch(self):
        clean = [{"premise": "A dog runs.", "hypothesis": "A dog moves.", "label": 0}]
        noisy = [{"premise": "A dog runs. lowkey",
                  "hypothesis": "A dog moves.", "label": 0}]
        with self.assertRaisesRegex(
            AssertionError, "example 0, stage text"
        ):
            audit_pipeline_identity(
                clean,
                noisy,
                tokenizer=DeterministicTokenizer(),
                model=DeterministicModel(),
                reconstruct=reconstruct_with_preprocessing("noise"),
                max_length=32,
                raise_on_failure=True,
            )

    def test_5_conditions_receive_exactly_declared_transforms(self):
        source = {
            "premise": "A dog is going to see a man with a camera.",
            "hypothesis": "A dog sees a man.",
            "label": 0,
        }
        for mode in ("slang", "emoji", "noise", "combined"):
            with self.subTest(mode=mode):
                transformed = transform_example(
                    source, mode, seed=8, noise_prob=1.0, include_metadata=True
                )
                validate_transform_condition(
                    transformed, mode, require_each_declared=True, training=True
                )

    def test_6_held_out_markers_never_enter_augmentation_training(self):
        for seed in range(100):
            transformed = transform_example(
                SOURCES[0], "noise", seed=seed, noise_prob=1.0,
                marker_pool="train", include_metadata=True,
            )
            validate_transform_condition(transformed, "noise", training=True)
            inserted = {
                event["replacement"].strip()
                for events in transformed["transform_metadata"]["events"].values()
                for event in events
            }
            self.assertTrue(inserted <= set(TRAIN_MARKERS))
            self.assertFalse(inserted & set(HELD_OUT_MARKERS))

    def test_training_condition_is_marker_only_and_compute_matched(self):
        outputs = [
            apply_training_condition(
                source,
                augmentation_probability=1.0,
                marker_placements=("hypothesis_suffix", "premise_suffix"),
                seed=index,
                include_metadata=True,
            )
            for index, source in enumerate(SOURCES)
        ]
        self.assertEqual(len(outputs), len(SOURCES))
        for output in outputs:
            validate_transform_condition(
                output, "noise", require_each_declared=True, training=True
            )

    def test_7_every_retained_evaluation_example_differs(self):
        for mode in ("emoji", "noise", "combined"):
            records, audit = generate_audited_records(
                SOURCES, mode, dataset_name="fixture", source_split="validation", seed=11
            )
            validate_principal_audit(audit)
            self.assertGreater(len(records), 0)
            for transformed in records:
                source = SOURCES[transformed["source_index"]]
                self.assertTrue(any(
                    canonicalize_whitespace(transformed[field])
                    != canonicalize_whitespace(source[field])
                    for field in ("premise", "hypothesis")
                ))


class DatasetAuditTests(unittest.TestCase):
    def test_robustness_drop_uses_condition_matched_clean_accuracy(self):
        results = {
            "original": {"accuracy": 0.99},
            "clean_emoji": {"accuracy": 0.80},
            "emoji": {"accuracy": 0.70},
        }
        drops = robustness_drops(results, variants=("emoji",))
        self.assertAlmostEqual(drops["emoji"], 10.0)

    def test_audit_has_required_counts_distributions_and_checksum(self):
        records, audit = generate_audited_records(
            SOURCES, "combined", dataset_name="fixture", source_split="validation", seed=5
        )
        self.assertEqual(audit["examples"]["attempted"], len(SOURCES))
        self.assertEqual(
            audit["examples"]["changed"] + audit["examples"]["unchanged"],
            len(SOURCES),
        )
        self.assertEqual(audit["inverse_reconstruction"]["rate"], 1.0)
        self.assertTrue(audit["replacement_counts"])
        self.assertTrue(audit["replacement_positions"])
        self.assertTrue(audit["marker_frequencies"])
        self.assertEqual(audit["marker_oracle"]["marker_removal_recall"], 1.0)
        self.assertEqual(audit["marker_oracle"]["exact_reconstruction_rate"], 1.0)
        self.assertEqual(audit["marker_train_test_overlap"], [])
        self.assertIn("fixture", audit["transformation_coverage_by_dataset_and_label"])
        self.assertEqual(audit["checksum_sha256"], dataset_checksum(records))

    def test_paired_clean_records_preserve_order_and_source_ids(self):
        records, _ = generate_audited_records(
            SOURCES, "emoji", dataset_name="fixture", source_split="validation"
        )
        clean = paired_clean_records(SOURCES, records)
        self.assertEqual(
            [row["source_index"] for row in clean],
            [row["source_index"] for row in records],
        )
        for clean_row, transformed in zip(clean, records):
            self.assertEqual(clean_row["label"], transformed["label"])

    def test_generation_uses_frozen_marker_probability_and_placement(self):
        config = {
            "emoji_probability": 1.0,
            "marker_probability": 1.0,
            "max_emoji_replacements": 2,
            "marker_placements": ["hypothesis_prefix"],
        }
        records, audit = generate_audited_records(
            SOURCES, "noise", dataset_name="fixture", source_split="validation",
            transformation_config=config,
        )
        self.assertEqual(
            audit["resolved_transformation_settings"]["marker_probability"], 1.0
        )
        self.assertEqual(
            audit["resolved_transformation_settings"]["marker_pool"], "held_out"
        )
        self.assertTrue(set(audit["marker_frequencies"]) <= set(HELD_OUT_MARKERS))
        self.assertFalse(set(audit["marker_frequencies"]) & set(TRAIN_MARKERS))
        for record in records:
            events = [
                event
                for field_events in record["transform_metadata"]["events"].values()
                for event in field_events
            ]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["placement"], "prefix")

    def test_audit_counts_unchanged_examples_without_retaining_them(self):
        source = [{"premise": "Nothing lexical matches.",
                   "hypothesis": "Still nothing matches.", "label": 0}]
        records, audit = generate_audited_records(
            source, "emoji", dataset_name="fixture", source_split="validation"
        )
        self.assertEqual(records, [])
        self.assertEqual(audit["examples"]["unchanged"], 1)
        with self.assertRaisesRegex(ValueError, "retained no transformed examples"):
            validate_principal_audit(audit)

    def test_atomic_output_removes_failed_staging_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "eval-output"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with atomic_output_directory(target) as staging:
                    (staging / "partial.txt").write_text("partial")
                    raise RuntimeError("injected failure")
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(parent).glob(".eval-output.staging-*")), [])

    def test_atomic_output_refuses_to_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "eval-output"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                with atomic_output_directory(target):
                    pass
            self.assertEqual(marker.read_text(), "preserve")


class RecordingTokenizer:
    def __call__(self, premise, hypothesis, **kwargs):
        del kwargs
        return {"input_ids": [premise, hypothesis], "attention_mask": [1, 1]}


class ProductionPathTests(unittest.TestCase):
    def test_all_smoke_mode_skips_inferential_statistics(self):
        experiment = {
            "training": {
                "warmup_ratio": 0.1, "epochs": 1, "max_length": 32,
                "checkpoint_selection": "fixture development metric",
            },
            "dataset_protocol": {"snli": {}},
        }
        cfg = {
            "model_name": "fixture", "batch_size": 2, "grad_accum": 1,
            "lr": 1e-4, "prefix": "fixture",
        }
        with patch.object(run_all_v2, "step_baseline"), \
             patch.object(run_all_v2, "step_augmentation"), \
             patch.object(run_all_v2, "step_preprocessing"), \
             patch.object(run_all_v2, "step_hybrid"), \
             patch.object(run_all_v2, "step_clean_control"), \
             patch.object(run_all_v2, "step_stats") as stats, \
             patch.object(run_all_v2, "validate_smoke_matrix"):
            run_all_v2.run_step(
                "all", cfg, experiment, Path("config.json"), Path("/tmp/out"),
                "snli", 42, True, None, "test", "fold_1",
                ["slang", "emoji", "noise", "combined"], "final",
            )
        stats.assert_not_called()

    def test_role_and_split_scoping_prevents_prediction_collisions(self):
        matched = scoped_run_dir(
            "/runs", "electra", "baseline", "final", "validation_matched"
        )
        mismatched = scoped_run_dir(
            "/runs", "electra", "baseline", "final", "validation_mismatched"
        )
        development = scoped_run_dir(
            "/runs", "electra", "baseline", "development", "train_holdout"
        )
        self.assertEqual(len({matched, mismatched, development}), 3)

    def test_holm_adjustment_controls_one_primary_family(self):
        entries = [
            ("a", "v1", {"pvalue": 0.01}),
            ("a", "v2", {"pvalue": 0.03}),
            ("b", "v1", {"pvalue": 0.04}),
        ]
        _holm_adjust(entries)
        adjusted = [entry[2]["holm_adjusted_pvalue"] for entry in entries]
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])
        self.assertEqual(
            [entry[2]["significant_holm"] for entry in entries],
            [True, False, False],
        )

    def test_primary_pair_alignment_uses_source_ids(self):
        full = {
            "predictions": [0, 1, 2], "labels": [0, 1, 2],
            "source_indices": [10, 20, 30],
        }
        subset = {
            "predictions": [2, 0], "labels": [2, 0],
            "source_indices": [30, 10],
        }
        a, b, labels = _align_to_reference(full, subset)
        self.assertEqual((a, b, labels), ([2, 0], [2, 0], [2, 0]))

    def test_statistics_separate_four_primary_from_exploratory_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = {
                "predictions": [0, 1, 2, 0], "labels": [0, 1, 2, 1],
                "source_indices": [10, 20, 30, 40],
            }
            predictions = {
                variant: dict(record)
                for variant in (
                    "original", "emoji",
                    "marker_unseen_fold_1_hypothesis_suffix",
                    "emoji_marker_combined_fold_1",
                )
            }
            identity = {
                "config_sha256": "cfg", "generation_seed": 42,
                "dataset": "snli", "split_role": "final", "split_name": "test",
                "source_checksum_sha256": "source",
                "evaluated_variants": [
                    "emoji", "emoji_marker_combined_fold_1",
                    "marker_unseen_fold_1_hypothesis_suffix",
                ],
                "variant_checksums": {},
            }
            approaches = (
                "baseline", "augmented_fold_1", "clean_control", "preprocessing",
                "marker_oracle", "hybrid_fold_1",
            )
            for approach in approaches:
                directory = root / f"electra_{approach}_final_test"
                directory.mkdir()
                (directory / "predictions.json").write_text(json.dumps(predictions))
                (directory / "predictions_manifest.json").write_text(json.dumps(identity))
            report = run_tests(root, "electra", "fold_1", "final", "test")
            self.assertEqual(report["primary"]["n_tests"], 4)
            self.assertEqual(report["exploratory"]["n_tests"], 60)
            self.assertEqual(
                report["primary"]["tests"]["H1_emoji_vs_clean_baseline"]["pvalue"],
                1.0,
            )

    def test_statistics_accepts_canonical_fold_one_variant_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = {
                "predictions": [0, 1, 2, 0], "labels": [0, 1, 2, 1],
                "source_indices": [10, 20, 30, 40],
            }
            evaluated = [
                "emoji_raw", "emoji_marker_combined",
                "marker_unseen_fold_1_hypothesis_suffix",
            ]
            predictions = {
                variant: dict(record) for variant in ("original", *evaluated)
            }
            identity = {
                "config_sha256": "cfg", "generation_seed": 42,
                "dataset": "snli", "split_role": "final", "split_name": "test",
                "source_checksum_sha256": "source",
                "evaluated_variants": evaluated, "variant_checksums": {},
            }
            for approach in (
                "baseline", "augmented_fold_1", "clean_control", "preprocessing",
                "marker_oracle", "hybrid_fold_1",
            ):
                directory = root / f"electra_{approach}_final_test"
                directory.mkdir()
                (directory / "predictions.json").write_text(json.dumps(predictions))
                (directory / "predictions_manifest.json").write_text(
                    json.dumps(identity)
                )

            report = run_tests(root, "electra", "fold_1", "final", "test")
            primary = report["primary"]["tests"]
            self.assertEqual(
                primary["H1_emoji_vs_clean_baseline"]["variant_b"], "emoji_raw"
            )
            self.assertEqual(
                primary["H4_hybrid_vs_baseline_combined"]["variant_a"],
                "emoji_marker_combined",
            )

    def test_on_the_fly_dataset_has_one_fresh_presentation_per_example_epoch(self):
        dataset = OnTheFlyMarkerDataset(
            SOURCES,
            RecordingTokenizer(),
            32,
            augmentation_probability=1.0,
            marker_placements=("hypothesis_suffix", "premise_suffix"),
            seed=17,
        )
        rendered = []
        for epoch in range(2):
            dataset.set_epoch(epoch)
            rendered.append([dataset[index]["input_ids"] for index in range(len(dataset))])
        audit = dataset.audit(completed_epochs=2)
        validate_training_audit(audit)
        self.assertEqual(audit["unique_presentations"], len(SOURCES) * 2)
        self.assertEqual(audit["duplicate_presentations"], 0)
        self.assertEqual(audit["transformed_presentations"], len(SOURCES) * 2)
        self.assertEqual([row["premise"] for row in SOURCES], [
            "A man sees a dog.", "A woman holds a camera.", "A cat is near a car."
        ])
        for epoch_rows in rendered:
            for fields in epoch_rows:
                text = " ".join(fields)
                self.assertFalse(any(marker in text for marker in HELD_OUT_MARKERS))

if __name__ == "__main__":
    unittest.main()
