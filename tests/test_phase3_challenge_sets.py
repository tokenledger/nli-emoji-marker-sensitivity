"""Acceptance tests for implementation-plan Phase 3."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_config import load_experiment_config  # noqa: E402
from helpers import resolve_eval_base  # noqa: E402
from prepare_eval_sets import (  # noqa: E402
    generate_emoji_challenge_records,
    generate_fold_combined_records,
    generate_marker_challenge_records,
    stratified_holdout,
    validate_principal_audit,
)
from transforms import apply_training_condition  # noqa: E402
from helpers import robustness_drops  # noqa: E402


def marker_fixture(per_label=12):
    rows = []
    for label in range(3):
        for index in range(per_label):
            rows.append({
                "premise": f"A person sees a dog number {index}.",
                "hypothesis": "A woman holds a camera.",
                "label": label,
            })
    return rows


class EmojiConditionTests(unittest.TestCase):
    def test_raw_is_bijective_capped_and_gloss_renders_clean(self):
        rows = [{
            "premise": "A man sees a dog and a cat.",
            "hypothesis": "A woman holds a camera.",
            "label": 0,
        }]
        raw, audit = generate_emoji_challenge_records(
            rows, "emoji_raw", dataset_name="fixture", source_split="development"
        )
        validate_principal_audit(audit)
        self.assertEqual(len(raw), 1)
        self.assertIn(sum(audit["emoji_replacements_per_example"].values()), (1,))
        self.assertLessEqual(
            sum(len(events) for events in raw[0]["transform_metadata"]["events"].values()), 2
        )
        gloss, gloss_audit = generate_emoji_challenge_records(
            rows, "emoji_gloss", dataset_name="fixture", source_split="development"
        )
        self.assertEqual(gloss[0]["premise"], rows[0]["premise"])
        self.assertEqual(gloss[0]["hypothesis"], rows[0]["hypothesis"])
        self.assertTrue(gloss_audit["output_equals_paired_clean"])

    def test_lossy_condition_is_explicit_appendix_stress_test(self):
        rows = [{"premise": "Two men are running.",
                 "hypothesis": "Some people are walking.", "label": 1}]
        records, audit = generate_emoji_challenge_records(
            rows, "emoji_lossy_stress", dataset_name="fixture", source_split="development"
        )
        self.assertTrue(records)
        self.assertTrue(audit["stress_test"])
        self.assertFalse(audit["principal"])


class MarkerConditionTests(unittest.TestCase):
    def setUp(self):
        self.fold = load_experiment_config()["transformations"]["marker_folds"][0]

    def test_unseen_markers_are_balanced_by_label_and_do_not_leak(self):
        records, audit = generate_marker_challenge_records(
            marker_fixture(), dataset_name="fixture", source_split="development",
            fold=self.fold, status="unseen", placement="hypothesis_prefix",
            minimum_per_label=1,
        )
        self.assertEqual(len(records), 36)
        self.assertFalse(set(audit["marker_frequencies"]) & set(self.fold["train"]))
        for marker in self.fold["test"]:
            self.assertEqual(set(audit["marker_coverage_by_label"][marker]), {"0", "1", "2"})

    def test_every_frozen_fold_generates_seen_and_unseen_conditions(self):
        folds = load_experiment_config()["transformations"]["marker_folds"]
        for fold in folds:
            for status in ("seen", "unseen"):
                with self.subTest(fold=fold["id"], status=status):
                    records, audit = generate_marker_challenge_records(
                        marker_fixture(), dataset_name="fixture",
                        source_split="development", fold=fold, status=status,
                        placement="hypothesis_suffix", minimum_per_label=1,
                    )
                    self.assertEqual(len(records), 36)
                    validate_principal_audit(audit)
                    if status == "unseen":
                        self.assertFalse(audit["marker_train_test_overlap"])

    def test_every_fold_has_a_leak_free_fold_specific_combined_condition(self):
        folds = load_experiment_config()["transformations"]["marker_folds"]
        for fold in folds:
            with self.subTest(fold=fold["id"]):
                records, audit = generate_fold_combined_records(
                    marker_fixture(), dataset_name="fixture",
                    source_split="development", fold=fold,
                )
                self.assertTrue(records)
                self.assertFalse(audit["marker_train_test_overlap"])
                self.assertTrue(
                    set(audit["marker_frequencies"]) <= set(fold["test"])
                )

    def test_augmentation_uses_the_selected_fold_training_pool(self):
        fold = load_experiment_config()["transformations"]["marker_folds"][1]
        for seed in range(30):
            row = apply_training_condition(
                marker_fixture(1)[0], augmentation_probability=1.0,
                marker_placements=("hypothesis_suffix",),
                marker_choices=fold["train"], seed=seed, include_metadata=True,
            )
            inserted = {
                event["replacement"].strip()
                for events in row["transform_metadata"]["events"].values()
                for event in events
            }
            self.assertTrue(inserted <= set(fold["train"]))
            self.assertFalse(inserted & set(fold["test"]))

    def test_formal_and_random_controls_match_reference_token_count(self):
        for control in ("formal", "random"):
            records, audit = generate_marker_challenge_records(
                marker_fixture(), dataset_name="fixture", source_split="development",
                fold=self.fold, status="seen", placement="premise_suffix",
                control_type=control, minimum_per_label=1,
            )
            self.assertTrue(records)
            self.assertTrue(audit["token_count_matched"])
            references = audit["reference_marker_by_source"]
            for record in records:
                events = record["transform_metadata"]["events"]["premise"]
                inserted = events[0]["replacement"].strip()
                reference = references[str(record["source_index"])]
                self.assertEqual(len(inserted.split()), len(reference.split()))

    def test_control_collision_is_a_diagnostic_not_generation_failure(self):
        rows = marker_fixture()
        rows[0]["premise"] = "A person is nearby outside in the park."
        records, audit = generate_marker_challenge_records(
            rows, dataset_name="fixture", source_split="development",
            fold=self.fold, status="seen", placement="premise_suffix",
            control_type="random", minimum_per_label=1,
        )
        self.assertEqual(len(records), len(rows))
        self.assertEqual(audit["marker_oracle"]["exact_reconstruction_rate"], 1.0)
        self.assertTrue(audit["global_marker_deletion_diagnostic"]["not_a_principal_invariant"])


class SplitTests(unittest.TestCase):
    def test_custom_variant_drops_are_manifest_driven(self):
        drops = robustness_drops({
            "original": {"accuracy": 1.0},
            "emoji_raw": {"accuracy": 0.75},
            "clean_emoji_raw": {"accuracy": 1.0},
        })
        self.assertEqual(drops, {"emoji_raw": 25.0})

    def test_multinli_holdout_is_deterministic_stratified_and_disjoint(self):
        rows = marker_fixture(per_label=20)
        train_a, dev_a = stratified_holdout(rows, 0.2, 17)
        train_b, dev_b = stratified_holdout(rows, 0.2, 17)
        self.assertEqual(train_a, train_b)
        self.assertEqual(dev_a, dev_b)
        self.assertFalse(
            {row["source_index"] for row in train_a}
            & {row["source_index"] for row in dev_a}
        )
        self.assertEqual(Counter(row["label"] for row in dev_a), Counter({0: 4, 1: 4, 2: 4}))

    def test_final_suite_labels_are_blocked_for_tuning_purpose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "snli").mkdir()
            (root / "dataset_manifest.json").write_text(json.dumps({
                "datasets": {"snli": {
                    "default_final_split": "test",
                    "splits": {"final:test": {"base_path": "snli/final/test"}},
                }}
            }))
            with self.assertRaises(PermissionError):
                resolve_eval_base(
                    root / "snli", split_role="final", purpose="checkpoint_selection"
                )

    def test_direct_final_path_cannot_bypass_label_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / "snli" / "final" / "test"
            (final / "original").mkdir(parents=True)
            (root / "dataset_manifest.json").write_text(json.dumps({
                "schema_version": 2,
                "datasets": {"snli": {
                    "default_final_split": "test",
                    "splits": {"final:test": {
                        "role": "final", "base_path": "snli/final/test"
                    }},
                }}
            }))
            with self.assertRaises(PermissionError):
                resolve_eval_base(
                    final, split_role="final", purpose="checkpoint_selection"
                )


if __name__ == "__main__":
    unittest.main()
