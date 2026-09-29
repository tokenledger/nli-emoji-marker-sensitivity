"""TRAIN_SIZE: deterministic label-stratified training subsets."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import datasets

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from helpers import (parse_train_size, recorded_training_subset,  # noqa: E402
                     select_training_subset)
from on_the_fly_training import OnTheFlyMarkerDataset  # noqa: E402
import run_all_v2  # noqa: E402


def training_rows(counts=(500, 300, 200)):
    labels = [label for label, count in enumerate(counts) for _ in range(count)]
    order = sorted(range(len(labels)), key=lambda index: (index * 7919) % len(labels))
    return datasets.Dataset.from_dict({
        "premise": [f"Premise {index}." for index in order],
        "hypothesis": [f"Hypothesis {index}." for index in order],
        "label": [labels[index] for index in order],
    })


class ParseTrainSizeTests(unittest.TestCase):
    def test_full_and_integers_are_accepted(self):
        self.assertEqual(parse_train_size("full"), "full")
        self.assertEqual(parse_train_size("200"), 200)
        self.assertEqual(parse_train_size(200), 200)

    def test_invalid_values_are_rejected(self):
        for value in ("0", "-5", "half", "1.5", None):
            with self.assertRaises(ValueError):
                parse_train_size(value)


class SelectTrainingSubsetTests(unittest.TestCase):
    def test_full_returns_every_row_and_is_not_marked_as_subset(self):
        rows = training_rows()
        selected, record = select_training_subset(rows, "full", seed=42)
        self.assertEqual(len(selected), len(rows))
        self.assertEqual(record["train_size"], "full")
        self.assertFalse(record["is_subset"])
        self.assertEqual(record["selected_examples"], 1000)
        self.assertEqual(record["label_counts"], {"0": 500, "1": 300, "2": 200})

    def test_subset_has_requested_size_and_proportional_labels(self):
        selected, record = select_training_subset(training_rows(), 200, seed=42)
        self.assertEqual(len(selected), 200)
        self.assertEqual(Counter(selected["label"]), {0: 100, 1: 60, 2: 40})
        self.assertTrue(record["is_subset"])
        self.assertEqual(record["train_size"], 200)
        self.assertEqual(record["source_examples"], 1000)
        self.assertEqual(record["selected_examples"], 200)
        self.assertEqual(record["selection_seed"], 42)
        self.assertEqual(record["label_counts"], {"0": 100, "1": 60, "2": 40})

    def test_remainders_are_allocated_so_that_the_size_is_exact(self):
        selected, record = select_training_subset(training_rows((334, 333, 333)), 100, seed=1)
        self.assertEqual(len(selected), 100)
        self.assertEqual(sum(record["label_counts"].values()), 100)
        self.assertLessEqual(
            max(record["label_counts"].values()) - min(record["label_counts"].values()), 1
        )

    def test_subset_is_deterministic_for_a_seed_and_differs_between_seeds(self):
        rows = training_rows()
        first, first_record = select_training_subset(rows, 200, seed=42)
        again, again_record = select_training_subset(rows, 200, seed=42)
        other, other_record = select_training_subset(rows, 200, seed=43)
        self.assertEqual(first["premise"], again["premise"])
        self.assertEqual(first_record, again_record)
        self.assertNotEqual(first["premise"], other["premise"])
        self.assertNotEqual(
            first_record["source_indices_sha256"], other_record["source_indices_sha256"]
        )

    def test_subset_keeps_source_order_and_has_no_duplicates(self):
        rows = training_rows()
        selected, _ = select_training_subset(rows, 200, seed=42)
        positions = {premise: index for index, premise in enumerate(rows["premise"])}
        chosen = [positions[premise] for premise in selected["premise"]]
        self.assertEqual(chosen, sorted(chosen))
        self.assertEqual(len(set(chosen)), 200)

    def test_size_equal_to_the_training_set_selects_everything(self):
        rows = training_rows()
        selected, record = select_training_subset(rows, 1000, seed=42)
        self.assertEqual(selected["premise"], rows["premise"])
        self.assertTrue(record["is_subset"])

    def test_size_above_the_training_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            select_training_subset(training_rows(), 1001, seed=42)


class RecordingTokenizer:
    def __call__(self, premise, hypothesis, **kwargs):
        del kwargs
        return {"input_ids": [premise, hypothesis], "attention_mask": [1, 1]}


class SubsetBeforeAugmentationTests(unittest.TestCase):
    def test_subset_rows_are_clean_and_augmentation_sees_only_the_subset(self):
        rows = training_rows()
        selected, _ = select_training_subset(rows, 200, seed=42)
        self.assertTrue(set(selected["hypothesis"]) <= set(rows["hypothesis"]))
        augmented = OnTheFlyMarkerDataset(
            selected, RecordingTokenizer(), 128, augmentation_probability=0.5,
            marker_placements=("hypothesis_suffix",), seed=42,
            marker_choices=("fr", "tbh", "ngl"),
        )
        self.assertEqual(len(augmented), 200)
        for index in range(len(augmented)):
            premise, hypothesis = augmented[index]["input_ids"]
            self.assertEqual(premise, selected[index]["premise"])
            self.assertTrue(hypothesis.startswith(selected[index]["hypothesis"]))
        audit = augmented.audit(1)
        self.assertEqual(audit["dataset_examples"], 200)
        self.assertGreater(audit["transformed_presentations"], 0)
        self.assertLess(audit["transformed_presentations"], 200)


class RecordedSubsetTests(unittest.TestCase):
    def test_record_is_read_back_from_a_trained_run(self):
        _, record = select_training_subset(training_rows(), 200, seed=42)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.assertIsNone(recorded_training_subset(run_dir))
            (run_dir / "results.json").write_text(
                json.dumps({"training_subset": record}), encoding="utf-8"
            )
            self.assertEqual(recorded_training_subset(run_dir), record)


class OrchestrationTests(unittest.TestCase):
    def common_flags(self, train_size):
        return run_all_v2.build_common(
            {"model_name": "model", "batch_size": 32, "grad_accum": 1, "lr": 2e-5},
            {"warmup_ratio": 0.06, "epochs": 3, "max_length": 128,
             "checkpoint_selection": "highest development accuracy"},
            "snli", 42, False, None, train_size=train_size,
        )

    def test_train_size_is_passed_to_the_training_scripts(self):
        flags = self.common_flags(200)
        self.assertEqual(flags[flags.index("--train_size") + 1], "200")
        flags = self.common_flags("full")
        self.assertEqual(flags[flags.index("--train_size") + 1], "full")

    def test_training_scripts_accept_the_option(self):
        for name in ("train_baseline", "train_augmentation", "train_hybrid"):
            source = (SOURCE_ROOT / f"{name}.py").read_text(encoding="utf-8")
            self.assertIn("'--train_size', type=parse_train_size, default='full'", source)
            self.assertLess(
                source.index("select_training_subset(\n"),
                source.index("Tokenizing training data")
                if name == "train_baseline"
                else source.index("prepare_epoch_tokenized_dataset(\n"),
            )
            self.assertIn("'training_subset': training_subset,", source)

    def test_subset_and_full_runs_have_different_manifest_identities(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--train_size", type=parse_train_size, default="full")
        self.assertEqual(parser.parse_args([]).train_size, "full")
        self.assertEqual(parser.parse_args(["--train_size", "200"]).train_size, 200)
        with tempfile.TemporaryDirectory() as directory:
            config = run_all_v2.load_experiment_config(run_all_v2.DEFAULT_CONFIG_PATH)
            cfg = dict(run_all_v2.model_config(config, "electra"), train_size=200)
            output = run_all_v2.materialize_manifest(
                directory, "baseline", cfg, config, run_all_v2.DEFAULT_CONFIG_PATH,
                "snli", 42, "test", system_id="clean_baseline",
            )
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runtime_overrides"]["train_size"], 200)
            with self.assertRaisesRegex(RuntimeError, "Refusing to reuse"):
                run_all_v2.materialize_manifest(
                    directory, "baseline", dict(cfg, train_size="full"), config,
                    run_all_v2.DEFAULT_CONFIG_PATH, "snli", 42, "test",
                    system_id="clean_baseline",
                )


if __name__ == "__main__":
    unittest.main()
