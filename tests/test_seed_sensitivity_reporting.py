"""Tests for the frozen additional-seed reporting layer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from seed_sensitivity_reporting import (  # noqa: E402
    build_additional_runtime_summary,
    build_cell_seed_summary,
    build_clean_performance_summary,
    build_electra_clean_noninferiority,
    build_seed42_comparison,
    validate_compact_inputs,
    write_analysis,
)


def effect(seed: int, value_pp: float, *, model: str = "electra") -> dict:
    reference = 0.8
    return {
        "model": model,
        "dataset": "snli",
        "split": "test",
        "fold": "shared",
        "hypothesis": "H1",
        "seed": seed,
        "treatment_accuracy": reference + value_pp / 100,
        "reference_accuracy": reference,
        "effect_pp": value_pp,
        "significant_holm": True if model == "electra" else None,
    }


class SeedSensitivitySummaryTests(unittest.TestCase):
    def test_cell_summary_uses_training_seed_interval_and_direction(self):
        rows = [effect(seed, value) for seed, value in (
            (13, -2.0), (21, -2.2), (42, -1.9), (87, -2.1), (101, -2.0)
        )]
        summary = build_cell_seed_summary(rows, confidence=0.95)[0]
        self.assertEqual(summary["n_seeds"], 5)
        self.assertTrue(summary["all_seeds_expected_direction"])
        self.assertTrue(summary["ci95_supports_expected_direction"])
        self.assertEqual(summary["example_holm_significant_count"], 5)

    def test_seed42_is_compared_only_with_new_seeds(self):
        rows = [
            effect(13, -2.0),
            effect(21, -4.0),
            effect(42, -3.0),
            effect(87, -2.0),
            effect(101, -4.0),
        ]
        comparison = build_seed42_comparison(rows)[0]
        self.assertEqual(comparison["new_seed_mean_effect_pp"], -3.0)
        self.assertEqual(comparison["seed42_minus_new_mean_pp"], 0.0)
        self.assertTrue(comparison["seed42_within_new_seed_range"])

    def test_partial_profile_is_labeled_directional(self):
        rows = [
            effect(13, -1.0, model="bertweet"),
            effect(42, -1.5, model="bertweet"),
            effect(87, -2.0, model="bertweet"),
        ]
        summary = build_cell_seed_summary(rows, confidence=0.95)[0]
        self.assertEqual(summary["evidence_scope"], "partial_three_seed_fold1")
        self.assertEqual(summary["example_holm_tests_available"], 0)

    def test_clean_summary_is_paired_by_seed(self):
        rows = []
        for seed, baseline, hybrid in (
            (13, 0.80, 0.81),
            (42, 0.82, 0.83),
            (87, 0.79, 0.80),
        ):
            for system, accuracy in (
                ("baseline", baseline),
                ("augmented", baseline + 0.005),
                ("hybrid", hybrid),
            ):
                rows.append({
                    "model": "roberta",
                    "dataset": "snli",
                    "split": "test",
                    "fold": "fold_1",
                    "seed": seed,
                    "system": system,
                    "accuracy": accuracy,
                })
        summary = build_clean_performance_summary(rows, confidence=0.95)
        hybrid = next(row for row in summary if row["system"] == "hybrid")
        self.assertAlmostEqual(hybrid["mean_difference_pp"], 1.0)

    def test_registered_noninferiority_uses_one_sided_lower_bound(self):
        rows = []
        for seed, difference in (
            (13, 0.001),
            (21, 0.002),
            (42, 0.001),
            (87, 0.002),
            (101, 0.001),
        ):
            for system, accuracy in (
                ("baseline", 0.8),
                ("hybrid", 0.8 + difference),
            ):
                rows.append({
                    "model": "electra",
                    "dataset": "snli",
                    "split": "test",
                    "fold": "fold_1",
                    "seed": seed,
                    "system": system,
                    "accuracy": accuracy,
                })
        report = build_electra_clean_noninferiority(
            rows, margin_pp=0.5, confidence=0.95
        )[0]
        self.assertTrue(report["noninferior"])
        self.assertGreater(
            report["lower_one_sided_bound_percentage_points"], -0.5
        )

    def test_runtime_summary_is_labeled_as_observed_command_time(self):
        rows = [
            {
                "source_group": "additional_seed",
                "profile": "electra_full",
                "model": "electra",
                "runtime_seconds": 600,
            },
            {
                "source_group": "additional_seed",
                "profile": "electra_full",
                "model": "electra",
                "runtime_seconds": 1200,
            },
            {
                "source_group": "additional_seed",
                "profile": "electra_full",
                "model": "electra",
                "runtime_seconds": None,
            },
            {
                "source_group": "seed42_reference",
                "profile": "electra_full",
                "model": "electra",
                "runtime_seconds": 9999,
            },
        ]
        summary = build_additional_runtime_summary(rows)[0]
        self.assertEqual(summary["result_producing_calls"], 2)
        self.assertAlmostEqual(summary["observed_command_hours"], 0.5)
        self.assertAlmostEqual(summary["median_minutes_per_call"], 15)

    def test_compact_validation_rejects_inconsistent_effect(self):
        bad = effect(13, -2.0)
        bad["effect_pp"] = 99
        payload = {
            "schema_version": 1,
            "effects": [bad],
            "clean_accuracies": [{
                "model": "electra",
                "dataset": "snli",
                "split": "test",
                "fold": "fold_1",
                "seed": 13,
                "system": "baseline",
                "accuracy": 0.8,
            }],
            "artifacts": [{"remote_dir": "x"}],
        }
        with self.assertRaisesRegex(ValueError, "paired accuracies"):
            validate_compact_inputs(payload)

    def test_compact_validation_rejects_mixed_data_releases(self):
        payload = {
            "schema_version": 1,
            "effects": [effect(13, -2.0)],
            "clean_accuracies": [{
                "model": "electra",
                "dataset": "snli",
                "split": "test",
                "fold": "fold_1",
                "seed": 13,
                "system": "baseline",
                "accuracy": 0.8,
            }],
            "artifacts": [
                {
                    "remote_dir": "first",
                    "result_sha256": "a",
                    "config_sha256": "config",
                    "data_release_id": "release-a",
                    "source_manifest_sha256": "source",
                    "runtime_seconds": 1,
                },
                {
                    "remote_dir": "second",
                    "result_sha256": "b",
                    "config_sha256": "config",
                    "data_release_id": "release-b",
                    "source_manifest_sha256": "source",
                    "runtime_seconds": 1,
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "data_release_id"):
            validate_compact_inputs(payload)

    def test_writer_creates_hash_addressed_report_bundle(self):
        effects = [
            effect(seed, -2.0) for seed in (13, 21, 42, 87, 101)
        ]
        clean = []
        for seed in (13, 21, 42, 87, 101):
            for system, accuracy in (
                ("baseline", 0.8),
                ("augmented", 0.801),
                ("hybrid", 0.802),
                ("clean_control", 0.8),
            ):
                clean.append({
                    "model": "electra",
                    "dataset": "snli",
                    "split": "test",
                    "fold": "fold_1",
                    "seed": seed,
                    "system": system,
                    "accuracy": accuracy,
                })
        payload = {
            "schema_version": 1,
            "analysis_contract": {
                "confidence_level": 0.95,
                "clean_noninferiority_margin_percentage_points": 0.5,
            },
            "provenance": {"modal_volume": "fixture"},
            "effects": effects,
            "clean_accuracies": clean,
            "artifacts": [{"remote_dir": "fixture"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            summary = write_analysis(payload, output)
            self.assertEqual(summary["noninferiority_cells"], 1)
            self.assertTrue((output / "REPORT.md").is_file())
            manifest = json.loads(
                (output / "artifact_manifest.json").read_text()
            )
            self.assertTrue(manifest["files"])
            self.assertTrue((output / "seed_effects.csv").is_file())


if __name__ == "__main__":
    unittest.main()
