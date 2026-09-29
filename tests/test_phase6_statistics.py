"""Phase 6 seed-aware statistics acceptance tests."""

import sys
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aggregate_statistics import (  # noqa: E402
    APPROACH_DIRS, _paired_accuracy_difference, _resolve_primary_variant, latex_table,
    noninferiority_test, paired_example_interval, paired_seed_summary, seed_summary,
)
from statistical_tests import bootstrap_ci  # noqa: E402


class SeedStatisticsTests(unittest.TestCase):
    def test_all_six_production_systems_are_aggregated(self):
        self.assertEqual(
            set(APPROACH_DIRS),
            {'baseline', 'augmented', 'clean_control', 'preprocessing',
             'marker_oracle', 'hybrid'},
        )

    def test_bootstrap_report_uses_sample_accuracy_as_point_estimate(self):
        report = bootstrap_ci([0, 0, 1], [0, 1, 1], n_bootstrap=25)
        self.assertEqual(report['point_estimate'], 2 / 3)
        self.assertEqual(report['mean'], report['point_estimate'])
        self.assertIn('bootstrap_mean', report)

    def test_aggregate_resolves_canonical_and_compatibility_variant_names(self):
        self.assertEqual(
            _resolve_primary_variant(
                ["original", "emoji_raw"], ("emoji_raw", "emoji"), "emoji"
            ),
            "emoji_raw",
        )
        self.assertEqual(
            _resolve_primary_variant(
                ["original", "emoji"], ("emoji_raw", "emoji"), "emoji"
            ),
            "emoji",
        )

    def test_seed_summary_preserves_contributing_seeds_and_run_variance(self):
        report = seed_summary({42: 0.8, 13: 0.7, 87: 0.9})
        self.assertEqual(report["seeds"], [13, 42, 87])
        self.assertAlmostEqual(report["mean"], 0.8)
        self.assertEqual(report["uncertainty_scope"], "training_seed_variance")

    def test_paired_seed_sets_must_match(self):
        with self.assertRaisesRegex(ValueError, "different seed sets"):
            paired_seed_summary({1: 0.8, 2: 0.7}, {1: 0.8, 3: 0.7})

    def test_noninferiority_uses_lower_bound_and_frozen_margin(self):
        result = noninferiority_test(
            {1: .801, 2: .802, 3: .803}, {1: .800, 2: .800, 3: .800},
            margin_pp=0.5,
        )
        self.assertTrue(result["noninferior"])
        self.assertEqual(result["decision_rule"], "lower bound > -margin")

    def test_example_interval_is_paired_and_distinct_from_seed_variance(self):
        base = {
            "source_indices": [1, 2, 3], "labels": [0, 1, 0],
            "predictions": [0, 0, 0],
        }
        better = {**base, "predictions": [0, 1, 0]}
        report = paired_example_interval({1: better, 2: better}, {1: base, 2: base},
                                         n_bootstrap=50)
        self.assertTrue(report["training_seed_variance_reported_separately"])
        self.assertEqual(report["examples"], 3)

    def test_primary_accuracy_difference_aligns_subset_by_source_id(self):
        full = {
            "source_indices": [9, 3, 8], "labels": [2, 0, 1],
            "predictions": [2, 0, 0],
        }
        subset = {
            "source_indices": [3, 8], "labels": [0, 1],
            "predictions": [1, 0],
        }
        self.assertEqual(_paired_accuracy_difference(full, subset), 0.5)

    def test_latex_table_is_generated_from_aggregate_schema(self):
        metric = {"mean": .8, "standard_deviation": .01}
        report = {"accuracies": {"baseline": {"original": metric}}}
        table = latex_table(report)
        self.assertIn("80.00", table)
        self.assertIn("\\begin{tabular}", table)


if __name__ == "__main__":
    unittest.main()
