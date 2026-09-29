"""Synthetic-data tests for the baseline-only DeBERTa addendum statistics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from deberta_ext_statistics import (  # noqa: E402
    BOOTSTRAP_SEED,
    FOLDS,
    bootstrap_difference_ci,
    exact_mcnemar,
    family_specification,
    holm_family,
    paired_correctness,
    run_test,
)


def _record(predictions, labels, source_indices):
    return {
        "predictions": list(predictions),
        "labels": list(labels),
        "source_indices": list(source_indices),
    }


def _synthetic_systems(rng: np.random.Generator, n: int = 400):
    """Build baseline and preprocessing payloads covering the whole family."""
    labels = rng.integers(0, 3, n)
    ids = list(range(n))
    clean = labels.copy()
    clean[rng.choice(n, 30, replace=False)] = (clean[rng.choice(n, 30, replace=False)] + 1) % 3

    def degrade(base, k):
        out = base.copy()
        flip = rng.choice(n, k, replace=False)
        out[flip] = (out[flip] + 1) % 3
        return out

    emoji_ids = ids[: n // 2]
    baseline = {"original": _record(clean, labels, ids)}
    baseline["emoji_raw"] = _record(degrade(clean, 40)[: n // 2], labels[: n // 2], emoji_ids)
    baseline["clean_emoji_raw"] = _record(clean[: n // 2], labels[: n // 2], emoji_ids)
    for fold in FOLDS:
        stem = f"marker_unseen_{fold}_hypothesis_suffix"
        baseline[stem] = _record(degrade(clean, 80), labels, ids)
        baseline[f"clean_{stem}"] = _record(clean, labels, ids)
        baseline[f"{stem}_formal_control"] = _record(degrade(clean, 20), labels, ids)
        baseline[f"{stem}_random_control"] = _record(degrade(clean, 25), labels, ids)
    preprocessing = {
        "original": _record(clean, labels, ids),
        "emoji_raw": _record(degrade(clean, 5)[: n // 2], labels[: n // 2], emoji_ids),
    }
    return {"baseline": baseline, "preprocessing": preprocessing}


class FamilyDefinitionTests(unittest.TestCase):
    def test_family_has_exactly_eleven_tests_in_fixed_order(self):
        family = family_specification()
        self.assertEqual(len(family), 11)
        groups = [entry["group"] for entry in family]
        self.assertEqual(groups, ["H1"] + ["H2"] * 3 + ["control"] * 6 + ["normalization"])
        self.assertEqual(len({entry["test_id"] for entry in family}), 11)

    def test_family_pairs_each_condition_with_the_intended_reference(self):
        by_id = {entry["test_id"]: entry for entry in family_specification()}
        h1 = by_id["H1_emoji_raw_vs_paired_clean"]
        self.assertEqual((h1["treatment_variant"], h1["reference_variant"]), ("emoji_raw", "clean_emoji_raw"))
        for fold in FOLDS:
            stem = f"marker_unseen_{fold}_hypothesis_suffix"
            self.assertEqual(by_id[f"H2_{fold}_hsuffix_vs_paired_clean"]["reference_variant"], f"clean_{stem}")
            for control in ("formal", "random"):
                entry = by_id[f"control_{fold}_hsuffix_vs_{control}"]
                self.assertEqual(entry["treatment_variant"], stem)
                self.assertEqual(entry["reference_variant"], f"{stem}_{control}_control")
        norm = by_id["normalization_preprocessing_vs_baseline_emoji_raw"]
        self.assertEqual(norm["treatment_system"], "preprocessing")
        self.assertEqual(norm["reference_system"], "baseline")
        self.assertEqual(norm["treatment_variant"], norm["reference_variant"])

    def test_family_is_baseline_only(self):
        systems = {entry["treatment_system"] for entry in family_specification()}
        systems |= {entry["reference_system"] for entry in family_specification()}
        self.assertEqual(systems, {"baseline", "preprocessing"})


class PairingTests(unittest.TestCase):
    def test_alignment_follows_reference_source_order(self):
        reference = _record([0, 1, 2], [0, 1, 2], [10, 20, 30])
        treatment = _record([2, 1, 0, 9], [2, 1, 0, 1], [30, 20, 10, 40])
        t_correct, r_correct, n = paired_correctness(treatment, reference)
        self.assertEqual(n, 3)
        self.assertEqual(t_correct.tolist(), [True, True, True])
        self.assertEqual(r_correct.tolist(), [True, True, True])

    def test_alignment_rejects_missing_sources_and_label_mismatch(self):
        reference = _record([0, 1], [0, 1], [1, 2])
        with self.assertRaises(ValueError):
            paired_correctness(_record([0], [0], [1]), reference)
        with self.assertRaises(ValueError):
            paired_correctness(_record([0, 1], [0, 2], [1, 2]), reference)

    def test_run_test_requires_identical_source_indices(self):
        preds = {
            "baseline": {
                "emoji_raw": _record([0, 1], [0, 1], [1, 2]),
                "clean_emoji_raw": _record([0, 1, 2], [0, 1, 2], [1, 2, 3]),
            }
        }
        spec = family_specification()[0]
        with self.assertRaises(ValueError):
            run_test(spec, preds)

    def test_run_test_reports_difference_and_discordants(self):
        preds = {
            "baseline": {
                "emoji_raw": _record([0, 1, 2, 0, 1, 1], [0, 1, 2, 1, 1, 0], [0, 1, 2, 3, 4, 5]),
                "clean_emoji_raw": _record([0, 1, 2, 1, 0, 0], [0, 1, 2, 1, 1, 0], [0, 1, 2, 3, 4, 5]),
            }
        }
        result = run_test(family_specification()[0], preds)
        self.assertEqual(result["n_pairs"], 6)
        # treatment correct: T T T F T F (4/6); reference correct: T T T T F T (5/6)
        self.assertEqual(result["n_treatment_only_correct"], 1)
        self.assertEqual(result["n_reference_only_correct"], 2)
        self.assertEqual(result["n_discordant"], 3)
        self.assertAlmostEqual(result["difference_pp"], -100 / 6)
        self.assertEqual(result["pvalue"], 1.0)

    def test_whole_family_runs_on_synthetic_payloads(self):
        preds = _synthetic_systems(np.random.default_rng(7))
        results = [run_test(spec, preds) for spec in family_specification()]
        holm_family(results)
        self.assertEqual(len(results), 11)
        for result in results:
            self.assertIn("holm_adjusted_pvalue", result)
            self.assertGreaterEqual(result["holm_adjusted_pvalue"], result["pvalue"])
            self.assertLessEqual(result["ci_lower_pp"], result["difference_pp"])
            self.assertGreaterEqual(result["ci_upper_pp"], result["difference_pp"])
        by_id = {r["test_id"]: r for r in results}
        self.assertLess(by_id["H1_emoji_raw_vs_paired_clean"]["difference_pp"], 0)
        self.assertGreater(by_id["normalization_preprocessing_vs_baseline_emoji_raw"]["difference_pp"], 0)


class ExactMcNemarTests(unittest.TestCase):
    def test_zero_discordant_gives_p_one(self):
        result = exact_mcnemar(np.array([True, False]), np.array([True, False]))
        self.assertEqual(result["pvalue"], 1.0)
        self.assertEqual(result["n_discordant"], 0)

    def test_one_sided_discordance_matches_binomial(self):
        treatment = np.array([False] * 10 + [True] * 5)
        reference = np.array([True] * 10 + [True] * 5)
        result = exact_mcnemar(treatment, reference)
        self.assertEqual(result["n_reference_only_correct"], 10)
        self.assertEqual(result["n_treatment_only_correct"], 0)
        self.assertAlmostEqual(result["pvalue"], 2 * 0.5**10)

    def test_symmetric_in_direction(self):
        a = np.array([True, False, False, True, False])
        b = np.array([False, True, True, True, True])
        self.assertAlmostEqual(exact_mcnemar(a, b)["pvalue"], exact_mcnemar(b, a)["pvalue"])


class HolmTests(unittest.TestCase):
    def test_holm_adjustment_matches_hand_computation(self):
        results = [
            {"test_id": "a", "pvalue": 0.01},
            {"test_id": "b", "pvalue": 0.04},
            {"test_id": "c", "pvalue": 0.03},
            {"test_id": "d", "pvalue": 0.50},
        ]
        holm_family(results)
        by_id = {r["test_id"]: r for r in results}
        self.assertAlmostEqual(by_id["a"]["holm_adjusted_pvalue"], 0.04)
        self.assertAlmostEqual(by_id["c"]["holm_adjusted_pvalue"], 0.09)
        self.assertAlmostEqual(by_id["b"]["holm_adjusted_pvalue"], 0.09)
        self.assertAlmostEqual(by_id["d"]["holm_adjusted_pvalue"], 0.50)
        self.assertTrue(by_id["a"]["significant_holm"])
        self.assertFalse(by_id["b"]["significant_holm"])

    def test_holm_is_monotone_and_capped(self):
        rng = np.random.default_rng(3)
        results = [{"test_id": str(i), "pvalue": float(p)} for i, p in enumerate(rng.uniform(0, 1, 11))]
        holm_family(results)
        ordered = sorted(results, key=lambda r: r["pvalue"])
        adjusted = [r["holm_adjusted_pvalue"] for r in ordered]
        self.assertEqual(adjusted, sorted(adjusted))
        self.assertTrue(all(a <= 1.0 for a in adjusted))
        self.assertAlmostEqual(adjusted[0], min(1.0, 11 * ordered[0]["pvalue"]))


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_brackets_point_estimate(self):
        rng = np.random.default_rng(11)
        treatment = rng.random(500) < 0.85
        reference = rng.random(500) < 0.90
        first = bootstrap_difference_ci(treatment, reference)
        second = bootstrap_difference_ci(treatment, reference)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], BOOTSTRAP_SEED)
        self.assertEqual(first["n_resamples"], 2000)
        self.assertLessEqual(first["ci_lower_pp"], first["difference_pp"])
        self.assertGreaterEqual(first["ci_upper_pp"], first["difference_pp"])
        self.assertTrue(math.isfinite(first["ci_lower_pp"]))

    def test_identical_vectors_give_zero_width_interval(self):
        correct = np.array([True, False, True, True])
        result = bootstrap_difference_ci(correct, correct, n_resamples=50)
        self.assertEqual(result["difference_pp"], 0.0)
        self.assertEqual((result["ci_lower_pp"], result["ci_upper_pp"]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
