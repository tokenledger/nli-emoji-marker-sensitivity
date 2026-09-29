#!/usr/bin/env python3
"""Focused contract tests for the independent v2 validator."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().with_name("validate_v2.py")
SPEC = importlib.util.spec_from_file_location("independent_v2_validator", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def brute_fixed_margin_tail(population: int, margins: list[int], observed: int) -> float:
    supports = [list(itertools.combinations(range(population), margin)) for margin in margins]
    numerator = 0
    denominator = 0
    for selections in itertools.product(*supports):
        denominator += 1
        overlap = set(selections[0])
        for selection in selections[1:]:
            overlap.intersection_update(selection)
        numerator += len(overlap) >= observed
    return numerator / denominator


class IndependentValidatorTests(unittest.TestCase):
    def test_fixed_margin_tail_matches_enumeration(self) -> None:
        for observed in range(3):
            expected = brute_fixed_margin_tail(6, [2, 3, 2], observed)
            actual = VALIDATOR.fixed_margin_intersection_tail(6, [2, 3, 2], observed)
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))

    def test_mcnemar_counts_and_holm(self) -> None:
        left = np.asarray([1, 1, 1, 0, 0], dtype=bool)
        right = np.asarray([1, 0, 0, 1, 0], dtype=bool)
        p_value, left_only, right_only = VALIDATOR.exact_mcnemar(left, right)
        self.assertEqual((left_only, right_only), (2, 1))
        self.assertEqual(p_value, 1.0)
        np.testing.assert_allclose(
            VALIDATOR.holm_adjust([0.01, 0.03, 0.02, 1.0]),
            [0.04, 0.06, 0.06, 1.0],
        )

    def test_bootstrap_is_key_deterministic(self) -> None:
        values = np.asarray([-100.0, 0.0, 100.0])
        self.assertEqual(
            VALIDATOR.bootstrap_mean_ci(values, "contract-key"),
            VALIDATOR.bootstrap_mean_ci(values, "contract-key"),
        )

    def test_corrected_runtime_contract(self) -> None:
        runtime = {
            "device": "cpu",
            "threads": 4,
            "interop_threads": 1,
            "batch_size": 128,
            "max_length": 128,
            "input_preprocessing": "none",
            "tokenizer_backend": "slow",
            "tokenizer_normalization": "disabled",
            "logit_dtype": "float32",
            "seed": 42,
            "deterministic_algorithms": True,
            "torch_version": "test",
        }
        self.assertTrue(VALIDATOR.corrected_runtime_is_valid(runtime))
        cuda_runtime = {
            **runtime,
            "device": "cuda",
            "accelerator_name": "NVIDIA H100 80GB HBM3",
            "accelerator_count": 1,
            "cuda_version": "12.8",
            "allow_tf32": False,
        }
        self.assertTrue(VALIDATOR.corrected_runtime_is_valid(cuda_runtime))
        cuda_runtime["accelerator_name"] = "NVIDIA A100-SXM4-80GB"
        self.assertFalse(VALIDATOR.corrected_runtime_is_valid(cuda_runtime))
        runtime["tokenizer_backend"] = "fast"
        self.assertFalse(VALIDATOR.corrected_runtime_is_valid(runtime))

    def test_json_sidecar_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            VALIDATOR.write_json_with_sidecar(path, {"status": "pass", "value": 1})
            self.assertEqual(VALIDATOR.check_json_sidecar(path), VALIDATOR.logical_json_checksum(path))
            self.assertEqual(json.loads(path.read_text())["status"], "pass")

    def test_frozen_family_arithmetic(self) -> None:
        per_model = len(VALIDATOR.COMPONENT_PAIRS) + len(VALIDATOR.ON_GOD_COMPONENTS)
        primary = per_model * len(VALIDATOR.MODELS) * len(VALIDATOR.SUITES)
        controls = len(VALIDATOR.TWO_TOKEN_CONTROLS) * len(VALIDATOR.SUITES)
        self.assertEqual(primary + controls, VALIDATOR.PRIMARY_FAMILY_SIZE)
        self.assertEqual(24 * len(VALIDATOR.SUITES), VALIDATOR.FIXED_MARGIN_FAMILY_SIZE)


if __name__ == "__main__":
    unittest.main()
