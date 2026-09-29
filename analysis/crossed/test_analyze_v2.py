#!/usr/bin/env python3
"""Small-data checks for the corrected analysis statistics."""

from __future__ import annotations

import importlib.util
import gzip
import hashlib
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().with_name("analyze_v2.py")
SPEC = importlib.util.spec_from_file_location("corrected_analysis", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def brute_intersection_tail(n: int, margins: list[int], observed: int) -> float:
    subsets = [list(itertools.combinations(range(n), margin)) for margin in margins]
    numerator = 0
    denominator = 0
    for selection in itertools.product(*subsets):
        denominator += 1
        intersection = set(selection[0])
        for subset in selection[1:]:
            intersection.intersection_update(subset)
        numerator += len(intersection) >= observed
    return numerator / denominator


class StatisticalContractTests(unittest.TestCase):
    def test_label_semantics_are_explicit(self) -> None:
        self.assertEqual(
            ANALYSIS.LABEL_NAMES,
            {0: "entailment", 1: "neutral", 2: "contradiction"},
        )

    def test_fixed_margin_tail_matches_enumeration(self) -> None:
        n = 6
        margins = [2, 3, 2]
        for observed in range(3):
            exact = ANALYSIS.exact_intersection_upper_tail(n, margins, observed)
            brute = brute_intersection_tail(n, margins, observed)
            self.assertTrue(math.isclose(exact, brute, rel_tol=1e-12, abs_tol=1e-12))

    def test_fixed_margin_is_model_order_invariant(self) -> None:
        expected = ANALYSIS.exact_intersection_upper_tail(10, [3, 4, 5], 2)
        for permutation in itertools.permutations([3, 4, 5]):
            observed = ANALYSIS.exact_intersection_upper_tail(10, permutation, 2)
            self.assertTrue(math.isclose(expected, observed, abs_tol=1e-15))

    def test_mcnemar_reports_discordants(self) -> None:
        a = np.asarray([1, 1, 1, 0, 0], dtype=bool)
        b = np.asarray([1, 0, 0, 1, 0], dtype=bool)
        p_value, a_only, b_only = ANALYSIS.exact_mcnemar(a, b)
        self.assertEqual((a_only, b_only), (2, 1))
        self.assertEqual(p_value, 1.0)

    def test_holm_family(self) -> None:
        adjusted = ANALYSIS.holm_adjust([0.01, 0.03, 0.02, 1.0])
        np.testing.assert_allclose(adjusted, [0.04, 0.06, 0.06, 1.0])

    def test_bootstrap_is_key_deterministic(self) -> None:
        values = np.asarray([-100.0, 0.0, 0.0, 100.0])
        self.assertEqual(
            ANALYSIS.bootstrap_mean_ci(values, "stable-key"),
            ANALYSIS.bootstrap_mean_ci(values, "stable-key"),
        )

    def test_reporting_floor_never_prints_zero(self) -> None:
        self.assertEqual(ANALYSIS.format_tail_p(0.0), "<1e-300")
        self.assertEqual(ANALYSIS.format_tail_p(1e-310), "<1e-300")
        self.assertNotEqual(ANALYSIS.format_tail_p(1e-20), "<1e-300")

    def test_frozen_family_arithmetic(self) -> None:
        phrase_component = len(ANALYSIS.COMPONENT_PAIRS) * len(ANALYSIS.MODELS) * len(ANALYSIS.SUITES)
        neutral_destination = len(ANALYSIS.ON_GOD_COMPONENTS) * len(ANALYSIS.MODELS) * len(ANALYSIS.SUITES)
        all_five_controls = len(ANALYSIS.TWO_TOKEN_CONTROLS) * len(ANALYSIS.SUITES)
        self.assertEqual(
            phrase_component + neutral_destination + all_five_controls,
            ANALYSIS.FROZEN_PRIMARY_TESTS,
        )

    def test_secondary_topology_arithmetic(self) -> None:
        models = len(ANALYSIS.MODELS)
        suites = len(ANALYSIS.SUITES)
        secondary_contrasts = (
            len(ANALYSIS.COMPONENT_PAIRS) * models * suites
            + len(ANALYSIS.FULL_PHRASES)
            * len(ANALYSIS.TWO_TOKEN_CONTROLS)
            * models
            * suites
            * 2
        )
        pairwise = math.comb(models, 2) * 2 * 24 * suites
        self.assertEqual(secondary_contrasts, 450)
        self.assertEqual(pairwise, 1440)
        self.assertEqual(2 * 24 * suites + pairwise + 12, 1596)

    def test_global_condition_major_positions(self) -> None:
        conditions = ["clean", "marker", "control"]
        expected = [
            ("clean", 0), ("clean", 1),
            ("marker", 0), ("marker", 1),
            ("control", 0), ("control", 1),
        ]
        observed = [
            ANALYSIS.expected_crossing_position(index, 2, conditions)
            for index in range(6)
        ]
        self.assertEqual(observed, expected)
        with self.assertRaises(ValueError):
            ANALYSIS.expected_crossing_position(6, 2, conditions)

    def test_loader_rejects_interleaved_condition_blocks(self) -> None:
        registry = [
            {"condition_id": f"c{index:02d}_x", "condition_type": "synthetic"}
            for index in range(24)
        ]
        registry_by_id = {row["condition_id"]: row for row in registry}
        cell = {
            "cell_id": "model__dataset__split",
            "model": "model",
            "dataset": "dataset",
            "split": "split",
            "source_pairs": 1,
            "evaluation_rows": 24,
            "trained_checkpoint_package_sha256": "a" * 64,
        }
        rows = [
            {
                "schema_version": 1,
                "cell_id": cell["cell_id"],
                "model": cell["model"],
                "dataset": cell["dataset"],
                "split": cell["split"],
                "source_index": 0,
                "condition_id": condition["condition_id"],
                "condition_type": condition["condition_type"],
                "label": 0,
                "prediction": 0,
                "logits": [1.0, 0.0, -1.0],
                "checkpoint_package_sha256": cell["trained_checkpoint_package_sha256"],
            }
            for condition in registry
        ]
        rows[0], rows[1] = rows[1], rows[0]
        logical = hashlib.sha256()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "predictions.jsonl.gz"
            with gzip.open(artifact, "wt", encoding="utf-8") as handle:
                for row in rows:
                    encoded = json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    logical.update(encoded.encode("utf-8") + b"\n")
                    handle.write(encoded + "\n")
            completion = {
                "prediction_logical_checksum_sha256": logical.hexdigest(),
            }
            with self.assertRaisesRegex(RuntimeError, "schema/identity"):
                ANALYSIS.load_prediction_artifact(
                    artifact,
                    completion,
                    cell,
                    registry,
                    registry_by_id,
                )

    def test_correction_identity_payload_is_recomputed(self) -> None:
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
        cell = {
            "cell_id": "bertweet__multi_nli__validation_matched",
            "cell_identity_sha256": "1" * 64,
            "trained_checkpoint_package_sha256": "2" * 64,
            "condition_matrix_checksum_sha256": "3" * 64,
        }
        completion = {
            "checkpoint_identity": {
                "weight_source": "verified_remote_override",
                "weight_sha256": ANALYSIS.PINNED_MNLI_WEIGHT_SHA256,
            },
            "tokenizer_identity": {
                "repository": ANALYSIS.PINNED_BERTWEET_TOKENIZER_REPOSITORY,
                "revision": ANALYSIS.PINNED_BERTWEET_TOKENIZER_REVISION,
                "package_sha256": ANALYSIS.PINNED_BERTWEET_TOKENIZER_PACKAGE_SHA256,
            },
            "inference_code_sha256": "4" * 64,
            "runtime_profile": runtime,
        }
        payload = {
            "schema_version": 2,
            "cell_id": cell["cell_id"],
            "source_cell_identity_sha256": cell["cell_identity_sha256"],
            "checkpoint_package_sha256": cell["trained_checkpoint_package_sha256"],
            "checkpoint_weight_sha256": ANALYSIS.PINNED_MNLI_WEIGHT_SHA256,
            "evaluation_logical_checksum_sha256": cell["condition_matrix_checksum_sha256"],
            "tokenizer_repository": ANALYSIS.PINNED_BERTWEET_TOKENIZER_REPOSITORY,
            "tokenizer_revision": ANALYSIS.PINNED_BERTWEET_TOKENIZER_REVISION,
            "tokenizer_package_sha256": ANALYSIS.PINNED_BERTWEET_TOKENIZER_PACKAGE_SHA256,
            "inference_code_sha256": "4" * 64,
            "runtime_profile": runtime,
        }
        expected = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            ANALYSIS.recompute_correction_identity(completion, cell),
            expected,
        )
        self.assertTrue(ANALYSIS.corrected_runtime_profile_is_valid(runtime))
        cuda_runtime = {
            **runtime,
            "device": "cuda",
            "accelerator_name": "NVIDIA H100 80GB HBM3",
            "accelerator_count": 1,
            "cuda_version": "12.8",
            "allow_tf32": False,
        }
        self.assertTrue(ANALYSIS.corrected_runtime_profile_is_valid(cuda_runtime))
        cuda_runtime["allow_tf32"] = True
        self.assertFalse(ANALYSIS.corrected_runtime_profile_is_valid(cuda_runtime))

    def test_corrected_completion_binds_filename_and_full_identity(self) -> None:
        runtime = {
            "device": "cpu", "threads": 4, "interop_threads": 1,
            "batch_size": 128, "max_length": 128,
            "input_preprocessing": "none", "tokenizer_backend": "slow",
            "tokenizer_normalization": "disabled", "logit_dtype": "float32",
            "seed": 42, "deterministic_algorithms": True, "torch_version": "test",
        }
        cell = {
            "cell_id": "bertweet__multi_nli__validation_matched",
            "dataset": "multi_nli",
            "cell_identity_sha256": "1" * 64,
            "trained_checkpoint_package_sha256": "2" * 64,
            "condition_matrix_checksum_sha256": "3" * 64,
            "source_pairs": 2,
            "evaluation_rows": 48,
        }
        correction_code_sha = "4" * 64
        registry_sha = "5" * 64
        manifest_sha = "6" * 64
        completion = {
            "status": "complete",
            "cell_id": cell["cell_id"],
            "checkpoint_package_sha256": cell["trained_checkpoint_package_sha256"],
            "evaluation_logical_checksum_sha256": cell["condition_matrix_checksum_sha256"],
            "condition_registry_sha256": registry_sha,
            "prediction_rows": cell["evaluation_rows"],
            "prediction_logical_checksum_sha256": "7" * 64,
            "source_cell_identity_sha256": cell["cell_identity_sha256"],
            "source_run_manifest_logical_sha256": manifest_sha,
            "inference_code_sha256": correction_code_sha,
            "tokenizer_identity": {
                "repository": ANALYSIS.PINNED_BERTWEET_TOKENIZER_REPOSITORY,
                "revision": ANALYSIS.PINNED_BERTWEET_TOKENIZER_REVISION,
                "package_sha256": ANALYSIS.PINNED_BERTWEET_TOKENIZER_PACKAGE_SHA256,
                "file_sha256": ANALYSIS.PINNED_BERTWEET_TOKENIZER_FILE_SHA256,
                "backend": "slow",
                "normalization": False,
            },
            "checkpoint_identity": {
                "package_sha256": cell["trained_checkpoint_package_sha256"],
                "weight_source": "verified_remote_override",
                "weight_sha256": ANALYSIS.PINNED_MNLI_WEIGHT_SHA256,
            },
            "runtime_profile": runtime,
            "matrix_validation": {
                "passed": True,
                "condition_count": 24,
                "source_pairs": cell["source_pairs"],
                "matrix_rows": cell["evaluation_rows"],
            },
        }
        identity = ANALYSIS.recompute_correction_identity(completion, cell)
        completion["correction_identity_sha256"] = identity
        filename = f"{cell['cell_id']}__tokenizerfix__{identity[:16]}.jsonl.gz"
        completion["prediction_artifact"] = filename
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / filename
            artifact.write_bytes(b"prediction-bytes")
            completion["prediction_sha256"] = ANALYSIS.sha256_file(artifact)
            ANALYSIS.validate_corrected_completion(
                completion,
                artifact,
                cell,
                registry_sha,
                manifest_sha,
                correction_code_sha,
            )
            completion["tokenizer_identity"]["revision"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "tokenizer_identity"):
                ANALYSIS.validate_corrected_completion(
                    completion,
                    artifact,
                    cell,
                    registry_sha,
                    manifest_sha,
                    correction_code_sha,
                )


if __name__ == "__main__":
    unittest.main()
