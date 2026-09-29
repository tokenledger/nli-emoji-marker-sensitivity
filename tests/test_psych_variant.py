"""Acceptance tests for the evaluation-only psych inversion condition."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import datasets


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from experiment_config import load_experiment_config  # noqa: E402
from helpers import (  # noqa: E402
    attach_condition_metadata,
    limit_eval_suite,
    load_eval_suite,
    logical_dataset_checksum,
    psych_inversion_result,
    robustness_drops,
)
from prepare_eval_sets import (  # noqa: E402
    _condition_specs,
    dataset_checksum,
    generate_audited_records,
    paired_clean_records,
)
from preprocessing import preprocess_example  # noqa: E402
from statistical_tests import run_tests  # noqa: E402
from transforms import (  # noqa: E402
    PSYCH_LABEL_POLICY,
    invert_from_provenance,
    transform_example,
)


ROWS = [
    {"premise": "A person reads.", "hypothesis": "Someone reads.", "label": 0},
    {"premise": "A person reads.", "hypothesis": "A dog runs.", "label": 1},
    {"premise": "A person reads.", "hypothesis": "Nobody reads.", "label": 2},
]


def _write_suite(root: Path, variants: dict[str, tuple[list[dict], str]]) -> None:
    base = root / "snli" / "final" / "test"
    original = datasets.Dataset.from_list(ROWS)
    original.save_to_disk(str(base / "original"))
    declared = {}
    for name, (records, policy) in variants.items():
        transformed = datasets.Dataset.from_list(records)
        transformed.save_to_disk(str(base / name))
        clean = paired_clean_records(ROWS, records)
        declared[name] = {
            "path": f"snli/final/test/{name}",
            "paired_clean_storage": "source_index_view",
            "examples": len(records),
            "checksum_sha256": logical_dataset_checksum(transformed),
            "paired_clean_checksum_sha256": logical_dataset_checksum(clean),
            "label_policy": policy,
            "label_changing": policy == PSYCH_LABEL_POLICY,
            "evaluation_family": (
                "psych_instruction_inversion"
                if policy == PSYCH_LABEL_POLICY
                else "label_preserving_robustness"
            ),
        }
    manifest = {
        "schema_version": 4,
        "config_sha256": "fixture",
        "seed": 42,
        "label_policies": {
            "preserve": {"label_changing": False},
            PSYCH_LABEL_POLICY: {
                "label_changing": True,
                "mapping": {"0": 2, "1": 1, "2": 0},
            },
        },
        "datasets": {
            "snli": {
                "default_development_split": "validation",
                "default_final_split": "test",
                "splits": {
                    "final:test": {
                        "base_path": "snli/final/test",
                        "role": "final",
                        "source_split": "test",
                        "source_checksum_sha256": logical_dataset_checksum(original),
                        "variants": declared,
                    }
                },
            }
        },
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


class PsychTransformTests(unittest.TestCase):
    def test_all_label_transitions_and_hypothesis_only_suffix(self):
        for source, expected in zip(ROWS, (2, 1, 0)):
            with self.subTest(label=source["label"]):
                transformed = transform_example(
                    source, "psych", seed=999, include_metadata=True
                )
                self.assertEqual(transformed["label"], expected)
                self.assertEqual(transformed["premise"], source["premise"])
                self.assertEqual(
                    transformed["hypothesis"], source["hypothesis"] + " psych"
                )
                self.assertTrue(transformed["psych_applied"])
                self.assertFalse(
                    transformed["transform_metadata"]["events"]["premise"]
                )
                event = transformed["transform_metadata"]["events"]["hypothesis"]
                self.assertEqual(len(event), 1)
                self.assertEqual(event[0]["placement"], "suffix")
                self.assertEqual(
                    invert_from_provenance(transformed["hypothesis"], event),
                    source["hypothesis"],
                )

    def test_generation_is_deterministic_and_exactly_inverts_natural_collision(self):
        rows = [
            {"premise": "A student studies.", "hypothesis": "They study psych", "label": 1}
        ]
        first, first_audit = generate_audited_records(
            rows, "psych", dataset_name="fixture", source_split="test", seed=7
        )
        second, second_audit = generate_audited_records(
            rows, "psych", dataset_name="fixture", source_split="test", seed=999
        )
        self.assertEqual(first, second)
        self.assertEqual(dataset_checksum(first), dataset_checksum(second))
        self.assertEqual(first_audit["checksum_sha256"], second_audit["checksum_sha256"])
        self.assertEqual(first[0]["hypothesis"], "They study psych psych")
        events = first[0]["transform_metadata"]["events"]["hypothesis"]
        self.assertEqual(
            invert_from_provenance(first[0]["hypothesis"], events), "They study psych"
        )

    def test_natural_psych_text_does_not_trigger_relabeling(self):
        source = {
            "premise": "A student studies psychology.",
            "hypothesis": "The student studies psych.",
            "label": 0,
        }
        ordinary = transform_example(
            source, "noise", noise_prob=0.0, seed=4, include_metadata=True
        )
        self.assertEqual(ordinary["label"], 0)
        self.assertNotIn("psych_applied", ordinary)
        self.assertFalse(ordinary["transform_metadata"]["psych_applied"])

    def test_inference_preprocessing_preserves_psych(self):
        row = transform_example(ROWS[0], "psych", include_metadata=True)
        for mode in ("stage_matched", "emoji", "marker_oracle", "all"):
            with self.subTest(mode=mode):
                processed = preprocess_example(row, mode=mode)
                self.assertTrue(processed["hypothesis"].endswith(" psych"))

    def test_psych_is_evaluation_only_and_not_a_seventh_system(self):
        config = load_experiment_config()
        self.assertEqual(len(config["systems"]), 6)
        self.assertEqual(len(config["primary_hypotheses"]), 4)
        psych_specs = [spec for spec in _condition_specs(config) if spec["kind"] == "psych"]
        self.assertEqual(psych_specs, [{"id": "psych", "kind": "psych"}])
        self.assertTrue(config["transformations"]["psych"]["evaluation_only"])


class PsychManifestTests(unittest.TestCase):
    def _psych_records(self):
        records, _ = generate_audited_records(
            ROWS, "psych", dataset_name="fixture", source_split="test", seed=42
        )
        return records

    def test_exact_mapping_and_source_index_alignment_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self._psych_records()
            _write_suite(root, {"psych": (records, PSYCH_LABEL_POLICY)})
            suite = load_eval_suite(root / "snli", variants=["psych"])
            self.assertEqual(list(suite["psych"]["label"]), [2, 1, 0])
            self.assertEqual(list(suite["psych"]["source_index"]), [0, 1, 2])
            self.assertEqual(list(suite["clean_psych"]["label"]), [0, 1, 2])
            self.assertEqual(
                list(suite["psych"]["source_index"]),
                list(suite["clean_psych"]["source_index"]),
            )

    def test_manifest_and_logical_checksum_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self._psych_records()
            _write_suite(root / "first", {"psych": (records, PSYCH_LABEL_POLICY)})
            _write_suite(root / "second", {"psych": (records, PSYCH_LABEL_POLICY)})
            self.assertEqual(dataset_checksum(records), dataset_checksum(records))
            self.assertEqual(
                (root / "first" / "dataset_manifest.json").read_bytes(),
                (root / "second" / "dataset_manifest.json").read_bytes(),
            )

    def test_psych_rejects_any_mapping_other_than_declared_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self._psych_records()
            records[0] = {**records[0], "label": 0}
            _write_suite(root, {"psych": (records, PSYCH_LABEL_POLICY)})
            with self.assertRaisesRegex(ValueError, "invert_entailment_contradiction"):
                load_eval_suite(root / "snli", variants=["psych"])

    def test_natural_word_with_no_flag_or_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = [
                {
                    "premise": ROWS[0]["premise"],
                    "hypothesis": ROWS[0]["hypothesis"] + " psych",
                    "label": 2,
                    "source_index": 0,
                }
            ]
            _write_suite(root, {"psych": (raw, PSYCH_LABEL_POLICY)})
            with self.assertRaisesRegex(ValueError, "psych_applied=True"):
                load_eval_suite(root / "snli", variants=["psych"])

    def test_preserve_policy_still_requires_equal_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            changed = [
                {
                    "premise": "A person reads. frfr",
                    "hypothesis": ROWS[0]["hypothesis"],
                    "label": 2,
                    "source_index": 0,
                }
            ]
            _write_suite(root, {"noise": (changed, "preserve")})
            with self.assertRaisesRegex(ValueError, "preserve policy"):
                load_eval_suite(root / "snli", variants=["noise"])

    def test_eval_variants_all_and_limited_smoke_keep_psych_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            psych = self._psych_records()
            noise = [
                {
                    **row,
                    "hypothesis": row["hypothesis"] + " frfr",
                    "source_index": index,
                }
                for index, row in enumerate(ROWS)
            ]
            _write_suite(
                root,
                {"psych": (psych, PSYCH_LABEL_POLICY), "noise": (noise, "preserve")},
            )
            suite = load_eval_suite(root / "snli", variants=["all"])
            self.assertEqual(
                set(suite),
                {"original", "noise", "clean_noise", "psych", "clean_psych"},
            )
            limited = limit_eval_suite(suite, 2)
            self.assertEqual(len(limited["psych"]), 2)
            self.assertEqual(
                list(limited["psych"]["source_index"]),
                list(limited["clean_psych"]["source_index"]),
            )


class PsychReportingTests(unittest.TestCase):
    def test_psych_is_excluded_from_drops_and_reported_as_inversion_accuracy(self):
        results = {
            "original": {
                "accuracy": 1.0, "predictions": [0], "labels": [0]
            },
            "noise": {"accuracy": 0.75, "predictions": [0], "labels": [0]},
            "clean_noise": {
                "accuracy": 1.0, "predictions": [0], "labels": [0]
            },
            "psych": {"accuracy": 0.5, "predictions": [2, 1], "labels": [2, 0]},
            "clean_psych": {
                "accuracy": 1.0, "predictions": [0, 1], "labels": [0, 1]
            },
        }
        attach_condition_metadata(results)
        self.assertEqual(robustness_drops(results), {"noise": 25.0})
        report = psych_inversion_result(results)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertFalse(report["included_in_primary_holm_family"])
        self.assertTrue(results["psych"]["condition_metadata"]["label_changing"])

    def test_primary_and_exploratory_holm_families_exclude_psych(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_variants = [
                "emoji_raw",
                "emoji_marker_combined",
                "marker_unseen_fold_1_hypothesis_suffix",
            ]
            ordinary = {
                "predictions": [0, 1, 2],
                "labels": [0, 1, 2],
                "source_indices": [0, 1, 2],
            }
            psych = {
                "predictions": [2, 1, 0],
                "labels": [2, 1, 0],
                "source_indices": [0, 1, 2],
                "condition_metadata": {
                    "label_policy": PSYCH_LABEL_POLICY,
                    "label_changing": True,
                },
            }
            predictions = {
                "original": ordinary,
                **{variant: ordinary for variant in primary_variants},
                "psych": psych,
            }
            identity = {
                "config_sha256": "fixture",
                "generation_seed": 42,
                "dataset": "snli",
                "split_role": "final",
                "split_name": "test",
                "source_checksum_sha256": "source",
                "evaluated_variants": [*primary_variants, "psych"],
                "variant_checksums": {
                    **{variant: {"label_policy": "preserve"}
                       for variant in primary_variants},
                    "psych": {"label_policy": PSYCH_LABEL_POLICY},
                },
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
            self.assertEqual(report["primary"]["n_tests"], 4)
            self.assertEqual(report["exploratory"]["n_tests"], 60)
            psych_report = report["psych_instruction_inversion"]
            self.assertFalse(psych_report["included_in_primary_holm_family"])
            self.assertIn("psych", psych_report["conditions"])


if __name__ == "__main__":
    unittest.main()
