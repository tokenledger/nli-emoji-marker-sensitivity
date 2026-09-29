"""Focused regression tests for the frozen publication-analysis workstream."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from publication_analysis import (  # noqa: E402
    _portable_provenance,
    apply_control_holm,
    build_augmentation_transfer_rows,
    build_augmentation_transfer_summary,
    build_primary_condition_control_rows,
    build_primary_condition_control_summary,
    build_primary_rows,
    file_sha256,
    holm_adjust,
    load_marker_token_associations,
    modal_prediction_path,
    paired_test,
    predicted_label_distribution,
    parse_statistics_filename,
    write_small_snapshot,
    write_table_bundle,
)


def _write_marker_evidence_fixture(root: Path) -> dict[str, object]:
    config = {
        "transformations": {
            "marker_folds": [{
                "id": "fold_1",
                "train": ["fr"],
                "test": ["fr"],
            }]
        }
    }
    rows = []
    for dataset in ("snli", "multi_nli"):
        for context in ("sentence_start", "continuation"):
            rows.append({
                "model": "electra",
                "model_label": "ELECTRA-small",
                "checkpoint": "google/electra-small-discriminator",
                "dataset": dataset,
                "marker": "fr",
                "marker_context": context,
                "token_position": 0,
                "marker_token_count": 1,
                "token_id": 100,
                "token": "fr",
                "training_examples": 10,
                "training_tokens": 100,
                "subtoken_training_frequency": 2,
                "subtoken_entailment_occurrences": 1,
                "subtoken_entailment_pmi": 0.1,
                "subtoken_neutral_occurrences": 1,
                "subtoken_neutral_pmi": 0.2,
                "subtoken_contradiction_occurrences": 0,
                "subtoken_contradiction_pmi": None,
                "phrase_training_occurrences": 0,
                "status": "ok",
            })
    provenance = {
        "schema_version": 1,
        "config_sha256": "c" * 64,
        "release_manifest_sha256": "r" * 64,
        "source_manifest_sha256": "s" * 64,
        "models": ["electra"],
        "datasets": ["snli", "multi_nli"],
        "pmi_definition": "fixture",
    }
    root.mkdir()
    (root / "marker_token_associations.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.iterdir())
    ]
    (root / "artifact_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "files": files},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return config


class _FakeDataRelease:
    def examples(self, dataset, split, variant):
        return 100


def _test_record(a_wins, b_wins, a_system, a_variant, b_system, b_variant):
    return {
        "statistic": 1.0,
        "pvalue": 0.01,
        "holm_adjusted_pvalue": 0.04,
        "significant_holm": True,
        "n_a_wins": a_wins,
        "n_b_wins": b_wins,
        "system_a": a_system,
        "variant_a": a_variant,
        "system_b": b_system,
        "variant_b": b_variant,
    }


class PublicationAnalysisTests(unittest.TestCase):
    def test_statistics_filename_parsing(self):
        cases = [
            (
                "electra_snli_fold_1.json",
                ("electra", "snli", "test", "fold_1"),
            ),
            (
                "bertweet_snli_test_fold_3.json",
                ("bertweet", "snli", "test", "fold_3"),
            ),
            (
                "roberta_base_multi_nli_validation_matched_fold_2.json",
                ("roberta_base", "multi_nli", "validation_matched", "fold_2"),
            ),
            (
                "roberta_multi_nli_validation_mismatched_fold_1.json",
                ("roberta", "multi_nli", "validation_mismatched", "fold_1"),
            ),
        ]
        for name, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(parse_statistics_filename(Path(name)), expected)

    def test_holm_matches_step_down_reference_values(self):
        observed = holm_adjust([0.01, 0.03, 0.04])
        for value, expected in zip(observed, (0.03, 0.06, 0.06)):
            self.assertAlmostEqual(value, expected)

    def test_control_holm_refuses_to_shrink_incomplete_family(self):
        incomplete = [
            {"family_id": "f", "status": "ok", "raw_pvalue": 0.01, "alpha": 0.05},
            {"family_id": "f", "status": "missing_input", "alpha": 0.05},
            {"family_id": "f", "status": "ok", "raw_pvalue": 0.04, "alpha": 0.05},
        ]
        apply_control_holm(incomplete)
        self.assertEqual(
            {row["family_status"] for row in incomplete}, {"incomplete"}
        )
        self.assertTrue(
            all(row["holm_adjusted_pvalue"] is None for row in incomplete)
        )
        self.assertTrue(
            all(row["significant_holm"] is None for row in incomplete)
        )

        complete = [
            {"family_id": "f", "status": "ok", "raw_pvalue": value, "alpha": 0.05}
            for value in (0.01, 0.03, 0.04)
        ]
        apply_control_holm(complete)
        for row, expected in zip(complete, (0.03, 0.06, 0.06)):
            self.assertAlmostEqual(row["holm_adjusted_pvalue"], expected)

    def test_paired_test_aligns_source_ids_and_uses_treatment_direction(self):
        treatment = {
            "source_indices": [2, 1, 3],
            "labels": [1, 0, 2],
            "predictions": [1, 0, 2],
        }
        reference = {
            "source_indices": [1, 3, 2],
            "labels": [0, 2, 1],
            "predictions": [1, 0, 1],
        }
        result = paired_test(treatment, reference)
        self.assertEqual(result["examples"], 3)
        self.assertEqual(result["treatment_accuracy"], 1.0)
        self.assertAlmostEqual(result["reference_accuracy"], 1 / 3)
        self.assertAlmostEqual(result["difference_pp"], 200 / 3)
        self.assertEqual(result["treatment_only_correct"], 2)
        self.assertEqual(result["reference_only_correct"], 0)

    def test_predicted_label_distribution_names_all_nli_labels(self):
        record = {
            "source_indices": [10, 20, 30, 40],
            "labels": [0, 1, 2, 0],
            "predictions": [0, 1, 1, 2],
        }
        observed = predicted_label_distribution(record, [40, 10, 30])
        self.assertEqual(observed["predicted_entailment_count"], 1)
        self.assertEqual(observed["predicted_neutral_count"], 1)
        self.assertEqual(observed["predicted_contradiction_count"], 1)
        self.assertAlmostEqual(observed["predicted_entailment_rate"], 1 / 3)

    def test_primary_condition_control_contrast_is_descriptive_and_restricted(self):
        rows = []
        for fold in ("fold_1", "fold_2", "fold_3"):
            for control in ("formal", "random"):
                rows.append({
                    "model": "electra",
                    "model_label": "ELECTRA-small",
                    "dataset": "snli",
                    "split": "test",
                    "fold": fold,
                    "marker_status": "unseen",
                    "placement": "hypothesis_suffix",
                    "control_type": control,
                    "difference_pp": 1.0,
                    "raw_pvalue": 0.2,
                    "status": "ok",
                    "family_id": "broad",
                    "holm_adjusted_pvalue": 1.0,
                    "significant_holm": False,
                })
        rows.append({
            **rows[0],
            "placement": "premise_suffix",
        })
        selected = build_primary_condition_control_rows(rows)
        self.assertEqual(len(selected), 6)
        self.assertTrue(
            all(row["claim_status"] == "descriptive_not_preregistered"
                for row in selected)
        )
        summaries = build_primary_condition_control_summary(selected)
        split_summaries = [
            row for row in summaries
            if row["aggregation_scope"] == "model_split"
        ]
        self.assertEqual(len(split_summaries), 2)
        self.assertTrue(all(row["cells"] == 3 for row in split_summaries))
        all_models = [
            row for row in summaries
            if row["aggregation_scope"] == "all_models"
        ]
        self.assertEqual(len(all_models), 2)
        self.assertTrue(all(row["cells"] == 3 for row in all_models))

    def test_primary_rows_use_semantic_treatment_minus_reference_direction(self):
        payload = {
            "confidence_intervals": {
                "baseline": {
                    "original": {
                        "point_estimate": 0.8, "lower": 0.7, "upper": 0.9
                    },
                    "emoji_raw": {
                        "point_estimate": 0.7, "lower": 0.6, "upper": 0.8
                    },
                    "marker_unseen_fold_1_hypothesis_suffix": {
                        "point_estimate": 0.7, "lower": 0.6, "upper": 0.8
                    },
                    "emoji_marker_combined": {
                        "point_estimate": 0.6, "lower": 0.5, "upper": 0.7
                    },
                },
                "preprocessing": {
                    "emoji_raw": {
                        "point_estimate": 0.8, "lower": 0.7, "upper": 0.9
                    }
                },
                "augmented": {
                    "emoji_raw": {
                        "point_estimate": 0.7, "lower": 0.6, "upper": 0.8
                    }
                },
                "hybrid": {
                    "emoji_marker_combined": {
                        "point_estimate": 0.7, "lower": 0.6, "upper": 0.8
                    }
                },
            },
            "primary": {
                "tests": {
                    "H1_emoji_vs_clean_baseline": _test_record(
                        20, 10, "baseline", "original", "baseline", "emoji_raw"
                    ),
                    "H2_marker_vs_clean_baseline": _test_record(
                        20, 10, "baseline", "original", "baseline",
                        "marker_unseen_fold_1_hypothesis_suffix",
                    ),
                    "H3_emoji_normalization_vs_augmentation": _test_record(
                        20, 10, "preprocessing", "emoji_raw",
                        "augmented", "emoji_raw",
                    ),
                    "H4_hybrid_vs_baseline_combined": _test_record(
                        20, 10, "hybrid", "emoji_marker_combined",
                        "baseline", "emoji_marker_combined",
                    ),
                }
            },
        }
        rows = build_primary_rows(
            {("electra", "snli", "test", "fold_1"): {
                "payload": payload, "sha256": "a" * 64
            }},
            _FakeDataRelease(),
            ["electra"],
            [],
        )
        selected = {
            row["hypothesis"]: row
            for row in rows
            if row["dataset"] == "snli"
            and row["split"] == "test"
            and row["fold"] == "fold_1"
        }
        self.assertAlmostEqual(selected["H1"]["difference_pp"], -10)
        self.assertAlmostEqual(selected["H1"]["reference_accuracy"], 0.8)
        self.assertAlmostEqual(selected["H2"]["difference_pp"], -10)
        self.assertAlmostEqual(selected["H3"]["difference_pp"], 10)
        self.assertAlmostEqual(selected["H4"]["difference_pp"], 10)

    def test_augmentation_transfer_is_distinct_and_uses_existing_paired_test(self):
        seen = "marker_seen_fold_1_hypothesis_suffix"
        unseen = "marker_unseen_fold_1_hypothesis_suffix"
        paired = {
            "statistic": 1.0,
            "pvalue": 0.02,
            "holm_adjusted_pvalue": 0.2,
            "significant_holm": False,
            "n_a_wins": 5,
            "n_b_wins": 15,
        }
        report = {
            "confidence_intervals": {
                "baseline": {
                    seen: {"point_estimate": 0.6},
                    unseen: {"point_estimate": 0.5},
                },
                "augmented": {
                    seen: {"point_estimate": 0.7},
                    unseen: {"point_estimate": 0.6},
                },
                "hybrid": {
                    unseen: {"point_estimate": 0.65},
                },
            },
            "exploratory": {
                "n_tests": 930,
                "tests": {
                    "baseline_vs_augmented": {
                        seen: dict(paired),
                        unseen: dict(paired),
                    },
                    "baseline_vs_hybrid": {
                        unseen: {
                            **paired,
                            "n_a_wins": 5,
                            "n_b_wins": 20,
                        },
                    },
                },
            },
        }
        rows = build_augmentation_transfer_rows(
            {("electra", "snli", "test", "fold_1"): {
                "payload": report, "sha256": "b" * 64
            }},
            _FakeDataRelease(),
            ["electra"],
        )
        selected = [
            row for row in rows
            if row["dataset"] == "snli"
            and row["split"] == "test"
            and row["fold"] == "fold_1"
        ]
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(row["status"] == "ok" for row in selected))
        self.assertAlmostEqual(selected[0]["difference_pp"], 10)
        self.assertEqual(
            selected[0]["paired_test_status"],
            "available_existing_exploratory",
        )
        summaries = build_augmentation_transfer_summary(selected)
        self.assertEqual(
            len([row for row in summaries
                 if row["aggregation_scope"] == "all_models"]),
            3,
        )

    def test_portable_provenance_keeps_hash_and_modal_not_temp_path(self):
        source = {
            "schema_version": 1,
            "input_artifacts": [{
                "artifact_type": "baseline_predictions",
                "local_path": "/private/tmp/random/electra_snli_predictions.json",
                "sha256": "a" * 64,
                "modal_volume": "wnut2026-nli-results-v2",
                "modal_path": "publication_results/snli/example/predictions.json",
            }],
        }
        item = _portable_provenance(source)["input_artifacts"][0]
        self.assertNotIn("local_path", item)
        self.assertEqual(
            item["local_filename"], "electra_snli_predictions.json"
        )
        self.assertEqual(item["sha256"], "a" * 64)
        self.assertTrue(item["modal_path"].endswith("predictions.json"))

    def test_marker_token_associations_are_manifest_and_identity_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "evidence"
            config = _write_marker_evidence_fixture(evidence)
            missing = []
            rows, artifact, provenance = load_marker_token_associations(
                evidence,
                config=config,
                config_sha256="c" * 64,
                models=["electra"],
                release_manifest_sha256="r" * 64,
                source_manifest_sha256="s" * 64,
                missing=missing,
            )
            self.assertEqual(len(rows), 4)
            self.assertEqual(missing, [])
            self.assertTrue(artifact["manifest_verified"])
            self.assertEqual(artifact["rows"], 4)
            self.assertEqual(provenance["rows"], 4)

            rows_path = evidence / "marker_token_associations.json"
            rows_path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, r"Artifact (size|SHA-256) mismatch"
            ):
                load_marker_token_associations(
                    evidence,
                    config=config,
                    config_sha256="c" * 64,
                    models=["electra"],
                    release_manifest_sha256="r" * 64,
                    source_manifest_sha256="s" * 64,
                    missing=[],
                )

    def test_absent_marker_token_evidence_is_explicitly_not_run(self):
        missing = []
        rows, artifact, provenance = load_marker_token_associations(
            None,
            config={"transformations": {"marker_folds": []}},
            config_sha256="c" * 64,
            models=["electra"],
            release_manifest_sha256=None,
            source_manifest_sha256=None,
            missing=missing,
        )
        self.assertEqual(rows, [])
        self.assertIsNone(artifact)
        self.assertIsNone(provenance)
        self.assertEqual(
            missing[0]["artifact_type"],
            "marker_training_token_associations",
        )
        self.assertEqual(missing[0]["status"], "not_run")

    def test_roberta_modal_sources_cover_separate_colab_roots(self):
        self.assertTrue(
            modal_prediction_path("roberta", "snli", "test").startswith(
                "_colab/publication_results_roberta42_snli/"
            )
        )
        self.assertTrue(
            modal_prediction_path(
                "roberta", "multi_nli", "validation_matched"
            ).startswith("_colab/publication_results_roberta42_mnli/")
        )
        self.assertTrue(
            modal_prediction_path(
                "roberta_base", "snli", "test"
            ).startswith("_colab/publication_results_roberta_base42/")
        )

    def test_table_bundle_is_byte_deterministic(self):
        rows = [{
            "model": "electra",
            "difference_pp": -1.25,
            "significant_holm": True,
            "status": "ok",
        }]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            write_table_bundle(first, "example", rows)
            write_table_bundle(second, "example", rows)
            for suffix in ("csv", "json", "md", "tex"):
                self.assertEqual(
                    file_sha256(first / f"example.{suffix}"),
                    file_sha256(second / f"example.{suffix}"),
                )

    def test_empty_missing_table_keeps_publication_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_table_bundle(root, "missing_artifacts", [])
            header = (root / "missing_artifacts.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("artifact_type", header)
            self.assertIn("status", header)
            self.assertEqual(
                json.loads(
                    (root / "missing_artifacts.json").read_text(
                        encoding="utf-8"
                    )
                ),
                [],
            )

    def test_compact_snapshot_retains_primary_cells_and_readable_command(self):
        table_names = (
            "primary_results",
            "primary_summary",
            "primary_condition_control_tests",
            "primary_condition_control_summary",
            "augmentation_transfer_diagnostics",
            "augmentation_transfer_summary",
            "marker_control_summary",
            "emoji_diagnostics",
            "intensity_diagnostics",
            "placement_diagnostics",
            "generalization_diagnostics",
            "per_marker_diagnostics",
            "marker_token_associations",
            "truncation_diagnostics",
            "missing_artifacts",
        )
        tables = {name: [] for name in table_names}
        tables["primary_results"] = [{
            "model": "electra",
            "dataset": "snli",
            "split": "test",
            "fold": "fold_1",
            "hypothesis": "H1",
            "status": "ok",
        }]
        tables["truncation_diagnostics"] = [{
            "model": "bertweet",
            "model_label": "BERTweet",
            "status": "ok",
        }]
        tables["marker_token_associations"] = [{
            "model": "bertweet",
            "dataset": "snli",
            "marker": "fr",
            "status": "ok",
        }]
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "snapshot"
            write_small_snapshot(
                destination,
                tables=tables,
                portable_provenance={"schema_version": 1},
                report="# Report\n",
                force=False,
            )
            self.assertTrue((destination / "primary_results.json").is_file())
            readme = (destination / "README.md").read_text(encoding="utf-8")
            self.assertIn(
                "python scripts/publication_analysis.py \\\n"
                "  --statistics-dir",
                readme,
            )
            self.assertIn(
                "computed truncation diagnostics locally for BERTweet",
                readme,
            )
            self.assertNotIn("BERTweet as `missing_tokenizer`", readme)
            self.assertTrue(
                (destination / "marker_token_associations.json").is_file()
            )
            self.assertIn(
                "--marker-token-associations "
                "audit_evidence/marker_token_associations",
                readme,
            )
            self.assertIn(
                "No planned artifacts are marked missing or not run.",
                readme,
            )


if __name__ == "__main__":
    unittest.main()
