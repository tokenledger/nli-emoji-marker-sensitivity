"""Focused acceptance tests for implementation-plan Phase 1."""

from __future__ import annotations

import json
from copy import deepcopy
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from experiment_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    config_hash,
    data_contract_hash,
    load_experiment_config,
    model_config,
    validate_config,
    write_run_manifest,
)
from helpers import model_input_text  # noqa: E402
from preprocessing import normalize_emoji, normalize_slang, remove_markers  # noqa: E402
from transforms import (  # noqa: E402
    EXCLUDED_MARKERS,
    HELD_OUT_MARKERS,
    MAIN_EMOJI_MAP,
    TRAIN_MARKERS,
    apply_emoji_with_metadata,
    apply_marker_with_metadata,
    apply_slang_with_metadata,
    canonicalize_whitespace,
    invert_from_provenance,
    transform_example,
    validate_transform_condition,
)


class TransformationRegistryTests(unittest.TestCase):
    def test_principal_emoji_registry_is_bijective(self):
        self.assertEqual(len(MAIN_EMOJI_MAP), len(set(MAIN_EMOJI_MAP.values())))

    def test_marker_partitions_are_disjoint(self):
        train = set(TRAIN_MARKERS)
        held_out = set(HELD_OUT_MARKERS)
        excluded = set(EXCLUDED_MARKERS)
        self.assertFalse(train & held_out)
        self.assertFalse(excluded & (train | held_out))

    def test_emoji_transform_is_seeded_capped_and_exactly_invertible(self):
        source = "A man, woman, dog, and cat stand beside a car."
        transformed, events = apply_emoji_with_metadata(source, seed=17)
        repeated, repeated_events = apply_emoji_with_metadata(source, seed=17)

        self.assertEqual(transformed, repeated)
        self.assertEqual(events, repeated_events)
        self.assertEqual(len(events), 2)
        self.assertEqual(invert_from_provenance(transformed, events), source)
        for event in events:
            self.assertEqual(source[event.source_start:event.source_end], event.source)
            self.assertEqual(
                transformed[event.output_start:event.output_end], event.replacement
            )

    def test_plural_nouns_are_not_collapsed(self):
        source = "Two dogs watch a dog."
        transformed, events = apply_emoji_with_metadata(source, seed=3)
        self.assertIn("dogs", transformed)
        self.assertEqual([event.source.lower() for event in events], ["dog"])

    def test_marker_provenance_supports_exact_inversion(self):
        source = "A person is outside."
        transformed, events = apply_marker_with_metadata(
            source, prob=1.0, marker="real talk", marker_pool="train", seed=2
        )
        self.assertEqual(transformed, "A person is outside. real talk")
        self.assertEqual(invert_from_provenance(transformed, events), source)

    def test_legacy_slang_provenance_is_stage_invertible(self):
        source = "A dog is going to take a picture."
        transformed, events = apply_slang_with_metadata(source, prob=1.0, rng=None)
        self.assertEqual(transformed, "A doggo is gonna take a pic.")
        self.assertEqual(invert_from_provenance(transformed, events), source)

    def test_legacy_slang_preserves_sentence_initial_case(self):
        source = "Going to the store. Do not stop."
        transformed, events = apply_slang_with_metadata(source, prob=1.0, rng=None)
        self.assertEqual(transformed, "Gonna the store. Dont stop.")
        self.assertEqual(invert_from_provenance(transformed, events), source)

    def test_slang_normalizer_does_not_corrupt_ambiguous_clean_words(self):
        source = "A player kicks the ball near shades during a boxing bout."
        self.assertEqual(normalize_slang(source), source)

    def test_lossy_emoji_registry_is_distinct_in_condition_validation(self):
        example = {"premise": "Two dogs run.", "hypothesis": "Dogs move.", "label": 0}
        lossy = transform_example(example, "lossy_emoji", seed=1, include_metadata=True)
        validate_transform_condition(lossy, "lossy_emoji", require_each_declared=True)

        mislabeled = transform_example(
            example, "emoji", seed=1, emoji_registry="lossy", include_metadata=True
        )
        with self.assertRaisesRegex(ValueError, "wrong emoji registry"):
            validate_transform_condition(mislabeled, "emoji", require_each_declared=True)

    def test_multiword_marker_oracle_removes_complete_phrase(self):
        self.assertEqual(remove_markers("A dog runs. no cap"), "A dog runs.")
        self.assertEqual(remove_markers("real talk A dog runs."), "A dog runs.")

    def test_principal_emoji_normalization_is_unambiguous(self):
        source = "A boy holds a camera."
        transformed, _ = apply_emoji_with_metadata(source, seed=1)
        self.assertEqual(normalize_emoji(transformed), source)

    def test_transform_example_metadata_is_json_serializable(self):
        example = {
            "premise": "A man sees a dog.",
            "hypothesis": "A man sees an animal.",
            "label": 0,
        }
        transformed = transform_example(
            example, "combined", seed=9, noise_prob=1.0, include_metadata=True
        )
        json.dumps(transformed, ensure_ascii=False)
        self.assertEqual(transformed["transform_metadata"]["mode"], "combined")
        self.assertTrue(transformed["transform_metadata"]["events"]["premise"])

    def test_canonical_whitespace_preserves_case_and_punctuation(self):
        self.assertEqual(canonicalize_whitespace("  A\tDog.\n"), "A Dog.")


class FrozenConfigurationTests(unittest.TestCase):
    def test_model_addition_does_not_change_data_contract(self):
        config = load_experiment_config()
        expanded = deepcopy(config)
        expanded['models']['fixture'] = deepcopy(config['models']['electra'])
        expanded['models']['fixture']['checkpoint'] = 'example/fixture-model'
        self.assertNotEqual(config_hash(config), config_hash(expanded))
        self.assertEqual(data_contract_hash(config), data_contract_hash(expanded))

    def test_default_configuration_validates(self):
        config = load_experiment_config()
        self.assertTrue(config["frozen"])
        self.assertEqual(len(config["primary_hypotheses"]), 4)
        self.assertEqual(len(config_hash(config)), 64)
        self.assertEqual(
            config["dataset_protocol"]["snli"]["huggingface_name"],
            "stanfordnlp/snli",
        )
        self.assertEqual(
            config["dataset_protocol"]["multi_nli"]["huggingface_name"],
            "nyu-mll/multi_nli",
        )
        self.assertEqual(
            set(config['models']),
            {'electra', 'roberta', 'roberta_base', 'timelm', 'bertweet'},
        )
        self.assertNotIn('byt5', config['models'])

    def test_social_model_input_contracts_are_explicit(self):
        config = load_experiment_config()
        self.assertEqual(config['models']['roberta']['mixed_precision'], 'bf16')
        self.assertEqual(config['models']['timelm']['input_preprocessing'], 'timelm')
        self.assertEqual(config['models']['bertweet']['tokenizer_backend'], 'slow')
        self.assertEqual(
            config['models']['bertweet']['tokenizer_normalization'], 'disabled'
        )

    def test_timelm_input_preprocessing_only_rewrites_handles_and_links(self):
        source = 'A @person sees https://example.test and an emoji 🐕 frfr'
        self.assertEqual(
            model_input_text(source, 'timelm'),
            'A @user sees http and an emoji 🐕 frfr',
        )
        self.assertEqual(model_input_text(source, 'none'), source)

    def test_model_resolution_uses_frozen_values(self):
        config = load_experiment_config()
        electra = model_config(config, "electra")
        self.assertEqual(electra["model_name"], "google/electra-small-discriminator")
        self.assertEqual(electra["lr"], 0.00002)
        self.assertGreaterEqual(len(electra["seeds"]), 3)
        bertweet = model_config(config, 'bertweet')
        self.assertEqual(bertweet['model_name'], 'vinai/bertweet-base')
        self.assertEqual(bertweet['tokenizer_backend'], 'slow')

    def test_validation_rejects_marker_fold_leakage(self):
        config = deepcopy(load_experiment_config())
        config["transformations"]["marker_folds"][0]["train"].append("no cap")
        with self.assertRaisesRegex(ValueError, "Marker leakage"):
            validate_config(config)

    def test_validation_rejects_runtime_registry_drift(self):
        config = deepcopy(load_experiment_config())
        config["transformations"]["main_emoji_map"]["dog"] = "🐶"
        with self.assertRaisesRegex(ValueError, "Config/runtime registry mismatch"):
            validate_config(config)

    def test_run_manifest_copies_config_and_hash(self):
        config = load_experiment_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = write_run_manifest(
                temp_dir,
                config,
                source_path=DEFAULT_CONFIG_PATH,
                runtime_overrides={"seed": 42, "approach": "baseline"},
            )
            copied = json.loads(
                (Path(temp_dir) / "experiment_config.json").read_text(encoding="utf-8")
            )
            saved_manifest = json.loads(
                (Path(temp_dir) / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(copied, config)
            self.assertEqual(manifest["config_sha256"], config_hash(config))
            self.assertEqual(saved_manifest["runtime_overrides"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
