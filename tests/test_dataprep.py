"""Crossed-study and content-word sets built by the dataprep module."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "dataprep",
                  REPOSITORY_ROOT / "analysis" / "content_words"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import phrase_sets  # noqa: E402
from conditions import NEW_CONDITIONS  # noqa: E402
from prepare_eval_sets import (EDIT_GROUPS, _condition_specs,  # noqa: E402
                               condition_edit_group)
from experiment_config import load_experiment_config  # noqa: E402

SOURCES = [
    {"source_index": 0, "premise": "A dog runs.", "hypothesis": "An animal moves.", "label": 0},
    {"source_index": 1, "premise": "A man sleeps.", "hypothesis": "A man is awake.", "label": 2},
]


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class PhraseSetTests(unittest.TestCase):
    def build(self, builder, directory):
        with patch.object(phrase_sets, "load_source", return_value=SOURCES):
            builder(Path(directory), [("snli", "test")])

    def test_crossed_study_has_24_conditions_on_every_source_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            self.build(phrase_sets.build_crossed, directory)
            rows = read_rows(Path(directory) / "crossed" / "snli__final__test.jsonl.gz")
        self.assertEqual(len(rows), 24 * len(SOURCES))
        self.assertEqual(len({row["condition_id"] for row in rows}), 24)
        for row in rows:
            source = SOURCES[row["source_index"]]
            self.assertEqual(row["premise"], source["premise"])
            self.assertEqual(row["label"], source["label"])
            expected = source["hypothesis"]
            if row["inserted_text"]:
                expected += " " + row["inserted_text"]
            self.assertEqual(row["hypothesis"], expected)

    def test_content_word_phrases_match_the_inference_registry(self):
        self.assertEqual(
            [tuple(item) for item in phrase_sets.CONTENT_WORD_CONDITIONS],
            [tuple(item) for item in NEW_CONDITIONS],
        )
        with tempfile.TemporaryDirectory() as directory:
            self.build(phrase_sets.build_content_words, directory)
            rows = read_rows(
                Path(directory) / "content_words" / "snli__final__test.jsonl.gz"
            )
        self.assertEqual(len(rows), 9 * len(SOURCES))
        for row in rows:
            self.assertEqual(len(row["inserted_text"].split()), 2)
            self.assertEqual(
                row["hypothesis"],
                SOURCES[row["source_index"]]["hypothesis"] + " " + row["inserted_text"],
            )

    def test_build_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self.build(phrase_sets.build_crossed, first)
            self.build(phrase_sets.build_crossed, second)
            name = "snli__final__test.jsonl.gz"
            self.assertEqual(
                (Path(first) / "crossed" / name).read_bytes(),
                (Path(second) / "crossed" / name).read_bytes(),
            )


class EditGroupTests(unittest.TestCase):
    def test_every_condition_belongs_to_one_edit_group(self):
        specs = _condition_specs(load_experiment_config())
        groups = [condition_edit_group(spec) for spec in specs]
        self.assertTrue(set(groups) <= set(EDIT_GROUPS))
        self.assertEqual(groups.count("emoji"), 2)
        self.assertEqual(groups.count("markers"), 18)
        self.assertEqual(groups.count("controls"), 36)
        self.assertEqual(groups.count("combined"), 3)


if __name__ == "__main__":
    unittest.main()
