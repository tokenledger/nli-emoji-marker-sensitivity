"""Tests for compact live training progress artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from progress_logging import ProgressCallback  # noqa: E402


class ProgressLoggingTests(unittest.TestCase):
    def test_callback_writes_latest_state_and_append_only_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            callback = ProgressCallback(temporary)
            state = SimpleNamespace(global_step=10, max_steps=100, epoch=1.0)
            callback.on_train_begin(None, state, None)
            state.global_step = 20
            state.epoch = 1.2
            callback.on_log(None, state, None, logs={'loss': 0.5})

            latest = json.loads((Path(temporary) / 'progress.json').read_text())
            events = (Path(temporary) / 'progress.jsonl').read_text().splitlines()
            self.assertEqual(latest['step'], 20)
            self.assertEqual(latest['percent'], 20.0)
            self.assertEqual(latest['loss'], 0.5)
            self.assertEqual(len(events), 2)


if __name__ == '__main__':
    unittest.main()
