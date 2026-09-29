"""Tests for evaluation-only checkpoint reuse."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from evaluate_checkpoint import stage_model_locally  # noqa: E402


class EvaluateCheckpointTests(unittest.TestCase):
    def test_drive_model_is_staged_once_in_local_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'drive-run'
            model = run / 'final_model'
            model.mkdir(parents=True)
            (model / 'config.json').write_text('{"fixture": 1}')
            (run / 'run_manifest.json').write_text(json.dumps({'id': 'run'}))
            cache = root / 'local-cache'

            first = stage_model_locally(run, str(cache))
            second = stage_model_locally(run, str(cache))

            self.assertEqual(first, second)
            self.assertEqual((first / 'config.json').read_text(), '{"fixture": 1}')
            self.assertTrue((first / '.complete').is_file())


if __name__ == '__main__':
    unittest.main()
