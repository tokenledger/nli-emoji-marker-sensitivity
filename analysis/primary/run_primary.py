#!/usr/bin/env python3
"""Compute the four prespecified contrasts and Table 2 from saved predictions.

Steps:
  1. paired tests for every encoder, split, and fold of the five-encoder fold
     study (``src/statistical_tests.py``), written to OUTPUT_DIR/primary/statistics;
  2. publication tables (``src/publication_analysis.py``), written to
     OUTPUT_DIR/primary/tables;
  3. the separate DeBERTa-v3-base family (``src/deberta_ext_statistics.py``),
     written to OUTPUT_DIR/primary/deberta;
  4. Table 2 of the paper (``table2.py``), written to OUTPUT_DIR/primary.

No model is run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))
sys.path.insert(0, str(ANALYSIS_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
import table2  # noqa: E402
from deberta_ext_statistics import run as run_deberta_statistics  # noqa: E402
from publication_analysis import (DEFAULT_MODELS, modal_prediction_path,  # noqa: E402
                                  prediction_filename, run_publication_analysis)
from statistical_tests import run_tests  # noqa: E402

SPLITS = (
    ('snli', 'test'),
    ('multi_nli', 'validation_matched'),
    ('multi_nli', 'validation_mismatched'),
)
FOLDS = ('fold_1', 'fold_2', 'fold_3')


def baseline_predictions(fold_study: Path, model: str, dataset: str, split: str) -> Path:
    path = fold_study / modal_prediction_path(model, dataset, split)
    if not path.is_file():
        raise SystemExit(f'Required input is missing: {path}')
    return path


def compute_statistics(fold_study: Path, models: list[str], destination: Path) -> None:
    for model in models:
        for dataset, split in SPLITS:
            results_dir = baseline_predictions(fold_study, model, dataset, split).parents[1]
            for fold in FOLDS:
                target = destination / f'{model}_{dataset}_{split}_{fold}.json'
                if target.is_file():
                    print(f'Reusing {target.name}')
                    continue
                run_tests(results_dir, model, fold, 'final', split, output_path=target)


def stage_predictions(fold_study: Path, models: list[str], destination: Path) -> None:
    for model in models:
        for dataset, split in SPLITS:
            link = destination / prediction_filename(model, dataset, split)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(baseline_predictions(fold_study, model, dataset, split))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', choices=DEFAULT_MODELS,
                        default=list(DEFAULT_MODELS))
    parser.add_argument('--config', type=Path,
                        default=REPOSITORY_ROOT / 'configs' / 'experiment_v6.json')
    parser.add_argument('--tokenizer-mode', choices=('off', 'local', 'download'),
                        default='off',
                        help='Truncation diagnostics: off, cached tokenizers only, '
                             'or download tokenizers.')
    args = parser.parse_args()

    fold_study = paths.predictions('fold_study')
    statistics = paths.output('primary', 'statistics')
    staged = paths.output('primary', 'predictions')
    compute_statistics(fold_study, args.models, statistics)
    stage_predictions(fold_study, args.models, staged)
    run_publication_analysis(
        statistics_dirs=[statistics],
        config_path=args.config,
        data_release_path=paths.data(),
        predictions_dir=staged,
        output_dir=paths.output('primary') / 'tables',
        models=args.models,
        tokenizer_mode=args.tokenizer_mode,
        force=True,
    )
    run_deberta_statistics(
        paths.predictions('deberta'), fold_study, paths.output('primary', 'deberta'),
    )
    table2.write(paths.output('primary'))


if __name__ == '__main__':
    main()
