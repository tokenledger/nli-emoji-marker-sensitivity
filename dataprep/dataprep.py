#!/usr/bin/env python3
"""Download the source datasets and build the edited evaluation sets.

Subcommands:
  download   freeze SNLI and MultiNLI (training, development, final splits)
  external   fetch ChaosNLI, SICK, and ANLI for the case study
  modify     build the edited evaluation sets from the frozen source rows
  verify     check the generated data against the properties the paper states

Layout of the data directory:
  source_manifest.json, source/<dataset>/...      written by ``download``
  eval_sets/, release_manifest.json               fold study (``modify``)
  crossed/<dataset>__final__<split>.jsonl.gz      crossed study (``modify``)
  content_words/<dataset>__final__<split>.jsonl.gz
  external/                                       written by ``external``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))
sys.path.insert(0, str(MODULE_ROOT))

from experiment_config import (config_hash, data_contract_hash,  # noqa: E402
                               load_experiment_config)

FOLD_STUDY_EDITS = ('emoji', 'markers', 'controls', 'combined', 'other')
PHRASE_EDITS = ('crossed', 'content-words')
ALL_EDITS = FOLD_STUDY_EDITS + PHRASE_EDITS
DATASET_KEYS = {'snli': 'snli', 'mnli': 'multi_nli', 'multi_nli': 'multi_nli'}
FINAL_SPLITS = {
    'snli': ('test',),
    'multi_nli': ('validation_matched', 'validation_mismatched'),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_source(data_dir: Path) -> None:
    if not (data_dir / 'source_manifest.json').is_file():
        raise SystemExit(
            f'No source data in {data_dir}. Run "make download" first, or point '
            'DATA_DIR at an existing data directory.'
        )


def resolve_datasets(names: list[str] | None) -> list[str]:
    if not names:
        return ['snli', 'multi_nli']
    unknown = [name for name in names if name not in DATASET_KEYS]
    if unknown:
        raise SystemExit(f'Unknown DATASETS {unknown}; choose from snli, mnli')
    return sorted({DATASET_KEYS[name] for name in names}, key=['snli', 'multi_nli'].index)


def command_download(args: argparse.Namespace) -> None:
    from prepare_data_release import (atomic_release_directory,
                                      build_source_snapshot,
                                      parse_revision_overrides)

    config = load_experiment_config(args.config)
    overrides = parse_revision_overrides(args.revision)
    with atomic_release_directory(args.data_dir) as staging:
        build_source_snapshot(
            staging, config, overrides, workers=args.workers,
            cache_dir=args.hf_cache_dir,
        )
    print(f'Source data written to {args.data_dir.resolve()}')


def command_external(args: argparse.Namespace) -> None:
    import external_data

    external_data.download(args.data_dir / 'external')


def write_release_manifest(data_dir: Path, config: dict) -> None:
    """Write the release identity that the training code requires."""

    source_path = data_dir / 'source_manifest.json'
    eval_path = data_dir / 'eval_sets' / 'dataset_manifest.json'
    source_manifest = json.loads(source_path.read_text(encoding='utf-8'))
    evaluation_manifest = json.loads(eval_path.read_text(encoding='utf-8'))
    release = {
        'schema_version': 1,
        'experiment_id': config['experiment_id'],
        'config_sha256': config_hash(config),
        'data_contract_sha256': data_contract_hash(config),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_manifest_sha256': sha256_file(source_path),
        'evaluation_manifest_sha256': sha256_file(eval_path),
        'generator_code_sha256': source_manifest['generator_code_sha256'],
    }
    if 'generation_filter' in evaluation_manifest:
        release['generation_filter'] = evaluation_manifest['generation_filter']
    release['release_id'] = hashlib.sha256(
        json.dumps(release, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    (data_dir / 'release_manifest.json').write_text(
        json.dumps(release, indent=2) + '\n', encoding='utf-8'
    )


def build_fold_study(args: argparse.Namespace, datasets: list[str],
                     edits: list[str]) -> None:
    target = args.data_dir / 'eval_sets'
    if target.exists():
        raise SystemExit(
            f'{target} already exists and is never overwritten. Run "make clean" '
            'to remove the generated evaluation sets, then repeat.'
        )
    command = [
        sys.executable, str(REPOSITORY_ROOT / 'src' / 'prepare_eval_sets.py'),
        '--config', str(args.config),
        '--source_snapshot', str(args.data_dir),
        '--out', str(target),
        '--seed', str(args.seed),
        '--workers', str(args.workers),
    ]
    if len(datasets) < 2:
        command += ['--datasets', *datasets]
    if args.splits:
        command += ['--splits', *args.splits]
    if set(edits) != set(FOLD_STUDY_EDITS):
        command += ['--edits', *edits]
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit('Building the fold-study evaluation sets failed; see the error above')
    write_release_manifest(args.data_dir, load_experiment_config(args.config))


def command_modify(args: argparse.Namespace) -> None:
    import phrase_sets

    require_source(args.data_dir)
    frozen_seed = load_experiment_config(args.config)['transformations']['generation_seed']
    if args.seed != frozen_seed:
        raise SystemExit(
            f'SEED={args.seed} is not supported: the configuration fixes the '
            f'generation seed at {frozen_seed}'
        )
    edits = args.edits or list(ALL_EDITS)
    unknown = [name for name in edits if name not in ALL_EDITS]
    if unknown:
        raise SystemExit(f'Unknown EDITS {unknown}; choose from {" ".join(ALL_EDITS)}')
    datasets = resolve_datasets(args.datasets)
    fold_edits = [name for name in FOLD_STUDY_EDITS if name in edits]
    if fold_edits:
        build_fold_study(args, datasets, fold_edits)
    suites = [
        (dataset, split) for dataset in datasets for split in FINAL_SPLITS[dataset]
        if not args.splits or split in args.splits
    ]
    if 'crossed' in edits:
        phrase_sets.build_crossed(args.data_dir, suites)
    if 'content-words' in edits:
        phrase_sets.build_content_words(args.data_dir, suites)


def command_verify(args: argparse.Namespace) -> None:
    import verify

    require_source(args.data_dir)
    failed = verify.run(args.data_dir, load_experiment_config(args.config))
    if failed:
        raise SystemExit(f'{failed} check(s) failed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path,
                        default=REPOSITORY_ROOT / 'configs' / 'experiment_v6.json')
    commands = parser.add_subparsers(dest='command', required=True)

    download = commands.add_parser('download')
    download.add_argument('--revision', action='append', default=[],
                          metavar='DATASET=REVISION')
    download.add_argument('--workers', type=int, default=0)
    download.add_argument('--hf-cache-dir', default=None)
    download.set_defaults(run=command_download)

    external = commands.add_parser('external')
    external.set_defaults(run=command_external)

    modify = commands.add_parser('modify')
    modify.add_argument('--edits', nargs='*', default=None)
    modify.add_argument('--datasets', nargs='*', default=None)
    modify.add_argument('--splits', nargs='*', default=None)
    modify.add_argument('--seed', type=int, default=42)
    modify.add_argument('--workers', type=int, default=0)
    modify.set_defaults(run=command_modify)

    verify_parser = commands.add_parser('verify')
    verify_parser.set_defaults(run=command_verify)

    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
