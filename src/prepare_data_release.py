"""Build one immutable, checksummed data release for every experiment.

This is the only production command that contacts the Hugging Face Hub.  It
resolves each dataset repository to a commit SHA, downloads that exact commit,
freezes the train/development/final rows, and generates the evaluation suite
from those frozen rows.  Training jobs subsequently use ``load_from_disk`` and
can run with ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from experiment_config import (DEFAULT_CONFIG_PATH, config_hash,
                               data_contract_hash, load_experiment_config)
from helpers import logical_dataset_checksum, select_training_and_development


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_tree_sha256() -> str:
    """Hash the source/configuration inputs that define data generation."""

    digest = hashlib.sha256()
    candidates = [
        *sorted((REPOSITORY_ROOT / 'src').glob('*.py')),
        *sorted((REPOSITORY_ROOT / 'configs').glob('*.json')),
        REPOSITORY_ROOT / 'requirements.txt',
    ]
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        digest.update(relative.encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


@contextmanager
def atomic_release_directory(destination: Path):
    """Publish a complete release with one rename; never reuse partial data."""

    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(
            f'Data release already exists: {destination}. Use a new versioned path; '
            'paper data releases are immutable.'
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f'.{destination.name}.building-{uuid.uuid4().hex}'
    staging.mkdir()
    try:
        yield staging
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_revision_overrides(values: list[str]) -> dict[str, str]:
    resolved = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'Expected DATASET=REVISION, got {value!r}')
        dataset, revision = value.split('=', 1)
        if not dataset or not revision:
            raise ValueError(f'Expected DATASET=REVISION, got {value!r}')
        resolved[dataset] = revision
    return resolved


def resolve_revision(repository: str, requested: str | None) -> str:
    """Resolve a branch/tag to the immutable Hub commit used for the release."""

    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repository, revision=requested)
    if not info.sha:
        raise RuntimeError(f'Hugging Face did not return a commit for {repository}')
    return str(info.sha)


def split_record(dataset, *, path: Path, source_split: str, root: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    labels = Counter(int(value) for value in dataset['label'])
    return {
        'path': str(path.relative_to(root)),
        'source_split': source_split,
        'examples': len(dataset),
        'checksum_sha256': logical_dataset_checksum(dataset),
        'arrow_fingerprint': getattr(dataset, '_fingerprint', None),
        'label_counts': {str(key): value for key, value in sorted(labels.items())},
    }


def _matches_source_exclusion(example: dict, rule: dict) -> bool:
    value = str(example.get(rule['field'], ''))
    if rule['match'] == 'exact':
        return value == rule['value']
    if rule['match'] == 'prefix':
        return value.startswith(rule['value'])
    raise ValueError(f"Unsupported source exclusion match: {rule['match']!r}")


def apply_source_exclusions(dataset_dict, rules: list[dict], *, num_proc: int = 1):
    """Remove only config-declared source rows and return a complete audit."""

    if not rules:
        return dataset_dict, {
            'rules': [], 'excluded_examples_by_split': {},
            'excluded_examples_total': 0,
        }
    rule_counts = {rule['id']: Counter() for rule in rules}
    unique_counts = Counter()
    for split_name, rows in dataset_dict.items():
        for row in rows:
            matched = [rule for rule in rules if _matches_source_exclusion(row, rule)]
            if len(matched) > 1:
                raise ValueError(
                    f'Source row in {split_name} matches overlapping exclusions: '
                    f'{[rule["id"] for rule in matched]}'
                )
            if matched:
                rule_counts[matched[0]['id']][split_name] += 1
                unique_counts[split_name] += 1
    for rule in rules:
        observed = sum(rule_counts[rule['id']].values())
        expected = int(rule['expected_total_matches'])
        if observed != expected:
            raise ValueError(
                f'Source exclusion {rule["id"]!r} expected {expected} matches, '
                f'found {observed}'
            )
        expected_by_split = rule.get('expected_matches_by_split')
        if expected_by_split is not None:
            observed_by_split = rule_counts[rule['id']]
            mismatches = {
                split_name: {
                    'expected': int(split_expected),
                    'observed': int(observed_by_split.get(split_name, 0)),
                }
                for split_name, split_expected in expected_by_split.items()
                if observed_by_split.get(split_name, 0) != split_expected
            }
            undeclared = {
                split_name: count
                for split_name, count in observed_by_split.items()
                if split_name not in expected_by_split and count
            }
            if mismatches or undeclared:
                raise ValueError(
                    f'Source exclusion {rule["id"]!r} split counts changed: '
                    f'mismatches={mismatches}, undeclared={undeclared}'
                )
    filtered = dataset_dict.filter(
        lambda example: not any(
            _matches_source_exclusion(example, rule) for rule in rules
        ),
        **({'num_proc': num_proc} if num_proc > 1 else {}),
    )
    for split_name, rows in dataset_dict.items():
        if len(rows) - len(filtered[split_name]) != unique_counts[split_name]:
            raise AssertionError(f'Source exclusion count drift in {split_name}')
    return filtered, {
        'rules': [
            {
                **rule,
                'matches_by_split': dict(sorted(rule_counts[rule['id']].items())),
                'observed_total_matches': sum(rule_counts[rule['id']].values()),
            }
            for rule in rules
        ],
        'excluded_examples_by_split': dict(sorted(unique_counts.items())),
        'excluded_examples_total': sum(unique_counts.values()),
    }


def select_release_splits(filtered, dataset_name: str, protocol: dict, *,
                          num_proc: int = 1):
    """Select stable roles and apply exclusions at the declared contract stage."""

    rules = list(protocol.get('source_exclusions', []))
    stage = protocol.get(
        'source_exclusion_stage', 'before_train_development_selection'
    )
    if stage == 'before_train_development_selection':
        filtered, audit = apply_source_exclusions(
            filtered, rules, num_proc=num_proc,
        )
        training, development = select_training_and_development(
            filtered,
            dataset_name,
            fraction=protocol.get('development_fraction', 0.05),
            seed=protocol.get('development_seed', 42),
        )
        finals = {name: filtered[name] for name in protocol['final_splits']}
        return training, development, finals, audit
    if stage != 'after_train_development_selection':
        raise ValueError(f'Unsupported source exclusion stage: {stage!r}')

    # Select from the unchanged upstream population first. This preserves the
    # frozen train/development membership when a later release removes rows.
    training, development = select_training_and_development(
        filtered,
        dataset_name,
        fraction=protocol.get('development_fraction', 0.05),
        seed=protocol.get('development_seed', 42),
    )
    import datasets
    selected = datasets.DatasetDict({
        'training': training,
        'development': development,
        **{name: filtered[name] for name in protocol['final_splits']},
    })
    selected, audit = apply_source_exclusions(
        selected, rules, num_proc=num_proc,
    )
    finals = {name: selected[name] for name in protocol['final_splits']}
    return selected['training'], selected['development'], finals, audit


def resolve_worker_budget(requested: int) -> int:
    """Resolve ``0`` to the machine's full logical-CPU budget."""
    if requested < 0:
        raise ValueError('workers must be zero (auto) or a positive integer')
    available = max(1, os.cpu_count() or 1)
    return available if requested == 0 else requested


def resolve_worker_count(requested: int, task_count: int) -> int:
    return max(1, min(resolve_worker_budget(requested), max(1, task_count)))


def distribute_worker_budget(total: int, slots: int) -> list[int]:
    if total < 1 or slots < 1 or total < slots:
        raise ValueError('worker budget must cover every active slot')
    base, remainder = divmod(total, slots)
    return [base + int(index < remainder) for index in range(slots)]


def _build_dataset_snapshot(task: dict) -> tuple[str, dict]:
    """Build one dataset subtree in an isolated process."""

    import datasets

    staging = Path(task['staging'])
    config = task['config']
    dataset_name = task['dataset_name']
    requested = task['requested_revision']
    cache_dir = task.get('cache_dir')
    protocol = config['dataset_protocol'][dataset_name]
    repository = protocol['huggingface_name']
    revision = resolve_revision(repository, requested)
    filter_workers = max(1, int(task.get('filter_workers', 1)))
    print(
        f'[source:{dataset_name}] Downloading {repository}@{revision}...',
        flush=True,
    )
    raw = datasets.load_dataset(
        repository,
        revision=revision,
        **({'cache_dir': cache_dir} if cache_dir else {}),
    )
    filtered = raw.filter(
        lambda example: example['label'] != -1,
        **({'num_proc': filter_workers} if filter_workers > 1 else {}),
    )
    training, development, final_splits, exclusion_audit = select_release_splits(
        filtered, dataset_name, protocol, num_proc=filter_workers,
    )
    dataset_root = staging / 'source' / dataset_name
    development_name = (
        protocol['development_split'] if dataset_name == 'snli'
        else 'train_holdout'
    )
    entry = {
        'huggingface_repository': repository,
        'huggingface_revision': revision,
        'requested_revision': requested,
        'source_exclusions': exclusion_audit,
        'training': split_record(
            training, path=dataset_root / 'training',
            source_split=protocol['train_split'], root=staging,
        ),
        'development': split_record(
            development, path=dataset_root / 'development' / development_name,
            source_split=development_name, root=staging,
        ),
        'final': {},
    }
    for split_name in protocol['final_splits']:
        final_data = final_splits[split_name]
        entry['final'][split_name] = split_record(
            final_data, path=dataset_root / 'final' / split_name,
            source_split=split_name, root=staging,
        )
    print(
        f'[source:{dataset_name}] Frozen {len(training):,} train / '
        f'{len(development):,} development', flush=True,
    )
    return dataset_name, entry


def build_source_snapshot(staging: Path, config: dict,
                          revision_overrides: dict[str, str], *,
                          workers: int = 0,
                          cache_dir: str | None = None) -> dict:

    manifest = {
        'schema_version': 1,
        'experiment_id': config['experiment_id'],
        'config_sha256': config_hash(config),
        'data_contract_sha256': data_contract_hash(config),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'generator_code_sha256': code_tree_sha256(),
        'policy': (
            'Hub access occurs only during release creation; all experiment jobs '
            'load these checksummed Arrow snapshots offline.'
        ),
        'datasets': {},
    }
    dataset_names = ('snli', 'multi_nli')
    tasks = []
    for dataset_name in dataset_names:
        protocol = config['dataset_protocol'][dataset_name]
        requested = revision_overrides.get(dataset_name, protocol.get('revision'))
        tasks.append({
            'staging': str(staging),
            'config': config,
            'dataset_name': dataset_name,
            'requested_revision': requested,
            'cache_dir': cache_dir,
        })

    worker_budget = resolve_worker_budget(workers)
    resolved_workers = min(worker_budget, len(tasks))
    allocations = distribute_worker_budget(worker_budget, resolved_workers)
    for index, task in enumerate(tasks):
        task['filter_workers'] = (
            allocations[index] if index < resolved_workers else 1
        )
    print(
        f'Building {len(tasks)} source snapshots with {resolved_workers} dataset '
        f'process(es) and a {worker_budget}-CPU filter budget...',
        flush=True,
    )
    started = time.monotonic()
    completed_entries: dict[str, dict] = {}
    if resolved_workers == 1:
        for index, task in enumerate(tasks, 1):
            dataset_name, entry = _build_dataset_snapshot(task)
            completed_entries[dataset_name] = entry
            print(
                f'[source {index}/{len(tasks)}] completed {dataset_name} '
                f'({(time.monotonic() - started) / 60:.1f} min)', flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            future_tasks = {
                executor.submit(_build_dataset_snapshot, task): task
                for task in tasks
            }
            try:
                for index, future in enumerate(as_completed(future_tasks), 1):
                    dataset_name, entry = future.result()
                    completed_entries[dataset_name] = entry
                    print(
                        f'[source {index}/{len(tasks)}] completed {dataset_name} '
                        f'({(time.monotonic() - started) / 60:.1f} min)', flush=True,
                    )
            except BaseException:
                for future in future_tasks:
                    future.cancel()
                raise

    # Completion order is intentionally discarded so dataset ordering and
    # logical-content checksums remain stable under parallel execution.
    manifest['datasets'] = {
        dataset_name: completed_entries[dataset_name]
        for dataset_name in dataset_names
    }
    manifest_path = staging / 'source_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def generate_evaluation_suite(staging: Path, config_path: Path, *, workers: int) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / 'src' / 'prepare_eval_sets.py'),
        '--config', str(config_path),
        '--source_snapshot', str(staging),
        '--out', str(staging / 'eval_sets'),
        '--workers', str(workers),
    ]
    print('\nGenerating evaluation suite from the frozen source rows...', flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New versioned release directory')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        '--revision', action='append', default=[], metavar='DATASET=REVISION',
        help='Optional Hub tag/branch/commit override; repeat for each dataset.',
    )
    parser.add_argument(
        '--workers', type=int, default=0,
        help='Total parallel CPU budget across source filters and eval conditions; '
             '0 uses all logical CPUs.',
    )
    parser.add_argument(
        '--cache_dir', default=None,
        help='Optional local Hugging Face cache directory.',
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)
    if args.workers < 0:
        parser.error('--workers must be zero (auto) or a positive integer')
    overrides = parse_revision_overrides(args.revision)
    unknown = set(overrides) - set(config['dataset_protocol'])
    if unknown:
        parser.error(f'Unknown revision overrides: {sorted(unknown)}')

    destination = Path(args.out)
    with atomic_release_directory(destination) as staging:
        source_manifest = build_source_snapshot(
            staging, config, overrides,
            workers=args.workers, cache_dir=args.cache_dir,
        )
        generate_evaluation_suite(staging, config_path, workers=args.workers)
        source_path = staging / 'source_manifest.json'
        eval_path = staging / 'eval_sets' / 'dataset_manifest.json'
        release = {
            'schema_version': 1,
            'experiment_id': config['experiment_id'],
            'config_sha256': config_hash(config),
            'data_contract_sha256': data_contract_hash(config),
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_manifest_sha256': _sha256(source_path),
            'evaluation_manifest_sha256': _sha256(eval_path),
            'generator_code_sha256': source_manifest['generator_code_sha256'],
        }
        release_id = hashlib.sha256(
            json.dumps(release, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        release['release_id'] = release_id
        (staging / 'release_manifest.json').write_text(
            json.dumps(release, indent=2) + '\n', encoding='utf-8'
        )
        (staging / 'IMMUTABLE').write_text(
            f'{release_id}\nDo not write caches or run outputs into this directory.\n',
            encoding='utf-8',
        )
    print(f'\nPublished immutable data release: {destination.resolve()}', flush=True)
    print(f'Release ID: {release_id}', flush=True)


if __name__ == '__main__':
    main()
