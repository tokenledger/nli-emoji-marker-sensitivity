"""Run the frozen publication experiment locally, one step or one model at a time.

Generate an immutable data release with prepare_data_release.py first. Then
provide that release with --data_release for every production training run.
Use --step all for the complete serial pipeline, or select individual steps
when jobs need to be scheduled separately.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiment_config import (
    DEFAULT_CONFIG_PATH,
    config_hash,
    data_contract_hash,
    load_experiment_config,
    model_config,
    write_run_manifest,
)
from training_metadata import system_stage_metadata
from helpers import parse_train_size, source_dataset_identity

STEPS = [
    'baseline', 'preprocessing', 'augmentation', 'hybrid',
    'marker_oracle', 'clean_control', 'stats', 'all',
]


def scoped_run_dir(p, prefix, approach, eval_role, eval_split):
    safe_split = str(eval_split).replace('/', '_').replace(':', '_')
    return Path(p) / f'{prefix}_{approach}_{eval_role}_{safe_split}'


def record_run_attempt(output_dir, status):
    """Persist both the latest status and an append-only attempt history."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(status, sort_keys=True)
    (destination / 'run_status.json').write_text(
        json.dumps(status, indent=2) + '\n', encoding='utf-8'
    )
    with (destination / 'run_attempts.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(serialized + '\n')


def validate_completed_run(output_dir, *, system_id, require_model=False,
                           require_prediction_identity=True, expected_manifest=None):
    """Validate a reusable completed artifact rather than trusting path existence."""

    destination = Path(output_dir)
    required = ['run_status.json', 'run_manifest.json', 'results.json', 'predictions.json']
    if require_prediction_identity:
        required.append('predictions_manifest.json')
    if require_model:
        required.append('final_model')
    missing = [name for name in required if not (destination / name).exists()]
    if require_model and (destination / 'final_model').is_dir() and not any(
        (destination / 'final_model').iterdir()
    ):
        missing.append('final_model/*')
    if missing:
        raise RuntimeError(f'Incomplete run artifact {destination}; missing {missing}')
    status = json.loads((destination / 'run_status.json').read_text(encoding='utf-8'))
    if status.get('status') != 'completed' or status.get('exit_code') != 0:
        raise RuntimeError(
            f'Run artifact {destination} is not reusable: status={status.get("status")!r}'
        )
    results = json.loads((destination / 'results.json').read_text(encoding='utf-8'))
    if results.get('system_id') != system_id:
        raise RuntimeError(
            f'Run artifact {destination} has system_id={results.get("system_id")!r}, '
            f'expected {system_id!r}'
        )
    if results.get('run_metadata', {}).get('run_origin') != 'frozen_config':
        raise RuntimeError(f'Run artifact {destination} is not a frozen-config result')
    if expected_manifest:
        manifest = json.loads((destination / 'run_manifest.json').read_text(encoding='utf-8'))
        overrides = manifest.get('runtime_overrides', {})
        expected_hash = expected_manifest.get('config_sha256')
        if expected_hash and manifest.get('config_sha256') != expected_hash:
            raise RuntimeError(f'Run artifact {destination} has a different config hash')
        mismatches = {
            key: (overrides.get(key), value)
            for key, value in expected_manifest.items()
            if key != 'config_sha256' and overrides.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f'Run artifact {destination} identity mismatch: {mismatches}')
    return results


def completed_run_exists(output_dir):
    status_path = Path(output_dir) / 'run_status.json'
    if not status_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding='utf-8'))
    return status.get('status') == 'completed' and status.get('exit_code') == 0


def run(cmd, label, output_dir=None):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        result = subprocess.run(cmd, check=False)
    except BaseException as error:
        status = {
            'label': label,
            'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            'exit_code': None,
            'started_at_utc': started_at,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'runtime_seconds': time.monotonic() - started,
            'command': [str(part) for part in cmd],
            'error': f'{type(error).__name__}: {error}',
        }
        if output_dir is not None:
            record_run_attempt(output_dir, status)
        raise
    status = {
        'label': label,
        'status': 'completed' if result.returncode == 0 else 'failed',
        'exit_code': result.returncode,
        'started_at_utc': started_at,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'runtime_seconds': time.monotonic() - started,
        'command': [str(part) for part in cmd],
    }
    if output_dir is not None:
        record_run_attempt(output_dir, status)
    if result.returncode != 0:
        print(f'ERROR: {label} failed (exit {result.returncode})')
        sys.exit(result.returncode)


def build_common(cfg, training, dataset, seed, smoke_test, eval_sets_dir, eval_split=None,
                 eval_variants=None, development_seed=42, eval_role='final',
                 data_release=None, cache_dir=None, model_revision=None,
                 preprocessing_workers=1, smoke_train_samples=100,
                 smoke_eval_samples=100, train_size='full'):
    smoke_flag = (
        ['--smoke_test', '--smoke_train_samples', str(smoke_train_samples),
         '--smoke_eval_samples', str(smoke_eval_samples)]
        if smoke_test else []
    )
    eval_sets_flag = ['--eval_sets_dir', str(eval_sets_dir)] if eval_sets_dir else []
    if eval_sets_flag and eval_split:
        eval_sets_flag += ['--eval_split', eval_split]
    if eval_sets_flag:
        eval_sets_flag += ['--eval_role', eval_role]
    if eval_sets_flag and eval_variants:
        eval_sets_flag += ['--eval_variants', *eval_variants]
    source_flags = []
    if data_release:
        source_flags += ['--data_release', str(data_release)]
    if cache_dir:
        source_flags += ['--cache_dir', str(cache_dir)]
    if model_revision:
        source_flags += ['--model_revision', str(model_revision)]
    source_flags += ['--preprocessing_workers', str(preprocessing_workers)]
    return [
        '--model_name', cfg['model_name'],
        '--dataset', dataset,
        '--batch_size', str(cfg['batch_size']),
        '--grad_accum', str(cfg['grad_accum']),
        '--lr', str(cfg['lr']),
        '--warmup_ratio', str(training['warmup_ratio']),
        '--epochs', str(training['epochs']),
        '--max_length', str(training['max_length']),
        '--seed', str(seed),
        '--train_size', str(train_size),
        '--development_seed', str(development_seed),
        '--checkpoint_selection', training['checkpoint_selection'],
        '--tokenizer_backend', cfg.get('tokenizer_backend', 'fast'),
        '--tokenizer_normalization', cfg.get('tokenizer_normalization', 'default'),
        '--input_preprocessing', cfg.get('input_preprocessing', 'none'),
        '--mixed_precision', cfg.get('mixed_precision', 'auto'),
        '--frozen_run',
    ] + smoke_flag + eval_sets_flag + source_flags


def materialize_manifest(p, approach, cfg, experiment, config_path, dataset, seed,
                         eval_split=None, eval_role='final', eval_variants=None,
                         system_id=None):
    system_id = system_id or approach
    run_mode = cfg.get('run_mode', 'production')
    stage_metadata = system_stage_metadata(system_id)
    output_dir = scoped_run_dir(
        p, cfg['prefix'], approach, eval_role, eval_split or 'manifest_default'
    )
    existing_manifest = output_dir / 'run_manifest.json'
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding='utf-8'))
        overrides = existing.get('runtime_overrides', {})
        expected = {
            'system_id': system_id,
            'dataset': dataset,
            'seed': seed,
            'train_size': cfg.get('train_size', 'full'),
            'learning_rate': cfg['lr'],
            'run_mode': run_mode,
            'reported_evaluation_role': eval_role,
            'reported_evaluation_split': eval_split or 'manifest_default',
            'evaluated_variants': sorted(eval_variants or []),
            'data_release_id': cfg.get('data_release_id'),
            'source_manifest_sha256': cfg.get('source_manifest_sha256'),
            'model_revision': cfg.get('model_revision'),
            'tokenizer_backend': cfg.get('tokenizer_backend', 'fast'),
            'tokenizer_normalization': cfg.get('tokenizer_normalization', 'default'),
            'input_preprocessing': cfg.get('input_preprocessing', 'none'),
            'mixed_precision': cfg.get('mixed_precision', 'auto'),
            'trained_checkpoint_split': cfg.get('reuse_trained_from_split') or eval_split,
        }
        observed = {key: overrides.get(key) for key in expected}
        observed['train_size'] = overrides.get('train_size', 'full')
        if observed != expected or existing.get('config_sha256') != config_hash(experiment):
            raise RuntimeError(
                f'Refusing to reuse {output_dir}: existing identity {observed} '
                f'does not match requested identity {expected}'
            )
        return output_dir
    write_run_manifest(
        output_dir,
        experiment,
        source_path=config_path,
        runtime_overrides={
            'approach': approach,
            **stage_metadata,
            'dataset': dataset,
            'model': cfg['prefix'],
            'seed': seed,
            'train_size': cfg.get('train_size', 'full'),
            'learning_rate': cfg['lr'],
            'run_mode': run_mode,
            'learning_rate_sweep': cfg['learning_rate_sweep'],
            'hyperparameter_selection_role': 'development',
            'checkpoint_selection_role': 'development',
            'reported_evaluation_role': eval_role,
            'reported_evaluation_split': eval_split or 'manifest_default',
            'evaluated_variants': sorted(eval_variants or []),
            'data_release_id': cfg.get('data_release_id'),
            'source_manifest_sha256': cfg.get('source_manifest_sha256'),
            'model_revision': cfg.get('model_revision'),
            'tokenizer_backend': cfg.get('tokenizer_backend', 'fast'),
            'tokenizer_normalization': cfg.get('tokenizer_normalization', 'default'),
            'input_preprocessing': cfg.get('input_preprocessing', 'none'),
            'mixed_precision': cfg.get('mixed_precision', 'auto'),
            'trained_checkpoint_split': cfg.get('reuse_trained_from_split') or eval_split,
        },
    )
    return output_dir


def evaluate_reused_checkpoint(
    *, cfg, p, approach, source_approach, system_id, source_system_id,
    experiment, dataset, seed, output_dir, eval_split, eval_role,
    eval_variants, eval_sets_dir,
):
    """Evaluate a primary-split checkpoint without repeating model training."""

    source_split = cfg.get('reuse_trained_from_split')
    if not source_split or source_split == eval_split:
        return False
    source_dir = scoped_run_dir(
        p, cfg['prefix'], source_approach, eval_role, source_split
    )
    validate_completed_run(
        source_dir, system_id=source_system_id, require_model=True,
        require_prediction_identity=True,
        expected_manifest={
            'config_sha256': config_hash(experiment),
            'system_id': source_system_id,
            'dataset': dataset,
            'seed': seed,
            'learning_rate': cfg['lr'],
            'run_mode': cfg.get('run_mode', 'production'),
            'reported_evaluation_role': eval_role,
            'reported_evaluation_split': source_split,
            'evaluated_variants': sorted(eval_variants or []),
            'data_release_id': cfg.get('data_release_id'),
            'source_manifest_sha256': cfg.get('source_manifest_sha256'),
            'model_revision': cfg.get('model_revision'),
            'tokenizer_backend': cfg.get('tokenizer_backend', 'fast'),
            'tokenizer_normalization': cfg.get(
                'tokenizer_normalization', 'default'
            ),
            'input_preprocessing': cfg.get('input_preprocessing', 'none'),
            'mixed_precision': cfg.get('mixed_precision', 'auto'),
            'trained_checkpoint_split': source_split,
        },
    )
    command = [
        sys.executable, 'src/evaluate_checkpoint.py',
        '--source_run_dir', str(source_dir),
        '--output_dir', str(output_dir),
        '--model_name', cfg['model_name'],
        '--dataset', dataset,
        '--system_id', system_id,
        '--eval_sets_dir', str(eval_sets_dir),
        '--eval_split', str(eval_split),
        '--eval_role', eval_role,
        '--eval_variants', *(eval_variants or ['all']),
        '--batch_size', str(cfg['batch_size'] * 2),
        '--max_length', str(experiment['training']['max_length']),
        '--preprocessing_workers', str(cfg.get('preprocessing_workers', 1)),
        '--tokenizer_backend', cfg.get('tokenizer_backend', 'fast'),
        '--tokenizer_normalization', cfg.get('tokenizer_normalization', 'default'),
        '--input_preprocessing', cfg.get('input_preprocessing', 'none'),
        '--mixed_precision', cfg.get('mixed_precision', 'auto'),
        '--save_logits',
    ]
    if cfg.get('model_revision'):
        command += ['--model_revision', str(cfg['model_revision'])]
    if cfg.get('cache_dir'):
        command += ['--cache_dir', str(cfg['cache_dir'])]
    run(
        command,
        f'{cfg["prefix"].upper()} — {approach} evaluation reuse '
        f'({source_split} checkpoint -> {eval_split})',
        output_dir,
    )
    return True


def step_baseline(cfg, p, common, experiment, config_path, dataset, seed, eval_split=None,
                  eval_role='final', eval_variants=None, marker_fold='fold_1'):
    del marker_fold
    approach = 'baseline'
    output_dir = materialize_manifest(
        p, approach, cfg, experiment, config_path, dataset, seed, eval_split, eval_role,
        eval_variants, system_id='clean_baseline'
    )
    if completed_run_exists(output_dir):
        validate_completed_run(
            output_dir, system_id='clean_baseline',
            require_model=not bool(cfg.get('reuse_trained_from_split')),
            require_prediction_identity=cfg.get('run_mode') != 'smoke',
        )
        print(f'SKIP: verified completed baseline at {output_dir}')
        return
    if evaluate_reused_checkpoint(
        cfg=cfg, p=p, approach=approach, source_approach=approach,
        system_id='clean_baseline', source_system_id='clean_baseline',
        experiment=experiment, dataset=dataset, seed=seed,
        output_dir=output_dir, eval_split=eval_split, eval_role=eval_role,
        eval_variants=eval_variants, eval_sets_dir=cfg.get('eval_sets_dir'),
    ):
        return
    run(
        [sys.executable, 'src/train_baseline.py',
         '--output_dir', str(output_dir), '--save_logits'] + common,
        f'{cfg["prefix"].upper()} — Baseline', output_dir
    )


def condition_flags(experiment, marker_fold):
    fold = next(
        item for item in experiment['transformations']['marker_folds']
        if item['id'] == marker_fold
    )
    return [
        '--augment_ratio', str(experiment['training']['augmentation_probability']),
        '--marker_placements', *experiment['transformations']['marker_placements'],
        '--marker_choices', *fold['train'],
    ]


def step_augmentation(cfg, p, common, experiment, config_path, dataset, seed,
                      eval_split=None, marker_fold='fold_1', eval_role='final',
                      eval_variants=None):
    output_dir = materialize_manifest(
        p, f'augmented_{marker_fold}', cfg, experiment, config_path, dataset, seed,
        eval_split, eval_role, eval_variants, system_id='marker_augmentation'
    )
    if completed_run_exists(output_dir):
        validate_completed_run(
            output_dir, system_id='marker_augmentation',
            require_model=not bool(cfg.get('reuse_trained_from_split')),
            require_prediction_identity=cfg.get('run_mode') != 'smoke',
        )
        print(f'SKIP: verified completed augmentation at {output_dir}')
        return
    if evaluate_reused_checkpoint(
        cfg=cfg, p=p, approach=f'augmented_{marker_fold}',
        source_approach=f'augmented_{marker_fold}',
        system_id='marker_augmentation', source_system_id='marker_augmentation',
        experiment=experiment, dataset=dataset, seed=seed,
        output_dir=output_dir, eval_split=eval_split, eval_role=eval_role,
        eval_variants=eval_variants, eval_sets_dir=cfg.get('eval_sets_dir'),
    ):
        return
    run(
        [sys.executable, 'src/train_augmentation.py',
         '--output_dir', str(output_dir), '--save_logits']
        + condition_flags(experiment, marker_fold) + common,
        f'{cfg["prefix"].upper()} — Augmentation', output_dir
    )


def step_preprocessing(cfg, training, p, dataset, seed, smoke_test, eval_sets_dir,
                       experiment, config_path, eval_split=None, eval_variants=None,
                       eval_role='final', mode='emoji', marker_fold='fold_1'):
    smoke_flag = (
        ['--smoke_test', '--smoke_eval_samples',
         str(cfg.get('smoke_eval_samples', 100))]
        if smoke_test else []
    )
    eval_sets_flag = ['--eval_sets_dir', str(eval_sets_dir)] if eval_sets_dir else []
    if eval_sets_flag and eval_split:
        eval_sets_flag += ['--eval_split', eval_split]
    if eval_sets_flag:
        eval_sets_flag += ['--eval_role', eval_role]
    if eval_sets_flag and eval_variants:
        eval_sets_flag += ['--eval_variants', *eval_variants]
    prefix = cfg['prefix']
    model_path = scoped_run_dir(
        p, prefix, 'baseline', eval_role, eval_split or 'manifest_default'
    ) / 'final_model'
    approach_base = 'preprocessing' if mode == 'emoji' else 'marker_oracle'
    del marker_fold
    approach = approach_base
    system_id = ('emoji_normalization' if mode == 'emoji'
                 else 'known_marker_deletion_oracle')
    output_dir = materialize_manifest(
        p, approach, cfg, experiment, config_path, dataset, seed, eval_split,
        eval_role, eval_variants, system_id=system_id
    )
    if completed_run_exists(output_dir):
        validate_completed_run(
            output_dir, system_id=system_id,
            require_prediction_identity=cfg.get('run_mode') != 'smoke',
        )
        print(f'SKIP: verified completed {approach_base} at {output_dir}')
        return
    if evaluate_reused_checkpoint(
        cfg=cfg, p=p, approach=approach, source_approach='baseline',
        system_id=system_id, source_system_id='clean_baseline',
        experiment=experiment, dataset=dataset, seed=seed,
        output_dir=output_dir, eval_split=eval_split, eval_role=eval_role,
        eval_variants=eval_variants, eval_sets_dir=eval_sets_dir,
    ):
        return
    baseline_dir = Path(model_path).parent
    try:
        validate_completed_run(
            baseline_dir, system_id='clean_baseline', require_model=True,
            require_prediction_identity=cfg.get('run_mode') != 'smoke',
            expected_manifest={
                'config_sha256': config_hash(experiment),
                'system_id': 'clean_baseline',
                'dataset': dataset,
                'seed': seed,
                'learning_rate': cfg['lr'],
                'run_mode': cfg.get('run_mode', 'production'),
                'reported_evaluation_role': eval_role,
                'reported_evaluation_split': eval_split or 'manifest_default',
                'evaluated_variants': sorted(eval_variants or []),
                'data_release_id': cfg.get('data_release_id'),
                'source_manifest_sha256': cfg.get('source_manifest_sha256'),
                'model_revision': cfg.get('model_revision'),
                'tokenizer_backend': cfg.get('tokenizer_backend', 'fast'),
                'tokenizer_normalization': cfg.get(
                    'tokenizer_normalization', 'default'
                ),
                'input_preprocessing': cfg.get('input_preprocessing', 'none'),
                'mixed_precision': cfg.get('mixed_precision', 'auto'),
                'trained_checkpoint_split': (
                    cfg.get('reuse_trained_from_split') or eval_split
                ),
            },
        )
    except RuntimeError as error:
        message = f'baseline dependency is not reusable: {error}'
        record_run_attempt(output_dir, {
            'label': f'{prefix.upper()} — {approach_base}',
            'status': 'blocked', 'exit_code': 1,
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'runtime_seconds': 0.0, 'command': [], 'error': message,
        })
        print(f'ERROR: {message}')
        sys.exit(1)
    run(
        [sys.executable, 'src/train_preprocessing.py',
         '--model_path', model_path,
         '--model_name', cfg['model_name'],
         '--dataset', dataset,
         '--mode', mode,
         '--output_dir', str(output_dir),
         '--batch_size', str(cfg['batch_size'] * 2),
         '--max_length', str(training['max_length']),
         '--preprocessing_workers', str(cfg.get('preprocessing_workers', 1)),
         '--tokenizer_backend', cfg.get('tokenizer_backend', 'fast'),
         '--tokenizer_normalization', cfg.get('tokenizer_normalization', 'default'),
         '--input_preprocessing', cfg.get('input_preprocessing', 'none'),
         '--mixed_precision', cfg.get('mixed_precision', 'auto'),
         '--development_seed', str(
             experiment['dataset_protocol'].get(dataset, {}).get('development_seed', 42)
         ),
         '--frozen_run',
         '--save_logits'] + smoke_flag + eval_sets_flag
        + (['--data_release', str(cfg['data_release'])]
           if cfg.get('data_release') else [])
        + (['--cache_dir', str(cfg['cache_dir'])] if cfg.get('cache_dir') else [])
        + (['--model_revision', str(cfg['model_revision'])]
           if cfg.get('model_revision') else []),
        f'{prefix.upper()} — {"Emoji normalization" if mode == "emoji" else "Marker deletion oracle"}',
        output_dir,
    )


def step_hybrid(cfg, p, common, experiment, config_path, dataset, seed,
                eval_split=None, marker_fold='fold_1', eval_role='final',
                eval_variants=None):
    output_dir = materialize_manifest(
        p, f'hybrid_{marker_fold}', cfg, experiment, config_path, dataset, seed,
        eval_split, eval_role, eval_variants, system_id='stage_matched_hybrid'
    )
    if completed_run_exists(output_dir):
        validate_completed_run(
            output_dir, system_id='stage_matched_hybrid',
            require_model=not bool(cfg.get('reuse_trained_from_split')),
            require_prediction_identity=cfg.get('run_mode') != 'smoke',
        )
        print(f'SKIP: verified completed hybrid at {output_dir}')
        return
    if evaluate_reused_checkpoint(
        cfg=cfg, p=p, approach=f'hybrid_{marker_fold}',
        source_approach=f'hybrid_{marker_fold}',
        system_id='stage_matched_hybrid', source_system_id='stage_matched_hybrid',
        experiment=experiment, dataset=dataset, seed=seed,
        output_dir=output_dir, eval_split=eval_split, eval_role=eval_role,
        eval_variants=eval_variants, eval_sets_dir=cfg.get('eval_sets_dir'),
    ):
        return
    run(
        [sys.executable, 'src/train_hybrid.py',
         '--output_dir', str(output_dir), '--save_logits']
        + condition_flags(experiment, marker_fold) + common,
        f'{cfg["prefix"].upper()} — Hybrid', output_dir
    )


def step_clean_control(cfg, p, common, experiment, config_path, dataset, seed,
                       eval_split=None, eval_role='final', eval_variants=None,
                       marker_fold='fold_1'):
    del marker_fold
    approach = 'clean_control'
    output_dir = materialize_manifest(
        p, approach, cfg, experiment, config_path, dataset, seed,
        eval_split, eval_role, eval_variants,
        system_id='compute_matched_clean_control',
    )
    if completed_run_exists(output_dir):
        validate_completed_run(
            output_dir, system_id='compute_matched_clean_control',
            require_model=not bool(cfg.get('reuse_trained_from_split')),
            require_prediction_identity=cfg.get('run_mode') != 'smoke',
        )
        print(f'SKIP: verified completed clean control at {output_dir}')
        return
    if evaluate_reused_checkpoint(
        cfg=cfg, p=p, approach=approach, source_approach=approach,
        system_id='compute_matched_clean_control',
        source_system_id='compute_matched_clean_control',
        experiment=experiment, dataset=dataset, seed=seed,
        output_dir=output_dir, eval_split=eval_split, eval_role=eval_role,
        eval_variants=eval_variants, eval_sets_dir=cfg.get('eval_sets_dir'),
    ):
        return
    run(
        [sys.executable, 'src/train_augmentation.py',
         '--output_dir', str(output_dir), '--save_logits',
         '--system', 'compute_matched_clean_control'] + common,
        f'{cfg["prefix"].upper()} — Compute-matched clean control', output_dir,
    )


def step_stats(cfg, p, marker_fold='fold_1', eval_role='final', eval_split=None):
    run(
        [sys.executable, 'src/statistical_tests.py',
         '--results_dir', p,
         '--model_prefix', cfg['prefix'], '--marker_fold', marker_fold,
         '--eval_role', eval_role, '--eval_split', str(eval_split)],
        f'{cfg["prefix"].upper()} — Statistical Tests'
    )


def validate_smoke_matrix(p, cfg, marker_fold, eval_role, eval_split):
    """Fail unless the normal limited-data route completed every system."""

    expected = (
        ('baseline', 'clean_baseline', True),
        (f'augmented_{marker_fold}', 'marker_augmentation', True),
        ('preprocessing', 'emoji_normalization', False),
        (f'hybrid_{marker_fold}', 'stage_matched_hybrid', True),
        ('marker_oracle', 'known_marker_deletion_oracle', False),
        ('clean_control', 'compute_matched_clean_control', True),
    )
    for approach, system_id, require_model in expected:
        output_dir = scoped_run_dir(
            p, cfg['prefix'], approach, eval_role, eval_split or 'manifest_default'
        )
        validate_completed_run(
            output_dir, system_id=system_id, require_model=require_model,
            require_prediction_identity=False,
        )
        if not (output_dir / 'raw_logits.json').is_file():
            raise RuntimeError(f'Limited-data run is missing logits: {output_dir}')
    print('SMOKE VERIFIED: all six limited-data systems and artifacts completed.')


def run_step(step, cfg, experiment, config_path, out_dir, dataset, seed,
             smoke_test, eval_sets_dir, eval_split=None, marker_fold='fold_1',
             eval_variants=None, eval_role='final', smoke_train_samples=100,
             smoke_eval_samples=100):
    run_mode = 'smoke' if smoke_test else 'production'
    p = str(Path(out_dir) / 'smoke') if smoke_test else str(out_dir)
    cfg = dict(
        cfg, run_mode=run_mode, eval_sets_dir=eval_sets_dir,
        smoke_train_samples=smoke_train_samples,
        smoke_eval_samples=smoke_eval_samples,
    )
    training = experiment['training']
    common = build_common(
        cfg, training, dataset, seed, smoke_test, eval_sets_dir, eval_split,
        eval_variants,
        experiment['dataset_protocol'].get(dataset, {}).get('development_seed', 42),
        eval_role,
        cfg.get('data_release'), cfg.get('cache_dir'), cfg.get('model_revision'),
        cfg.get('preprocessing_workers', 1),
        smoke_train_samples, smoke_eval_samples,
        cfg.get('train_size', 'full'),
    )

    if step in ('baseline', 'all'):
        step_baseline(
            cfg, p, common, experiment, config_path, dataset, seed, eval_split, eval_role,
            eval_variants, marker_fold
        )
    if step in ('augmentation', 'all'):
        step_augmentation(
            cfg, p, common, experiment, config_path, dataset, seed, eval_split,
            marker_fold, eval_role, eval_variants
        )
    if step in ('preprocessing', 'all'):
        step_preprocessing(cfg, training, p, dataset, seed, smoke_test, eval_sets_dir,
                           experiment, config_path, eval_split, eval_variants, eval_role,
                           marker_fold=marker_fold)
    if step in ('hybrid', 'all'):
        step_hybrid(
            cfg, p, common, experiment, config_path, dataset, seed, eval_split,
            marker_fold, eval_role, eval_variants
        )
    if step in ('marker_oracle', 'all'):
        step_preprocessing(
            cfg, training, p, dataset, seed, smoke_test, eval_sets_dir,
            experiment, config_path, eval_split, eval_variants, eval_role,
            mode='marker_oracle', marker_fold=marker_fold,
        )
    if step in ('clean_control', 'all'):
        step_clean_control(
            cfg, p, common, experiment, config_path, dataset, seed, eval_split,
            eval_role, eval_variants, marker_fold,
        )
    if step == 'stats' or (step == 'all' and not smoke_test):
        step_stats(cfg, p, marker_fold, eval_role, eval_split)
    elif step == 'all' and smoke_test:
        validate_smoke_matrix(p, cfg, marker_fold, eval_role, eval_split)
        print('\nLimited-data integration run completed; inferential statistics were '
              'intentionally skipped.')

    print(f'\nDone. Results in: {p}/')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH),
                        help='Frozen experiment JSON. Defaults to configs/experiment_v6.json.')
    parser.add_argument('--model', type=str, default='electra',
                        help='Configured model key, "both", or "all".')
    parser.add_argument('--step', type=str, default='all', choices=STEPS,
                        help='Which experiment to run. Use "all" to run every step sequentially.')
    parser.add_argument('--out', type=str, default='./results',
                        help='Output directory for checkpoints and evaluation artifacts.')
    parser.add_argument('--dataset', type=str, default='snli',
                        choices=['snli', 'multi_nli'])
    parser.add_argument('--eval_sets_dir', type=str, default=None,
                        help='Path to pre-generated eval sets. Required except for --step stats.')
    parser.add_argument('--data_release', type=str, default=None,
                        help='Immutable release built by prepare_data_release.py.')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Optional writable cache, separate from the immutable data release.')
    parser.add_argument('--model_revision', type=str, default=None,
                        help='Pinned Hugging Face model/tokenizer commit SHA.')
    parser.add_argument('--preprocessing_workers', type=int, default=1)
    parser.add_argument('--eval_split', type=str, default=None,
                        help='Named final split (for MultiNLI: validation_matched or '
                             'validation_mismatched).')
    parser.add_argument('--eval_role', choices=['development', 'final'], default='final')
    parser.add_argument(
        '--reuse_trained_from_split', default=None,
        help='Evaluate a checkpoint trained on this split instead of retraining.',
    )
    parser.add_argument('--marker_fold', type=str, default='fold_1',
                        choices=['fold_1', 'fold_2', 'fold_3'],
                        help='Marker cross-validation fold for augmentation/hybrid training.')
    parser.add_argument('--eval_variants', nargs='+', default=None,
                        help='Variant names, or "all" to score every generated evaluation condition.')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run a real miniature end-to-end train/eval route.')
    parser.add_argument('--sample_limit', type=int, default=None,
                        help='Use this many frozen rows for both train and eval smoke picks.')
    parser.add_argument('--smoke_train_samples', type=int, default=100)
    parser.add_argument('--smoke_eval_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed override. Defaults to training.primary_seed.')
    parser.add_argument('--train_size', type=parse_train_size, default='full',
                        help='Number of training pairs: "full" or an integer. Subset '
                             'runs are written below train_size_<N>/.')
    parser.add_argument('--all_seeds', action='store_true',
                        help='Run every frozen seed for each selected model.')
    parser.add_argument('--learning_rate_sweep', action='store_true',
                        help='Run the model-specific frozen LR sweep on development data.')
    parser.add_argument('--show_config', action='store_true',
                        help='Validate and print the resolved frozen configuration, then exit.')
    args = parser.parse_args()

    if args.smoke_test and args.step == 'stats':
        parser.error('--step stats cannot use --smoke_test; statistics require fixed eval sets')
    if args.sample_limit is not None:
        if not args.smoke_test:
            parser.error('--sample_limit requires --smoke_test')
        args.smoke_train_samples = args.sample_limit
        args.smoke_eval_samples = args.sample_limit
    if args.smoke_train_samples < 1 or args.smoke_eval_samples < 1:
        parser.error('--smoke_train_samples and --smoke_eval_samples must be positive')

    config_path = Path(args.config)
    experiment = load_experiment_config(config_path)
    if args.show_config:
        print(json.dumps(experiment, indent=2, ensure_ascii=False))
        print(f'\nSHA-256: {config_hash(experiment)}')
        return

    configured_models = sorted(experiment['models'])
    if args.model == 'all':
        model_keys = configured_models
    elif args.model == 'both':
        model_keys = [name for name in ('electra', 'roberta') if name in experiment['models']]
    elif args.model in experiment['models']:
        model_keys = [args.model]
    else:
        parser.error(
            f'--model must be one of {configured_models} plus "both" or "all"'
        )
    if args.seed is not None and args.all_seeds:
        parser.error('--seed and --all_seeds are mutually exclusive')
    if args.learning_rate_sweep and args.eval_role != 'development':
        parser.error('--learning_rate_sweep is restricted to --eval_role development')
    if args.dataset not in experiment['dataset_protocol']:
        parser.error(f'--dataset is not configured: {args.dataset}')
    if args.reuse_trained_from_split:
        final_splits = experiment['dataset_protocol'][args.dataset]['final_splits']
        if args.eval_role != 'final':
            parser.error('--reuse_trained_from_split is restricted to final evaluation')
        if args.reuse_trained_from_split not in final_splits:
            parser.error(
                f'--reuse_trained_from_split must be one of {final_splits}'
            )

    needs_eval_sets = args.step in (
        'baseline', 'augmentation', 'preprocessing', 'hybrid',
        'marker_oracle', 'clean_control', 'all',
    )
    if args.data_release and not args.eval_sets_dir:
        args.eval_sets_dir = str(Path(args.data_release) / 'eval_sets')
    if needs_eval_sets and not args.smoke_test and not args.data_release:
        print('ERROR: --data_release is required for paper/production runs.')
        print('Generate once with: python src/prepare_data_release.py --out <versioned-dir>')
        sys.exit(1)
    if needs_eval_sets and not args.smoke_test and not args.eval_sets_dir:
        print('ERROR: --eval_sets_dir is required for this step.')
        print('Generate once with:  python src/prepare_eval_sets.py --out <dir>')
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_sets_dir = Path(args.eval_sets_dir) if args.eval_sets_dir else None
    release_metadata = {}
    if args.data_release:
        release_path = Path(args.data_release)
        release_manifest_path = release_path / 'release_manifest.json'
        if not release_manifest_path.is_file():
            raise SystemExit(f'Invalid data release: {release_manifest_path} is missing')
        release_manifest = json.loads(release_manifest_path.read_text(encoding='utf-8'))
        if release_manifest.get('data_contract_sha256') != data_contract_hash(experiment):
            raise SystemExit(
                'Data release was generated from a different dataset/transformation contract'
            )
        release_metadata = {
            'data_release': str(release_path.resolve()),
            'data_release_id': release_manifest['release_id'],
        }
    if args.cache_dir:
        cache_root = Path(args.cache_dir).resolve()
        if args.data_release and cache_root == Path(args.data_release).resolve():
            raise SystemExit('--cache_dir must not be the immutable data release directory')
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('HF_DATASETS_CACHE', str(cache_root / 'datasets'))
    if args.eval_split:
        resolved_eval_split = args.eval_split
    elif args.eval_role == 'development':
        resolved_eval_split = 'validation' if args.dataset == 'snli' else 'train_holdout'
    else:
        resolved_eval_split = experiment['dataset_protocol'][args.dataset]['final_splits'][0]
    if args.eval_variants:
        resolved_eval_variants = args.eval_variants
    elif args.smoke_test and args.data_release:
        resolved_eval_variants = [
            'emoji_raw',
            f'marker_unseen_{args.marker_fold}_hypothesis_suffix',
            f'emoji_marker_combined_{args.marker_fold}',
        ]
    else:
        resolved_eval_variants = ['all']

    for model_key in model_keys:
        cfg = model_config(experiment, model_key)
        revision = args.model_revision or cfg.get('revision')
        identity = source_dataset_identity(args.data_release, args.dataset) if args.data_release else {}
        cfg.update({
            **release_metadata,
            'source_manifest_sha256': identity.get('source_manifest_sha256'),
            'cache_dir': args.cache_dir,
            'model_revision': revision,
            'preprocessing_workers': args.preprocessing_workers,
            'reuse_trained_from_split': args.reuse_trained_from_split,
            'train_size': args.train_size,
        })
        if args.all_seeds:
            seeds = cfg['seeds']
        else:
            seeds = [args.seed if args.seed is not None else experiment['training']['primary_seed']]
        learning_rates = cfg['learning_rate_sweep'] if args.learning_rate_sweep else [cfg['lr']]
        for learning_rate in learning_rates:
            resolved_cfg = dict(cfg, lr=learning_rate)
            for seed in seeds:
                run_out = out_dir / args.dataset / f'seed_{seed}'
                if args.train_size != 'full':
                    run_out = run_out / f'train_size_{args.train_size}'
                if args.learning_rate_sweep:
                    run_out = run_out / f'lr_{learning_rate:.8g}'
                run_step(
                    args.step, resolved_cfg, experiment, config_path, run_out,
                    args.dataset, seed, args.smoke_test, eval_sets_dir,
                    resolved_eval_split, args.marker_fold, resolved_eval_variants,
                    args.eval_role,
                    args.smoke_train_samples, args.smoke_eval_samples,
                )


if __name__ == '__main__':
    main()
