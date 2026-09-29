"""Evaluate one already-trained system on another frozen evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np
from transformers import (AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from helpers import (attach_condition_metadata, compute_accuracy, load_eval_suite,
                     map_dataset_locally,
                     paired_clean_result, prepare_dataset_nli,
                     psych_inversion_result, robustness_drops,
                     write_prediction_manifest)
from preprocessing import preprocess_example
from training_runtime import (load_tokenizer, mixed_precision_kwargs,
                              tokenization_cache_operation,
                              assert_float32_parameters, model_pretrained_kwargs)
from training_metadata import runtime_environment, system_stage_metadata


PREPROCESS_MODES = {
    'clean_baseline': None,
    'marker_augmentation': None,
    'compute_matched_clean_control': None,
    'stage_matched_hybrid': 'stage_matched',
    'emoji_normalization': 'emoji',
    'known_marker_deletion_oracle': 'marker_oracle',
}
APPROACHES = {
    'clean_baseline': 'baseline',
    'marker_augmentation': 'augmentation',
    'compute_matched_clean_control': 'clean_control',
    'stage_matched_hybrid': 'hybrid',
    'emoji_normalization': 'preprocessing',
    'known_marker_deletion_oracle': 'marker_oracle',
}


def stage_model_locally(source_run: Path, cache_dir: str | None) -> Path:
    """Copy a Drive-hosted final model to local SSD once per worker."""

    source_model = source_run / 'final_model'
    if not cache_dir:
        return source_model
    from filelock import FileLock

    identity = hashlib.sha256(
        (str(source_run.resolve()) + '\0').encode('utf-8')
        + (source_run / 'run_manifest.json').read_bytes()
    ).hexdigest()
    root = Path(cache_dir) / 'staged_models'
    destination = root / identity
    marker = destination / '.complete'
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + '.lock'):
        if marker.is_file():
            return destination
        temporary = root / f'.{identity}.{uuid.uuid4().hex}.tmp'
        shutil.copytree(source_model, temporary)
        (temporary / '.complete').write_text(
            str(source_run.resolve()) + '\n', encoding='utf-8'
        )
        temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_revision', default=None)
    parser.add_argument('--dataset', required=True, choices=['snli', 'multi_nli'])
    parser.add_argument('--system_id', required=True, choices=sorted(PREPROCESS_MODES))
    parser.add_argument('--eval_sets_dir', required=True)
    parser.add_argument('--eval_split', required=True)
    parser.add_argument('--eval_role', default='final', choices=['development', 'final'])
    parser.add_argument('--eval_variants', nargs='+', default=['all'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--preprocessing_workers', type=int, default=1)
    parser.add_argument('--tokenizer_backend', choices=['fast', 'slow'], default='fast')
    parser.add_argument('--tokenizer_normalization',
                        choices=['default', 'enabled', 'disabled', 'spm_byte_fallback'],
                        default='default')
    parser.add_argument('--input_preprocessing', choices=['none', 'timelm'], default='none')
    parser.add_argument('--mixed_precision', choices=['auto', 'no', 'fp16', 'bf16'],
                        default='auto')
    parser.add_argument('--save_logits', action='store_true')
    args = parser.parse_args()

    import torch

    source = Path(args.source_run_dir)
    source_model_path = source / 'final_model'
    if not source_model_path.is_dir():
        raise FileNotFoundError(f'Trained checkpoint is missing: {source_model_path}')
    source_results = json.loads((source / 'results.json').read_text(encoding='utf-8'))
    if source_results.get('system_id') not in {
        args.system_id,
        'clean_baseline' if args.system_id in {
            'emoji_normalization', 'known_marker_deletion_oracle'
        } else args.system_id,
    }:
        raise ValueError('Source checkpoint system does not match requested evaluation')

    base = Path(args.eval_sets_dir) / args.dataset
    datasets = load_eval_suite(
        base, variants=args.eval_variants,
        split_role=args.eval_role, split_name=args.eval_split,
        purpose=('final_evaluation' if args.eval_role == 'final'
                 else 'development_evaluation'),
    )
    mode = PREPROCESS_MODES[args.system_id]
    if mode:
        datasets = {
            name: map_dataset_locally(
                rows, lambda row, _mode=mode: preprocess_example(row, mode=_mode),
                cache_dir=args.cache_dir,
                operation=f'preprocess:{mode}:{name}',
                desc=f'Preprocessing {name}',
                num_proc=args.preprocessing_workers,
            )
            for name, rows in datasets.items()
        }

    tokenizer = load_tokenizer(
        args.model_name, args.model_revision, backend=args.tokenizer_backend,
        normalization=args.tokenizer_normalization,
    )
    model_path = stage_model_locally(source, args.cache_dir)
    print(f'Loading trained checkpoint from local path: {model_path}', flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), **model_pretrained_kwargs(None)
    )
    parameter_dtype = assert_float32_parameters(model, str(model_path))
    print(f'Loaded {model_path} with {parameter_dtype} parameters', flush=True)
    trainer_identity = hashlib.sha256(
        str(Path(args.output_dir).resolve()).encode('utf-8')
    ).hexdigest()[:16]
    trainer_output = (
        Path(args.cache_dir or tempfile.gettempdir()) / 'nli_evaluation_trainers' / trainer_identity
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(trainer_output),
            per_device_eval_batch_size=args.batch_size,
            **mixed_precision_kwargs(torch, args.mixed_precision),
            report_to='none', disable_tqdm=True,
        ),
        compute_metrics=compute_accuracy,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    results = {}
    started = time.monotonic()
    for position, (name, raw) in enumerate(datasets.items(), start=1):
        print(
            f'EVAL_PROGRESS variant={position}/{len(datasets)} name={name}',
            flush=True,
        )
        if name.startswith('clean_'):
            results[name] = paired_clean_result(
                results, name, raw, save_logits=args.save_logits
            )
            print('EVAL_PROGRESS indexed_from_original=true', flush=True)
            continue
        source_indices = (
            list(raw['source_index']) if 'source_index' in raw.column_names
            else list(range(len(raw)))
        )
        tokenized = map_dataset_locally(
            raw,
            lambda batch: prepare_dataset_nli(
                batch, tokenizer, args.max_length, args.input_preprocessing
            ),
            batched=True, remove_columns=raw.column_names,
            cache_dir=args.cache_dir,
            operation=tokenization_cache_operation(
                args.model_name, args.model_revision, args.max_length,
                f'{mode}:evaluation:{name}',
                tokenizer_backend=args.tokenizer_backend,
                tokenizer_normalization=args.tokenizer_normalization,
                input_preprocessing=args.input_preprocessing,
            ),
            desc=f'Tokenizing {name}', num_proc=args.preprocessing_workers,
        )
        prediction = trainer.predict(tokenized)
        predicted = np.argmax(prediction.predictions, axis=1).tolist()
        labels = prediction.label_ids.tolist()
        value = {
            'accuracy': float(np.mean(np.asarray(predicted) == np.asarray(labels))),
            'predictions': predicted, 'labels': labels,
            'source_indices': source_indices,
        }
        if args.save_logits:
            value['logits'] = prediction.predictions.tolist()
        results[name] = value
    runtime = time.monotonic() - started
    attach_condition_metadata(results)

    source_metadata = source_results.get('run_metadata', {})
    if args.system_id in {'emoji_normalization', 'known_marker_deletion_oracle'}:
        run_metadata = {
            **system_stage_metadata(args.system_id),
            **runtime_environment(torch),
            'run_origin': 'frozen_config',
            'trained_model_system_id': 'clean_baseline',
            'actual_optimizer_steps': 0,
            'runtime_scope': 'evaluation_only_reusing_frozen_checkpoint',
            'runtime_seconds': runtime,
            'training_source_run': str(source.resolve()),
            'provenance': source_metadata.get('provenance', {}),
        }
    else:
        run_metadata = dict(source_metadata)
        run_metadata.update({
            'evaluation_runtime_seconds': runtime,
            'evaluation_reused_checkpoint': True,
            'training_source_run': str(source.resolve()),
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        'model_name': args.model_name,
        'dataset': args.dataset,
        'approach': APPROACHES[args.system_id],
        'system_id': args.system_id,
        'config': vars(args),
        'train_samples': source_results.get('train_samples'),
        'training_subset': source_results.get('training_subset'),
        'actual_optimizer_steps': source_results.get('actual_optimizer_steps', 0),
        'run_metadata': run_metadata,
        'accuracies': {name: value['accuracy'] for name, value in results.items()},
        'drops_pp': robustness_drops(results),
        'psych_instruction_inversion': psych_inversion_result(results),
        'condition_metadata': {
            name: value['condition_metadata'] for name, value in results.items()
        },
    }
    (output_dir / 'results.json').write_text(
        json.dumps(output, indent=2) + '\n', encoding='utf-8'
    )
    prediction_rows = {
        name: {key: value[key] for key in (
            'predictions', 'labels', 'source_indices', 'condition_metadata'
        )}
        for name, value in results.items()
    }
    (output_dir / 'predictions.json').write_text(
        json.dumps(prediction_rows) + '\n', encoding='utf-8'
    )
    write_prediction_manifest(
        output_dir, base, split_role=args.eval_role, split_name=args.eval_split,
        purpose=('final_evaluation' if args.eval_role == 'final'
                 else 'development_evaluation'),
        variants=prediction_rows.keys(),
    )
    if args.save_logits:
        (output_dir / 'raw_logits.json').write_text(
            json.dumps({name: {'logits': value['logits']}
                        for name, value in results.items()}) + '\n',
            encoding='utf-8',
        )
    print(f'Evaluation-only results saved to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
