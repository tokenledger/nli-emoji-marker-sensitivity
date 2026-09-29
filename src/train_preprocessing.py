"""
Preprocessing evaluation: load the saved baseline model, apply text normalization
before inference (no retraining). Tests whether deterministic cleanup recovers accuracy.

Requires: results/{model}_baseline/final_model to exist (run train_baseline.py first).

Usage:
    python src/train_preprocessing.py --model_path ./results/electra_baseline/final_model
    python src/train_preprocessing.py --model_path ./results/roberta_baseline/final_model \
        --model_name roberta-large --output_dir ./results/roberta_preprocessing
"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from helpers import (attach_condition_metadata, compute_accuracy, limit_dataset,
                     limit_eval_suite,
                     load_eval_suite, load_frozen_development_data,
                     recorded_training_subset,
                     map_dataset_locally, paired_clean_result,
                     prepare_dataset_nli,
                     source_dataset_identity,
                     psych_inversion_result, robustness_drops,
                     smoke_nli_splits, write_prediction_manifest)
from transforms import transform_example
from preprocessing import preprocess_example
from training_metadata import runtime_environment, system_stage_metadata
from training_runtime import (load_tokenizer, mixed_precision_kwargs,
                              model_source_metadata, tokenization_cache_operation,
                              assert_float32_parameters, model_pretrained_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./results/electra_baseline/final_model')
    parser.add_argument('--model_name', type=str,
                        default='google/electra-small-discriminator',
                        help='Must match the tokenizer used during baseline training')
    parser.add_argument('--model_revision', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='snli', choices=['snli', 'multi_nli'])
    parser.add_argument('--output_dir', type=str, default='./results/electra_preprocessing')
    parser.add_argument(
        '--mode', choices=['emoji', 'marker_oracle'], default='emoji',
        help='Inference-only condition; marker_oracle is explicitly non-principal.',
    )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--development_seed', type=int, default=42)
    parser.add_argument('--smoke_test', action='store_true',
                        help='Limited-data end-to-end check; uses frozen data when supplied.')
    parser.add_argument('--eval_sets_dir', type=str, default=None,
                        help='Path to pre-generated eval sets from prepare_eval_sets.py.')
    parser.add_argument('--data_release', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Optional writable cache, separate from the immutable data release.')
    parser.add_argument('--preprocessing_workers', type=int, default=1)
    parser.add_argument('--tokenizer_backend', choices=['fast', 'slow'], default='fast')
    parser.add_argument('--tokenizer_normalization',
                        choices=['default', 'enabled', 'disabled', 'spm_byte_fallback'],
                        default='default')
    parser.add_argument('--input_preprocessing', choices=['none', 'timelm'], default='none')
    parser.add_argument('--mixed_precision', choices=['auto', 'no', 'fp16', 'bf16'],
                        default='auto')
    parser.add_argument('--smoke_eval_samples', type=int, default=100)
    parser.add_argument('--eval_split', type=str, default=None,
                        help='Named final split; defaults to the dataset manifest primary split.')
    parser.add_argument('--eval_role', choices=['development', 'final'], default='final')
    parser.add_argument('--eval_variants', nargs='+', default=None,
                        help='Variant names, or "all" for every generated evaluation condition.')
    parser.add_argument('--save_logits', action='store_true',
                        help='Save raw logits to raw_logits.json (needed for confidence analysis)')
    parser.add_argument('--frozen_run', action='store_true',
                        help='Set only through run_all_v2.py with a validated frozen config.')
    args = parser.parse_args()

    import datasets
    import torch
    use_gpu = torch.cuda.is_available()
    print(f'Loading model from: {args.model_path}')
    print(f'Device: {"GPU " + torch.cuda.get_device_name(0) if use_gpu else "CPU"}')

    if args.smoke_test:
        if args.data_release:
            print(f'\nLoading limited frozen {args.dataset} release...')
            val_data = load_frozen_development_data(args.data_release, args.dataset)
            val_data = limit_dataset(val_data, args.smoke_eval_samples)
        else:
            print('\nLoading deterministic local smoke data...')
            _, val_data = smoke_nli_splits(
                datasets, train_size=1, validation_size=args.smoke_eval_samples
            )
        print(f'SMOKE TEST MODE: {len(val_data)} val examples')
    else:
        if not args.data_release:
            raise SystemExit('--data_release is required for production evaluation')
        print(f'\nLoading frozen {args.dataset} release...')
        val_data = load_frozen_development_data(args.data_release, args.dataset)

    print(f'Val: {len(val_data):,}')

    # Build eval suite — these are TRANSFORMED inputs that will be preprocessed back
    if args.smoke_test and args.eval_sets_dir:
        base = Path(args.eval_sets_dir) / args.dataset
        eval_datasets = limit_eval_suite(load_eval_suite(
            base, variants=args.eval_variants or ('slang', 'emoji', 'noise', 'combined'),
            split_role=args.eval_role, split_name=args.eval_split,
            purpose=('final_evaluation' if args.eval_role == 'final'
                     else 'development_evaluation'),
        ), args.smoke_eval_samples)
        print(f'Loaded limited fixed eval sets from {base}')
    elif args.smoke_test:
        eval_datasets = {'original': val_data}
        for mode in ['slang', 'emoji', 'noise', 'combined']:
            eval_datasets[mode] = val_data.map(lambda ex: transform_example(ex, mode))
    elif args.eval_sets_dir:
        base = Path(args.eval_sets_dir) / args.dataset
        eval_datasets = load_eval_suite(
            base, variants=args.eval_variants or ('slang', 'emoji', 'noise', 'combined'),
            split_role=args.eval_role, split_name=args.eval_split,
            purpose=('final_evaluation' if args.eval_role == 'final' else 'development_evaluation'),
        )
        print(f'Loaded fixed eval sets from {base}')
    else:
        print('ERROR: --eval_sets_dir required. Run prepare_eval_sets.py first.')
        sys.exit(1)

    # Load saved baseline model
    tokenizer = load_tokenizer(
        args.model_name, args.model_revision, backend=args.tokenizer_backend,
        normalization=args.tokenizer_normalization,
    )
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, **model_pretrained_kwargs(None)
    )
    parameter_dtype = assert_float32_parameters(model, str(args.model_path))
    print(f'Loaded {args.model_path} with {parameter_dtype} parameters', flush=True)

    trainer_identity = hashlib.sha256(
        str(Path(args.output_dir).resolve()).encode('utf-8')
    ).hexdigest()[:16]
    trainer_output = (
        Path(args.cache_dir or tempfile.gettempdir())
        / 'nli_preprocessing_trainers' / trainer_identity
    )
    training_args = TrainingArguments(
        output_dir=str(trainer_output),
        per_device_eval_batch_size=args.batch_size,
        **mixed_precision_kwargs(torch, args.mixed_precision),
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_accuracy,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    results = {}
    print('\nEvaluating with preprocessing...')
    inference_started = time.monotonic()
    for name, raw_ds in tqdm(eval_datasets.items(), desc='Evaluating variants', unit='variant'):
        if name.startswith('clean_'):
            results[name] = paired_clean_result(
                results, name, raw_ds, save_logits=args.save_logits
            )
            tqdm.write(f'  {name:<12}: indexed from original (no forward pass)')
            continue
        source_indices = (
            list(raw_ds['source_index']) if 'source_index' in raw_ds.column_names
            else list(range(len(raw_ds)))
        )
        # Apply preprocessing BEFORE tokenization
        preprocessed = map_dataset_locally(
            raw_ds,
            lambda example: preprocess_example(example, mode=args.mode),
            cache_dir=args.cache_dir, operation=f'preprocess:{args.mode}:{name}',
            desc=f'Preprocessing {name}',
            num_proc=args.preprocessing_workers,
        )
        prepare_fn = lambda exs: prepare_dataset_nli(
            exs, tokenizer, args.max_length, args.input_preprocessing
        )
        tok_ds = map_dataset_locally(
            preprocessed, prepare_fn, batched=True,
            remove_columns=preprocessed.column_names,
            cache_dir=args.cache_dir,
            operation=tokenization_cache_operation(
                args.model_name, args.model_revision, args.max_length,
                f'{args.mode}:evaluation:{name}',
                tokenizer_backend=args.tokenizer_backend,
                tokenizer_normalization=args.tokenizer_normalization,
                input_preprocessing=args.input_preprocessing,
            ),
            desc=f'Tokenizing {name}',
            num_proc=args.preprocessing_workers,
        )
        output = trainer.predict(tok_ds)
        preds  = np.argmax(output.predictions, axis=1).tolist()
        labels = output.label_ids.tolist()
        acc    = float(np.mean(np.array(preds) == np.array(labels)))
        results[name] = {'accuracy': acc, 'predictions': preds, 'labels': labels,
                         'source_indices': source_indices}
        if args.save_logits:
            results[name]['logits'] = output.predictions.tolist()
        tqdm.write(f'  {name:<12}: {acc*100:.2f}%')
    runtime_seconds = time.monotonic() - inference_started

    attach_condition_metadata(results)
    drops = robustness_drops(results)

    training_subset = recorded_training_subset(Path(args.model_path).parent)
    system_id = ('emoji_normalization' if args.mode == 'emoji'
                 else 'known_marker_deletion_oracle')
    stage_metadata = system_stage_metadata(system_id)
    stage_metadata['run_origin'] = ('frozen_config' if args.frozen_run
                                    else 'direct_override')
    if not args.frozen_run and stage_metadata['classification'] == 'principal':
        stage_metadata['classification'] = 'exploratory'
    output_data = {
        'model_name': args.model_name,
        'model_path': args.model_path,
        'dataset': args.dataset,
        'approach': ('preprocessing' if args.mode == 'emoji' else 'marker_oracle'),
        'system_id': system_id,
        'config': vars(args),
        'training_subset': training_subset,
        'run_metadata': {
            **stage_metadata,
            **runtime_environment(torch),
            'trained_model_system_id': 'clean_baseline',
            'actual_optimizer_steps': 0,
            'runtime_scope': 'inference_only',
            'runtime_seconds': runtime_seconds,
            'provenance': {
                'training_subset': training_subset,
                **(source_dataset_identity(args.data_release, args.dataset)
                   if args.data_release else {}),
                **model_source_metadata(
                    args.model_name, args.model_revision,
                    tokenizer_backend=args.tokenizer_backend,
                    tokenizer_normalization=args.tokenizer_normalization,
                    input_preprocessing=args.input_preprocessing,
                    parameter_dtype=parameter_dtype,
                ),
            },
        },
        'accuracies': {k: v['accuracy'] for k, v in results.items()},
        'drops_pp': drops,
        'psych_instruction_inversion': psych_inversion_result(results),
        'condition_metadata': {
            name: value['condition_metadata'] for name, value in results.items()
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    preds_output = {k: {'predictions': v['predictions'], 'labels': v['labels'],
                        'source_indices': v['source_indices'],
                        'condition_metadata': v['condition_metadata']}
                    for k, v in results.items()}
    with open(out_dir / 'predictions.json', 'w') as f:
        json.dump(preds_output, f)
    if args.eval_sets_dir and not args.smoke_test:
        write_prediction_manifest(
            out_dir, Path(args.eval_sets_dir) / args.dataset,
            split_role=args.eval_role, split_name=args.eval_split,
            purpose=('final_evaluation' if args.eval_role == 'final' else 'development_evaluation'),
            variants=preds_output.keys(),
        )

    if args.save_logits:
        logits_output = {k: {'logits': v['logits']} for k, v in results.items() if 'logits' in v}
        with open(out_dir / 'raw_logits.json', 'w') as f:
            json.dump(logits_output, f)
        print(f'Logits saved to {out_dir}/raw_logits.json')

    print(f'\nResults saved to {out_dir}/results.json')


if __name__ == '__main__':
    main()
