"""
Stage-matched hybrid: apply compute-matched marker augmentation during training,
then normalize emoji at inference without deleting discourse markers.

Usage:
    python src/train_hybrid.py
    python src/train_hybrid.py --model_name roberta-large --output_dir ./results/roberta_hybrid
"""
import argparse
import json
import os
import sys
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
                     parse_train_size, select_training_subset,
                     limit_eval_suite,
                     load_eval_suite, load_frozen_training_data,
                     map_dataset_locally, paired_clean_result,
                     prepare_dataset_nli,
                     source_dataset_identity,
                     psych_inversion_result, robustness_drops,
                     smoke_nli_splits, write_prediction_manifest)
from on_the_fly_training import (SetDatasetEpochCallback,
                                 prepare_epoch_tokenized_dataset,
                                 validate_training_audit,
                                 validate_training_world_size)
from transforms import transform_example
from preprocessing import preprocess_example
from training_metadata import training_run_metadata, validate_training_run_metadata
from training_runtime import (load_tokenizer, model_source_metadata,
                              mixed_precision_kwargs, pretrained_kwargs,
                              assert_float32_parameters, model_pretrained_kwargs,
                              resume_checkpoint, tokenization_cache_operation)
from progress_logging import ProgressCallback


def evaluate_with_predictions(trainer, tokenizer, eval_datasets, max_length,
                              save_logits=False, cache_dir=None, model_revision=None,
                              input_preprocessing='none', tokenizer_backend='fast',
                              tokenizer_normalization='default'):
    results = {}
    for name, raw_ds in tqdm(eval_datasets.items(), desc='Evaluating variants', unit='variant'):
        if name.startswith('clean_'):
            results[name] = paired_clean_result(
                results, name, raw_ds, save_logits=save_logits
            )
            tqdm.write(f'  {name:<12}: indexed from original (no forward pass)')
            continue
        source_indices = (
            list(raw_ds['source_index']) if 'source_index' in raw_ds.column_names
            else list(range(len(raw_ds)))
        )
        prepare_fn = lambda exs: prepare_dataset_nli(
            exs, tokenizer, max_length, input_preprocessing
        )
        tok_ds = map_dataset_locally(
            raw_ds, prepare_fn, batched=True, remove_columns=raw_ds.column_names,
            desc=f'Tokenizing {name}', cache_dir=cache_dir,
            operation=tokenization_cache_operation(
                tokenizer.name_or_path, model_revision, max_length,
                f'evaluation:{name}', tokenizer_backend=tokenizer_backend,
                tokenizer_normalization=tokenizer_normalization,
                input_preprocessing=input_preprocessing,
            ),
        )
        output = trainer.predict(tok_ds)
        preds  = np.argmax(output.predictions, axis=1).tolist()
        labels = output.label_ids.tolist()
        acc    = float(np.mean(np.array(preds) == np.array(labels)))
        results[name] = {'accuracy': acc, 'predictions': preds, 'labels': labels,
                         'source_indices': source_indices}
        if save_logits:
            results[name]['logits'] = output.predictions.tolist()
        tqdm.write(f'  {name:<12}: {acc*100:.2f}%')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='google/electra-small-discriminator')
    parser.add_argument('--model_revision', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='snli', choices=['snli', 'multi_nli'])
    parser.add_argument('--output_dir', type=str, default='./results/electra_hybrid')
    parser.add_argument('--augment_ratio', type=float, default=0.5)
    parser.add_argument('--marker_placements', nargs='+',
                        default=['hypothesis_suffix', 'hypothesis_prefix', 'premise_suffix'])
    parser.add_argument('--marker_choices', nargs='+', default=None,
                        help='Fold-specific augmentation markers.')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_size', type=parse_train_size, default='full',
                        help='Number of training pairs: "full" or an integer. An '
                             'integer draws a label-stratified subset with --seed '
                             'before any augmentation.')
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
    parser.add_argument('--smoke_train_samples', type=int, default=100)
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
    parser.add_argument('--checkpoint_selection', type=str,
                        default='highest development accuracy; ties resolved by earliest checkpoint')
    args = parser.parse_args()

    import datasets
    import torch
    validate_training_world_size(int(os.environ.get('WORLD_SIZE', '1')))
    use_gpu = torch.cuda.is_available()
    print(f'Model : {args.model_name}')
    print(f'Device: {"GPU " + torch.cuda.get_device_name(0) if use_gpu else "CPU"}')

    if args.smoke_test:
        if args.data_release:
            print(f'\nLoading limited frozen {args.dataset} release...')
            train_data, val_data = load_frozen_training_data(
                args.data_release, args.dataset
            )
            if args.train_size == 'full':
                train_data = limit_dataset(train_data, args.smoke_train_samples)
            val_data = limit_dataset(val_data, args.smoke_eval_samples)
        else:
            print('\nLoading deterministic local smoke data...')
            train_data, val_data = smoke_nli_splits(
                datasets, train_size=args.smoke_train_samples,
                validation_size=args.smoke_eval_samples,
            )
        args.epochs = 1
        print(f'SMOKE TEST MODE: {len(train_data)} train / {len(val_data)} val / 1 epoch')
    else:
        if not args.data_release:
            raise SystemExit('--data_release is required for production training')
        print(f'\nLoading frozen {args.dataset} release...')
        train_data, val_data = load_frozen_training_data(args.data_release, args.dataset)

    train_data, training_subset = select_training_subset(
        train_data, args.train_size, args.seed
    )
    print(f'Train: {len(train_data):,}  Val: {len(val_data):,}  '
          f'(train_size={args.train_size})')

    # Build eval suite — apply preprocessing at inference time too
    if args.smoke_test and args.eval_sets_dir:
        base = Path(args.eval_sets_dir) / args.dataset
        raw_eval = limit_eval_suite(load_eval_suite(
            base, variants=args.eval_variants or ('slang', 'emoji', 'noise', 'combined'),
            split_role=args.eval_role, split_name=args.eval_split,
            purpose=('final_evaluation' if args.eval_role == 'final'
                     else 'development_evaluation'),
        ), args.smoke_eval_samples)
        print(f'Loaded limited fixed eval sets from {base}')
    elif args.smoke_test:
        raw_eval = {'original': val_data}
        for mode in ['slang', 'emoji', 'noise', 'combined']:
            raw_eval[mode] = val_data.map(lambda ex: transform_example(ex, mode))
    elif args.eval_sets_dir:
        base = Path(args.eval_sets_dir) / args.dataset
        raw_eval = load_eval_suite(
            base, variants=args.eval_variants or ('slang', 'emoji', 'noise', 'combined'),
            split_role=args.eval_role, split_name=args.eval_split,
            purpose=('final_evaluation' if args.eval_role == 'final' else 'development_evaluation'),
        )
        print(f'Loaded fixed eval sets from {base}')
    else:
        print('ERROR: --eval_sets_dir required. Run prepare_eval_sets.py first.')
        sys.exit(1)

    # Emoji normalization is inference-only; training remains marker-only.
    eval_datasets = {
        key: map_dataset_locally(
            value, lambda example: preprocess_example(example, mode='stage_matched'),
            cache_dir=args.cache_dir, operation=f'preprocess:stage_matched:{key}',
            desc=f'Preprocessing {key}',
        )
        for key, value in raw_eval.items()
    }

    tokenizer = load_tokenizer(
        args.model_name, args.model_revision, backend=args.tokenizer_backend,
        normalization=args.tokenizer_normalization,
    )
    prepare_fn = lambda exs: prepare_dataset_nli(
        exs, tokenizer, args.max_length, args.input_preprocessing
    )

    print(f'Precomputing batched epoch presentations (ratio={args.augment_ratio})...')
    train_tok = prepare_epoch_tokenized_dataset(
        train_data, tokenizer, args.max_length,
        epochs=args.epochs,
        augmentation_probability=args.augment_ratio,
        marker_placements=args.marker_placements,
        marker_choices=args.marker_choices,
        seed=args.seed,
        cache_dir=args.cache_dir,
        num_proc=args.preprocessing_workers,
        # Training and inference stages are intentionally independent. Clean
        # training examples receive only stochastic marker augmentation.
        preprocess_mode=None,
        input_preprocessing=args.input_preprocessing,
        model_revision=args.model_revision,
        tokenizer_backend=args.tokenizer_backend,
        tokenizer_normalization=args.tokenizer_normalization,
    )
    val_tok = map_dataset_locally(
        val_data, prepare_fn, batched=True,
        remove_columns=val_data.column_names,
        cache_dir=args.cache_dir,
        operation=tokenization_cache_operation(
            args.model_name, args.model_revision, args.max_length,
            'hybrid-development', tokenizer_backend=args.tokenizer_backend,
            tokenizer_normalization=args.tokenizer_normalization,
            input_preprocessing=args.input_preprocessing,
        ),
        num_proc=args.preprocessing_workers,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3, **model_pretrained_kwargs(args.model_revision)
    )
    parameter_dtype = assert_float32_parameters(model, args.model_name)
    print(f'Loaded {args.model_name} with {parameter_dtype} parameters', flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        **mixed_precision_kwargs(torch, args.mixed_precision),
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_steps=500,
        logging_first_step=True,
        dataloader_num_workers=0,
        disable_tqdm=True,
        seed=args.seed,
        report_to='none',
    )
    validate_training_world_size(training_args.world_size)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_accuracy,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[SetDatasetEpochCallback(), ProgressCallback(args.output_dir)],
    )

    print('\nTraining...')
    training_started = time.monotonic()
    trainer.train(resume_from_checkpoint=resume_checkpoint(args.output_dir, train_tok))
    runtime_seconds = time.monotonic() - training_started
    training_audit = train_tok.audit(args.epochs)
    validate_training_audit(training_audit)
    trainer.save_model(args.output_dir + '/final_model')

    print('\nEvaluating on all variants (with preprocessing at inference)...')
    results = evaluate_with_predictions(trainer, tokenizer, eval_datasets, args.max_length,
                                        save_logits=args.save_logits,
                                        cache_dir=args.cache_dir,
                                        model_revision=args.model_revision,
                                        input_preprocessing=args.input_preprocessing,
                                        tokenizer_backend=args.tokenizer_backend,
                                        tokenizer_normalization=args.tokenizer_normalization)
    attach_condition_metadata(results)

    drops = robustness_drops(results)

    run_metadata = training_run_metadata(
        system_id='stage_matched_hybrid',
        train_examples=len(train_data), epochs=args.epochs,
        batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio, learning_rate=args.lr,
        actual_optimizer_steps=trainer.state.global_step,
        runtime_seconds=runtime_seconds,
        checkpoint_selection=args.checkpoint_selection,
        presentation_audit=training_audit,
        world_size=training_args.world_size, torch_module=torch,
        frozen_run=args.frozen_run,
        provenance={
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
    )
    validate_training_run_metadata(run_metadata)
    output_data = {
        'model_name': args.model_name,
        'dataset': args.dataset,
        'approach': 'hybrid',
        'system_id': 'stage_matched_hybrid',
        'config': vars(args),
        'train_samples': len(train_data),
        'training_subset': training_subset,
        'actual_optimizer_steps': trainer.state.global_step,
        'run_metadata': run_metadata,
        'training_presentation_audit': training_audit,
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
