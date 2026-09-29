# training

Fine-tunes the systems of the paper and writes per-pair predictions.

## Targets

| Target | Action |
|---|---|
| `make train` | Fine-tune one system and evaluate it on the primary final split (SNLI test, MultiNLI matched). |
| `make train-all` | Run all six systems of one encoder, dataset, and fold on `SPLIT`, then the paired tests. |
| `make predict` | Evaluate a trained system on `SPLIT`, with and without emoji normalization. |
| `make predict-content-words` | Run the seed-42 clean checkpoints on the nine content-word phrases. |
| `make predict-transfer` | Run the seed-42 clean checkpoints on SICK and ANLI. |
| `make smoke` | A tiny end-to-end run for a CPU. |
| `make clean` | Remove the outputs of the smoke run. Trained runs are kept. |

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | required | `electra` (ELECTRA-small), `roberta_base`, `roberta` (RoBERTa-large), `timelm` (TimeLM-21), `bertweet`, `deberta_v3_base`. |
| `DATASET` | required | `snli` or `mnli`. |
| `SYSTEM` | `baseline` | `baseline`, `augmented`, or `hybrid`. |
| `FOLD` | `1` | Marker fold 1, 2, or 3; ignored for `baseline`. |
| `SEED` | `42` | Training seed. It also seeds the training subset. |
| `TRAIN_SIZE` | `full` | Number of training pairs (below). |
| `SPLIT` | primary split | Final split for `make predict`: `test`, `validation_matched`, `validation_mismatched`. |
| `NORMALIZATION` | `both` | For `make predict SYSTEM=baseline`: `both`, `with`, or `without`. |
| `DATA_DIR` | `<repository>/data` | Data directory of the dataprep module, built without limits. |
| `RESULTS_DIR` | `<repository>/results` | Destination of runs. |
| `CHECKPOINTS_DIR` | required for the two extra predict targets | Directories `<model>_snli` and `<model>_mnli`, each a saved clean-baseline model. |
| `PREDICTIONS_DIR` | `<repository>/predictions` | Destination of the two extra predict targets (layout in `../analysis/README.md`). |

Examples:

```bash
make train MODEL=roberta_base DATASET=mnli SYSTEM=augmented FOLD=2
make train MODEL=electra DATASET=snli TRAIN_SIZE=20000
make predict MODEL=roberta_base DATASET=mnli SPLIT=validation_mismatched
```

## Systems

| `SYSTEM` | Training | Inference | `make predict` writes |
|---|---|---|---|
| `baseline` | clean pairs | unchanged input; with `NORMALIZATION` `with` or `both`, also input after emoji normalization | `<model>_baseline_*`, `<model>_preprocessing_*` |
| `augmented` | one of the fold's six training markers with probability 0.5, at one of three positions | unchanged input | `<model>_augmented_fold_<k>_*` |
| `hybrid` | as `augmented`, trained separately | input after emoji normalization | `<model>_hybrid_fold_<k>_*` |

Emoji normalization is an inference step applied to the clean baseline; it
trains no model. The frozen configuration defines two further systems that the
paper does not report: a clean control trained through the code path of the
augmented system, and an oracle that deletes known markers at inference. The
paired tests of `make primary` (analysis) read all six systems, so
`make train-all` runs them all. For DeBERTa-v3-base the paper trains only the clean baseline,
and the Makefile rejects other systems.

The two extra predict targets run inference only. They first classify the clean
pairs and compare the result with the saved clean predictions in
`PREDICTIONS_DIR/crossed/`; a checkpoint that deviates by more than 0.5 points
is not used. The reference accuracies are those of the checkpoints of the
paper (`analysis/content_words/nli_inference.py`, `ARCHIVED_CLEAN_ACC`), so
checkpoints from a new training run need new reference values. They set the Hugging Face libraries to offline mode unless
`HF_HUB_OFFLINE=0` is given, which the download of the BERTweet tokenizer
requires on a machine without a filled cache.

## Output

A run is written to
`RESULTS_DIR/<dataset>/seed_<seed>/<model>_<system>_final_<split>/` and holds
`results.json`, `predictions.json`, `predictions_manifest.json`,
`raw_logits.json`, `run_manifest.json`, `run_status.json`, and, for trained
systems, `final_model/`. A completed run is not repeated.

## TRAIN_SIZE

`TRAIN_SIZE=full` uses the whole training set. An integer draws a
label-stratified subset of that size: each label receives its proportional
share, and its pairs are drawn without replacement from a generator seeded with
`SEED`. The subset is drawn from the clean training pairs before augmentation
is applied, so the augmented and hybrid systems augment the subset.

A subset run cannot be mistaken for a full run:

- it is written to `RESULTS_DIR/<dataset>/seed_<seed>/train_size_<N>/`;
- `results.json` holds `training_subset` (requested size, number of selected
  pairs, size of the full training set, seed, label counts, and a hash of the
  selected indices), also under `run_metadata.provenance`;
- `run_manifest.json` holds `runtime_overrides.train_size`, and a run directory
  is refused when its recorded size differs from the requested one;
- predictions made from a subset checkpoint carry the same record.

The number of optimizer steps follows from the number of training pairs. Results
of the paper use `TRAIN_SIZE=full`.

## Smoke run

`make smoke` trains ELECTRA-small for one epoch on `TRAIN_SIZE=200` pairs,
evaluates 24 pairs per condition, and then applies emoji normalization to the
same checkpoint. It sets the Hugging Face libraries to offline mode, so
ELECTRA-small must be in the local cache. If `DATA_DIR` holds a data release,
the subset is drawn from the SNLI training set and the evaluation pairs come
from the edited evaluation sets. Otherwise the run uses synthetic pairs that
the code generates. Variables: `SMOKE_TRAIN_SIZE`, `SMOKE_EVAL_PAIRS`,
`SMOKE_DIR`.

## Hyperparameters of the paper

Shared by all six encoders (`configs/experiment_v6.json`, key `training`):

| Setting | Value |
|---|---|
| Epochs | 3 |
| Optimizer | AdamW, default settings of the `transformers` Trainer; weight decay 0 |
| Schedule | linear, warmup ratio 0.06 |
| Maximum length | 128 |
| Effective batch size | 32 |
| Evaluation and saving | after each epoch |
| Checkpoint selection | highest clean development accuracy; the earliest in case of a tie |
| Augmentation probability | 0.5 |
| Seed | 42 |

Per encoder (`configs/experiment_v6.json`, key `models`;
`configs/experiment_v6_deberta_ext.json` for DeBERTa-v3-base):

| Encoder | `MODEL` | Checkpoint | Learning rate | Batch | Accumulation | Tokenizer and input |
|---|---|---|---|---|---|---|
| ELECTRA-small | `electra` | `google/electra-small-discriminator` | 2e-5 | 32 | 1 | default |
| RoBERTa-base | `roberta_base` | `roberta-base` | 2e-5 | 32 | 1 | default |
| RoBERTa-large | `roberta` | `roberta-large` | 1e-5 | 16 | 2 | default; bf16 autocast |
| TimeLM-21 | `timelm` | `cardiffnlp/twitter-roberta-base-2021-124m` | 2e-5 | 32 | 1 | model-card preprocessing of user names and URLs |
| BERTweet | `bertweet` | `vinai/bertweet-base` | 2e-5 | 32 | 1 | slow tokenizer, normalization disabled |
| DeBERTa-v3-base | `deberta_v3_base` | `microsoft/deberta-v3-base`, revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` | 2e-5 | 32 | 1 | fast tokenizer, SentencePiece byte fallback enabled; bf16 autocast |

The `Makefile` selects the DeBERTa configuration when
`MODEL=deberta_v3_base`. `bf16` requires a CUDA device that supports it.

## Settings specific to DeBERTa-v3-base

- **Weights in fp32.** The checkpoint is stored in half precision, and
  `transformers` 5 loads a model in the precision of its checkpoint.
  Fine-tuning from half-precision weights did not converge. Every model is
  therefore loaded with `dtype=torch.float32`
  (`src/training_runtime.py`, `model_pretrained_kwargs`), and
  `assert_float32_parameters` stops a run whose parameters have another type.
  Autocast to bf16 is applied on top of the fp32 weights.
- **SentencePiece byte fallback.** The converted fast tokenizer has byte
  fallback disabled and maps 5 of the 16 emoji to the unknown token. The
  setting `tokenizer_normalization: spm_byte_fallback` enables byte fallback
  each time the tokenizer is loaded (`apply_spm_byte_fallback`), because a
  reload with default settings drops it.

Both settings are tested in `tests/test_phase4_training.py`.

## Files

`../src/run_all_v2.py` schedules the steps and calls `train_baseline.py`,
`train_augmentation.py`, `train_hybrid.py`, `train_preprocessing.py` (emoji
normalization), and `evaluate_checkpoint.py` (evaluation of a checkpoint on a
further split). The inference scripts for the content-word phrases and the
transfer are `../analysis/content_words/run_inference.py` and
`../analysis/case_study/transfer/run_inference.py`.
