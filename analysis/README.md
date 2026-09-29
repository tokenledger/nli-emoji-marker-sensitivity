# analysis

Turns saved predictions into the numbers, tables, and figures of the paper. No
target of this module trains or runs a model, and all run on a CPU.

## Targets

| Target | Content | Paper | Depends on |
|---|---|---|---|
| `make primary` | The four prespecified contrasts per encoder, split, and fold; emoji repair; augmentation gain; fold controls and position; the separate DeBERTa-v3-base family; Table 2 | Sections 5.1, 5.4; Table 2; App. S3, S4, S7 | |
| `make run-sensitivity` | Agreement of additional training runs with seed 42 | Section 5.1; App. S5 | `SENSITIVITY_INPUT`, which is not distributed (see below) |
| `make emoji-structure` | Emoji loss by replacement structure and by noun | Section 5.2; Table 3; App. S6 | |
| `make crossed` | Crossed study: condition effects, component-word contrasts, shared failure, flip rates, commitment-marker contrast | Sections 5.3, 5.5; App. S8, S13 | |
| `make crossed-validate` | Independent recomputation of the outputs of `make crossed` | | `crossed` |
| `make content-words` | Content-word phrases; content-word coding of 32 phrases | Sections 5.3, 6; App. S9 | `crossed` |
| `make case-study` | ChaosNLI subset, SICK and ANLI transfer, training-data account | Section 5.5; App. S13 | `crossed` |
| `make training-data-account` | The training-data account alone (part of `case-study`) | App. S13 | |
| `make figures` | Figures 1 and 2 | Figures 1, 2 | `crossed`, `content-words` |
| `make all` | All of the above except `run-sensitivity` and `crossed-validate`, in order | | |
| `make test` | Unit tests of the analysis code | | |
| `make clean` | Remove `OUTPUT_DIR` | | |

All-five failure (Section 5.5, Figure 2) is computed by `make crossed`
(`crossed/cofailure_results_v2.csv`) and, for the content-word phrases, by
`make content-words` (`content_words/results.csv`).

The input file of `make run-sensitivity`
(`primary/inputs/seed_sensitivity_compact_inputs.json`, the compact results of
the additional training runs) is not distributed in this repository. Without
it the target stops with the message `ERROR: required input ... is missing`
and a non-zero exit status, and writes nothing. To run the target, place a
compact input file at that location or pass its absolute path as
`SENSITIVITY_INPUT`.

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `PREDICTIONS_DIR` | `<repository>/predictions` | Saved predictions (layout below). Read only, except by the inference targets of the training module. |
| `DATA_DIR` | `<repository>/data` | Data directory of the dataprep module. |
| `OUTPUT_DIR` | `<repository>/outputs` | Destination of all results. |
| `SENSITIVITY_INPUT` | `<repository>/analysis/primary/inputs/seed_sensitivity_compact_inputs.json` | Input of `make run-sensitivity`. Not distributed. |
| `TOKENIZER_MODE` | `off` | Truncation diagnostics of `make primary`: `off`, `local` (tokenizers from the local cache), or `download`. |
| `CHECKPOINTS_DIR` | empty | Fine-tuned checkpoints. Only the token-length table of `make content-words` uses it; without it that table is skipped. |

No script contains an absolute path. Scripts resolve every file through
`paths.py`, which reads these variables, and stop with a message when a
variable or an input is missing.

## Layout of PREDICTIONS_DIR

```
PREDICTIONS_DIR/
  fold_study/                       fold study, five encoders, seed 42
    publication_results/<dataset>/seed_42/<model>_<system>_final_<split>/
    _colab/publication_results_roberta_base42/<dataset>/seed_42/roberta_base_<system>_final_<split>/
    _colab/publication_results_roberta42_snli/snli/seed_42/roberta_<system>_final_test/
    _colab/publication_results_roberta42_mnli/multi_nli/seed_42/roberta_<system>_final_<split>/
  deberta/<dataset>/deberta_v3_base_<system>_final_<split>/
  crossed/
    workspace/                      frozen record of the crossed inference
      CONDITION_REGISTRY.csv, RUN_PROTOCOL.md, run_manifest.json,
      run_manifest.json.logical.sha256
      predictions/<model>__<dataset>__<split>__<id>.jsonl.gz and its
        .complete.json and .complete.json.logical.sha256 files
    bertweet_corrected/             bertweet__<dataset>__<split>__tokenizerfix__<id>.jsonl.gz
                                    and its two .complete.json files
    deberta/                        deberta_v3_base_controlled_<train>_seed42__<dataset>__<split>.jsonl.gz
  content_words/
    predictions/<model>__<dataset>__<split>.jsonl.gz
    manifests/<model>__<dataset>__<split>.json
  transfer/
    predictions/<model>__<train>__<data>.jsonl.gz
    manifests/<model>__<train>__<data>.json
```

`<dataset>` is `snli` or `multi_nli`; `<split>` is `test`,
`validation_matched`, or `validation_mismatched`; `<model>` is `electra`,
`roberta_base`, `roberta`, `timelm`, `bertweet`, or `deberta_v3_base`.
`<system>` is `baseline`, `preprocessing` (emoji normalization),
`augmented_fold_<k>`, `hybrid_fold_<k>`, `clean_control`, or `marker_oracle`.

Every run directory of `fold_study/` and `deberta/` holds `predictions.json`,
`predictions_manifest.json`, and `results.json`, as written by the training
module. `predictions.json` maps each condition to `predictions`, `labels`, and
`source_indices`. The `fold_study/` tree is the layout of the storage volume
that held the runs of the paper: ELECTRA-small, TimeLM-21, and BERTweet below
`publication_results/`, the two RoBERTa encoders below `_colab/`.
`make primary` needs all six systems of the five encoders (written by
`make train-all` of the training module), for the three folds where a system
depends on the fold.

Runs produced by the training module have the layout of `publication_results/`.
To analyse them, link `RESULTS_DIR` to `fold_study/publication_results` and to
the three `_colab/` directories.

The `.jsonl.gz` files hold one JSON object per line with `condition_id`,
`source_index`, `label`, and `prediction` (0 entailment, 1 neutral,
2 contradiction).

## Layout of OUTPUT_DIR

| Directory | Written by |
|---|---|
| `primary/statistics/`, `primary/tables/`, `primary/deberta/`, `primary/table2.md`, `primary/table2.csv` | `primary` |
| `run_sensitivity/` | `run-sensitivity` |
| `emoji_structure/` | `emoji-structure` |
| `crossed/`, `flip_rates/`, `commitment_markers/` | `crossed` |
| `content_words/`, `content_word_coding/` | `content-words` |
| `case_study/chaosnli/`, `case_study/transfer/`, `case_study/training_data_account/` | `case-study` |
| `figures/` | `figures` |
| `cache/` | compact copies of the crossed predictions, written on first use |

`make primary` keeps finished files in `primary/statistics/` and computes only
the missing ones.

## Determinism

Bootstrap and permutation procedures use fixed seeds. On the predictions of
the paper, the targets reproduce the stored result files byte for byte, with
these exceptions: p-values in `primary/statistics/` can differ in the last
digits between versions of the numerical libraries; records of file names and
hashes differ with the location of the files; PDF files differ in their
creation date.

## Files

| Directory | Scripts | Origin |
|---|---|---|
| `primary/` | `run_primary.py`, `table2.py` | new; they call `../src/statistical_tests.py`, `publication_analysis.py`, `deberta_ext_statistics.py` |
| `primary/` | `seed_sensitivity_analysis.py` | copied; its input file is not distributed |
| `emoji_structure/` | `emoji_structure.py` | copied; paths changed |
| `crossed/` | `analyze_v2.py`, `run_bertweet_cpu.py`, unit tests | unchanged |
| `crossed/` | `validate_v2.py`, `common.py`, `flip_rates.py`, `commitment_markers.py` | copied; paths changed |
| `content_words/` | `analyze.py`, `content_word_coding.py`, `nli_inference.py`, `run_inference.py` | copied; paths changed |
| `content_words/` | `conditions.py`, `stats_utils.py` | unchanged |
| `case_study/` | `chaosnli_subset.py`, `transfer/`, `training_data_account/` | copied; paths changed |
| `figures/` | `make_phrase_figure.py` | copied; paths changed |
