# Emoji Substitution and Appended Discourse Markers in NLI

Code for the paper "Emoji Substitution and Appended Discourse Markers in NLI:
Accuracy Loss Depends on Replacement Structure and on the Appended Phrase"
(Avinash Goutham Aluguvelly).

The paper fine-tunes six encoders (ELECTRA-small, RoBERTa-base, RoBERTa-large,
TimeLM-21, BERTweet, DeBERTa-v3-base) on SNLI and MultiNLI, evaluates them on
edited copies of the test data, and analyses paired predictions. This
repository contains the code that builds the data, trains and runs the models,
and turns predictions into the numbers, tables, and figures of the paper.

**License:** the code is released under the MIT License (see `LICENSE`). SNLI, MultiNLI, ChaosNLI, SICK, and ANLI keep their own licenses; data derived from them is subject to those terms.

## Modules

The modules are used in this order. Each has its own README and Makefile.

| Module | Purpose |
|---|---|
| [`dataprep/`](dataprep/README.md) | Download SNLI and MultiNLI; build the edited evaluation sets (emoji substitution, markers by fold and position, fold controls, combined condition, the 24 crossed conditions, nine content-word phrases); verify them. |
| [`training/`](training/README.md) | Fine-tune the four systems (clean baseline, emoji normalization at inference, marker augmentation, hybrid) and write predictions. |
| [`analysis/`](analysis/README.md) | Compute the contrasts, tables, and figures from saved predictions. |

Shared code is in `src/`, the frozen experiment configurations in `configs/`,
and the unit tests in `tests/`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make help                                   # every target of every module
make data DATA_DIR=$PWD/data                # download, build, and verify the data
make -C training smoke DATA_DIR=$PWD/data   # tiny end-to-end run on a CPU
make train MODEL=electra DATASET=snli SYSTEM=baseline DATA_DIR=$PWD/data
make analysis PREDICTIONS_DIR=/path/to/predictions DATA_DIR=$PWD/data OUTPUT_DIR=$PWD/outputs
```

Paths given to `make` must be absolute. `make test` runs the unit
tests.

## What can be reproduced

| Step | From this repository alone | Needs |
|---|---|---|
| Source data and all edited evaluation sets | yes | network access to the Hugging Face Hub |
| Verification of the data against the paper's data | yes | `dataprep/expected_checksums.json` (included) |
| Training and inference | yes | GPU time; model weights from the Hub |
| Sensitivity to the training run (App. S5) | no | the compact input file of `make run-sensitivity`, which is not distributed; without it the target stops with an error message |
| Every other number, table, and figure | no | the saved predictions of the paper (`PREDICTIONS_DIR`) |

Evaluation sets are not stored in this repository; they are rebuilt by
`make -C dataprep modify` after the download. Saved predictions are not
distributed in this repository, and neither are the fine-tuned checkpoints or
the reviewers' judgments. Without the predictions, the analyses can be run only on predictions that a
user produces with the training module. Such predictions come from new
training runs. The paper reports that the direction of the four prespecified
contrasts is stable across runs and that their size is not, so new runs are not
expected to reproduce the published values exactly.

The following results of the paper have no runnable code in this repository:

| Result | Reason |
|---|---|
| Label-preservation review (Section 3.5, App. S2) | The rates were tallied by hand from the reviewers' returned files, which are not distributed. |
| Identity of token identifiers and logits after undoing the emoji substitution, and the comparison of the normalizer with paired clean input (App. S1) | The audit read raw logits in cloud storage. `make verify` repeats the check on the text only. |
| Tokenizer counts for DeBERTa-v3-base with and without byte fallback (App. S3) | The preflight script is not included. The setting is covered by unit tests in `tests/test_phase4_training.py`. |
| Adjusted model and emoji representation controls (App. S6) | Not included; both need prediction bundles or inference runs outside this repository. |
| DeBERTa-v3-base crossed results as tabulated in App. S10 | The flip rates and accuracy changes are produced by `make crossed`; the script that wrote the remaining S10 tables is not included. |
| Public DeBERTa-v3 NLI checkpoint and prompted model (App. S11) | Not included. |
| Fix interaction and composition of the two edits (App. S12) | Not included. |
| Crossed-study inference for the 24 conditions | The inference ran on cloud GPUs; the runner is not included. `analysis/crossed/run_bertweet_cpu.py`, which produced the BERTweet cells, is included unchanged because the analysis checks its hash. |

## Tables, figures, and appendix sections

Targets are those of `analysis/Makefile` unless another module is named.

| Item | Content | Target | Output below `OUTPUT_DIR` |
|---|---|---|---|
| Table 1 | Example conditions | `dataprep: make modify` (rows of the evaluation sets) | none |
| Table 2 | Per-encoder effects | `make primary` | `primary/table2.md` |
| Table 3 | Emoji loss by replacement structure | `make emoji-structure` | `emoji_structure/structure_by_model.csv` |
| Figure 1 | Paired accuracy change per phrase | `make figures` | `figures/phrase_effects.pdf` |
| Figure 2 | All-five failure on MultiNLI | `make figures` | `figures/shared_failure.pdf` |
| S1 | Emoji inverse, normalizer, evaluation noise | `dataprep: make verify` (text inversion); `make primary` (`tables/emoji_diagnostics.csv`) | partly; see above |
| S2 | Label-preservation review | none | not reproducible |
| S3 | Training setup; DeBERTa-v3-base per cell | `make primary` | `primary/deberta/REPORT.md`; setup in `training/README.md` |
| S4 | Test families | `make primary`, `crossed`, `content-words`, `case-study` | the test tables of each target |
| S5 | Run sensitivity and truncation | `make run-sensitivity` (input file not distributed); `make primary TOKENIZER_MODE=local` | `run_sensitivity/`; `primary/tables/truncation_diagnostics.csv` |
| S6 | Emoji structure and nouns | `make emoji-structure` | `emoji_structure/`; adjusted model and representation controls not included |
| S7 | Fold controls and position | `make primary` | `primary/tables/primary_condition_control_*.csv`, `placement_diagnostics.csv` |
| S8 | Crossed study: component words, flip rates | `make crossed` | `crossed/`, `flip_rates/` |
| S9 | Content-word phrases and phrase coding | `make content-words` | `content_words/`, `content_word_coding/` |
| S10 | DeBERTa-v3-base in the crossed study | `make crossed` | `flip_rates/flip_rates_long.csv`; partly, see above |
| S11 | Other models | none | not included |
| S12 | Fix interaction and composition | none | not included |
| S13 | Case study | `make case-study`; commitment markers: `make crossed` | `case_study/`, `commitment_markers/` |

## Hardware and time

Only figures that the experiment records of the project state are given here.

| Run | Hardware | Time |
|---|---|---|
| DeBERTa-v3-base clean baseline, SNLI, training and evaluation | one NVIDIA L40S | 4,723 s |
| DeBERTa-v3-base clean baseline, MultiNLI matched, training and evaluation | one NVIDIA L40S | 3,629 s |
| DeBERTa-v3-base emoji normalization (inference) | one NVIDIA L40S | 519 s (SNLI), 1,054 s (MultiNLI matched) |
| 216 additional training and evaluation calls of the run-sensitivity study | cloud GPUs | 53.8 hours of command time in total |
| Content-word phrases, six encoders, three splits (inference) | Apple MPS, fp32 | 102 min |
| SICK and ANLI transfer, six encoders (inference) | Apple MPS, fp32 | 33 min |
| DeBERTa-v3-base crossed study, SNLI (inference, 235,776 rows) | Apple MPS, fp32 | 6.8 min |

The five-encoder seed-42 runs for ELECTRA-small, TimeLM-21, and BERTweet used
NVIDIA L40S GPUs. All analyses run on a CPU.

## Appendix

The online appendix (App. S1 to S13) is `appendix/appendix.pdf`.
