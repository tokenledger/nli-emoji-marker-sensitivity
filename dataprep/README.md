# dataprep

Downloads SNLI and MultiNLI and builds every edited evaluation set of the
paper.

## Targets

| Target | Action |
|---|---|
| `make download` | Fetch SNLI and MultiNLI with the Hugging Face `datasets` library and freeze training, development, and final splits in `DATA_DIR`. Needs network access. Fails if `DATA_DIR` exists. |
| `make download-external` | Fetch ChaosNLI (MultiNLI part), SICK test, and ANLI dev R1 to R3 into `DATA_DIR/external/`. Needed only for `make case-study`. |
| `make modify` | Build the edited evaluation sets from the frozen source rows. |
| `make verify` | Run the unit tests of the data code, then check the generated data. |
| `make clean` | Remove the generated evaluation sets. The downloaded source data stay. |
| `make distclean` | Remove `DATA_DIR`. |

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `<repository>/data` | Data directory. |
| `SEED` | `42` | Generation seed. The configuration fixes it at 42; another value is rejected. |
| `EDITS` | all | Groups of edits to build, separated by spaces (below). |
| `DATASETS` | `snli mnli` | Datasets to build. |
| `SPLITS` | all | Source splits to build: `test`, `validation_matched`, `validation_mismatched`, and the development splits `validation` (SNLI) and `train_holdout` (MultiNLI). |
| `WORKERS` | `0` | Number of processes; 0 uses all CPUs. |
| `CONFIG` | `configs/experiment_v6.json` | Frozen experiment configuration. |
| `SNLI_REVISION`, `MNLI_REVISION` | revisions used for the paper | Revisions of the datasets on the Hugging Face Hub. |

Example: `make modify EDITS="emoji markers" DATASETS=snli SPLITS=test`.

## Groups of edits

| `EDITS` value | Conditions | Destination |
|---|---|---|
| `emoji` | `emoji_raw` (emoji substitution) and `emoji_gloss` (the same pairs with the nouns restored) | `eval_sets/` |
| `markers` | Markers by fold (1 to 3), status (in the augmentation pool, held out), and position (end of hypothesis, start of hypothesis, end of premise): 18 conditions | `eval_sets/` |
| `controls` | For every marker condition, a formal phrase and another appended word of the same word count: 36 conditions | `eval_sets/` |
| `combined` | Emoji substitution and a held-out marker on the same pair, per fold | `eval_sets/` |
| `other` | Conditions of the frozen configuration that the paper does not report (`slang`, `psych`, `emoji_lossy_stress`). They are built by default so that the evaluation manifest equals that of the paper. | `eval_sets/` |
| `crossed` | The crossed study: 24 conditions on every source pair of the three final splits (`CONDITION_REGISTRY.csv`) | `crossed/` |
| `content-words` | The nine content-word phrases on every source pair of the three final splits | `content_words/` |

`eval_sets/` is never overwritten. Run `make clean` before `make modify` builds
the first five groups again. A limited build records its limits in
`eval_sets/dataset_manifest.json` (`generation_filter`) and in
`release_manifest.json`. Training on the full protocol requires the unlimited
build.

## Checks of `make verify`

1. Unit tests: `tests/test_phase1_design.py`, `test_phase3_challenge_sets.py`,
   `test_data_release.py`, `test_psych_variant.py`, `test_dataprep.py`.
2. Source splits: row counts and checksums equal those of the paper's data
   (`expected_checksums.json`).
3. Evaluation sets: every generated condition has the row count and checksum
   of the paper's data.
4. Eligible emoji pairs: 6,430 (SNLI test), 522 (MultiNLI matched), 574
   (mismatched); at most two replacements per pair.
5. Exact inversion: undoing the recorded spans of every emoji pair reproduces
   the text of the paired clean pair. This check covers the text. The identity
   of token identifiers reported in the paper is not rechecked here.
6. No held-out marker in a training pool: per fold, the pools are disjoint and
   the held-out conditions contain only held-out markers.
7. Label balance: within each label, the markers of a held-out condition differ
   in frequency by at most one pair.
8. Crossed study: 24 conditions on every source pair; rows equal those used for
   the paper.
9. Content-word phrases: nine phrases on every source pair; each hypothesis is
   the clean hypothesis, a space, and the phrase.

A check whose input was not generated is reported as SKIP.

## Note on trailing whitespace

The crossed conditions and the content-word phrases follow the rules that
produced the predictions of the paper. The rules differ when a hypothesis ends
in whitespace: the crossed builder inserts no second space before the phrase,
and the content-word inference appended a space and the phrase to the unchanged
hypothesis. This affects 1,291 MultiNLI matched and 1,348 mismatched source
pairs and no SNLI test pair.

## Files

| File | Origin |
|---|---|
| `dataprep.py`, `verify.py` | new |
| `phrase_sets.py` | derived from `build_eval.py` of the crossed evaluation |
| `external_data.py` | derived from `download_data.py` of the transfer experiment |
| `CONDITION_REGISTRY.csv` | unchanged |
| `expected_checksums.json` | extracted from the manifests of the paper's data |
| `../src/prepare_data_release.py`, `prepare_eval_sets.py`, `transforms.py`, `preprocessing.py`, `helpers.py`, `experiment_config.py`, `pipeline_audit.py` | shared code |
