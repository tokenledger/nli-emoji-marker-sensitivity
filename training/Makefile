# Training and inference.

SHELL := /bin/bash
.DEFAULT_GOAL := help

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON ?= python3

DATA_DIR ?= $(ROOT)/data
RESULTS_DIR ?= $(ROOT)/results
PREDICTIONS_DIR ?= $(ROOT)/predictions
CHECKPOINTS_DIR ?=
MODEL ?=
DATASET ?=
SYSTEM ?= baseline
FOLD ?= 1
SEED ?= 42
TRAIN_SIZE ?= full
SPLIT ?=
NORMALIZATION ?= both
WORKERS ?= 1
SMOKE_TRAIN_SIZE ?= 200
SMOKE_EVAL_PAIRS ?= 24
SMOKE_DIR ?= $(ROOT)/results_smoke

MODELS := electra roberta_base roberta timelm bertweet deberta_v3_base
CONFIG ?= $(ROOT)/configs/$(if $(filter deberta_v3_base,$(MODEL)),experiment_v6_deberta_ext.json,experiment_v6.json)
DATASET_KEY = $(if $(filter mnli multi_nli,$(DATASET)),multi_nli,$(DATASET))
STEP_baseline := baseline
STEP_augmented := augmentation
STEP_hybrid := hybrid
STEP = $(STEP_$(SYSTEM))
DEFAULT_SPLIT = $(if $(filter multi_nli,$(DATASET_KEY)),validation_matched,test)
EVAL_SPLIT = $(if $(strip $(SPLIT)),$(SPLIT),$(DEFAULT_SPLIT))
REUSE = $(if $(filter-out $(DEFAULT_SPLIT),$(EVAL_SPLIT)),--reuse_trained_from_split $(DEFAULT_SPLIT))
APPROACH_baseline := baseline
APPROACH_augmented = augmented_fold_$(FOLD)
APPROACH_hybrid = hybrid_fold_$(FOLD)
RUN_ROOT = $(RESULTS_DIR)/$(DATASET_KEY)/seed_$(SEED)$(if $(filter-out full,$(TRAIN_SIZE)),/train_size_$(TRAIN_SIZE))
TRAINED_RUN = $(RUN_ROOT)/$(MODEL)_$(APPROACH_$(SYSTEM))_final_$(DEFAULT_SPLIT)

RUN = cd "$(ROOT)" && $(PYTHON) src/run_all_v2.py --config "$(CONFIG)" --model $(MODEL) \
	--dataset $(DATASET_KEY) --marker_fold fold_$(FOLD) --seed $(SEED) --train_size $(TRAIN_SIZE) \
	--data_release "$(DATA_DIR)" --out "$(RESULTS_DIR)" --preprocessing_workers $(WORKERS)

.PHONY: help train train-all predict predict-content-words predict-transfer smoke clean \
	require-run-variables require-data require-checkpoints

help: ## List the targets of this module
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

require-run-variables:
	@test -n "$(MODEL)" || { echo "ERROR: MODEL is required (one of: $(MODELS))"; exit 1; }
	@test -n "$(filter $(MODEL),$(MODELS))" || { \
		echo "ERROR: unknown MODEL=$(MODEL) (one of: $(MODELS))"; exit 1; }
	@test -n "$(DATASET)" || { echo "ERROR: DATASET is required (snli or mnli)"; exit 1; }
	@test -n "$(filter $(DATASET_KEY),snli multi_nli)" || { \
		echo "ERROR: unknown DATASET=$(DATASET) (snli or mnli)"; exit 1; }
	@test -n "$(STEP)" || { \
		echo "ERROR: unknown SYSTEM=$(SYSTEM) (baseline, augmented, or hybrid)"; exit 1; }
	@test -n "$(filter $(FOLD),1 2 3)" || { echo "ERROR: FOLD must be 1, 2, or 3"; exit 1; }
	@case "$(TRAIN_SIZE)" in full) ;; ""|*[!0-9]*|0*) \
		echo "ERROR: TRAIN_SIZE must be 'full' or a positive integer, got '$(TRAIN_SIZE)'"; exit 1 ;; esac
	@test "$(MODEL)" != "deberta_v3_base" -o "$(SYSTEM)" = "baseline" || { \
		echo "ERROR: the paper trains only the clean baseline for deberta_v3_base"; exit 1; }

require-data:
	@test -f "$(DATA_DIR)/release_manifest.json" || { \
		echo "ERROR: no evaluation data in DATA_DIR=$(DATA_DIR). Run 'make data' (dataprep) first."; exit 1; }

require-checkpoints:
	@test -n "$(CHECKPOINTS_DIR)" || { \
		echo "ERROR: CHECKPOINTS_DIR is required (directories <model>_snli and <model>_mnli)"; exit 1; }
	@test -d "$(CHECKPOINTS_DIR)" || { \
		echo "ERROR: CHECKPOINTS_DIR=$(CHECKPOINTS_DIR) does not exist"; exit 1; }
	@test -d "$(DATA_DIR)/crossed" || { \
		echo "ERROR: $(DATA_DIR)/crossed is missing. Run 'make modify EDITS=crossed' (dataprep)."; exit 1; }

train: require-run-variables require-data ## Fine-tune one system (MODEL, DATASET, SYSTEM, FOLD, SEED, TRAIN_SIZE)
	$(RUN) --step $(STEP)

train-all: require-run-variables require-data ## Run all six systems of one encoder, dataset, and fold, as the primary analysis requires
	@test "$(MODEL)" != "deberta_v3_base" || { \
		echo "ERROR: the paper trains only the clean baseline for deberta_v3_base; use 'make train' and 'make predict'"; exit 1; }
	$(RUN) --step all --eval_split $(EVAL_SPLIT) $(REUSE)

predict: require-run-variables require-data ## Evaluate a trained system on SPLIT, with and without emoji normalization
	@test -n "$(filter $(NORMALIZATION),both with without)" || { \
		echo "ERROR: NORMALIZATION must be both, with, or without"; exit 1; }
	@test "$(SYSTEM)" = "baseline" -o "$(NORMALIZATION)" = "both" || { \
		echo "ERROR: NORMALIZATION applies to SYSTEM=baseline only; augmented runs without it, hybrid with it"; exit 1; }
	@test -d "$(TRAINED_RUN)/final_model" || { \
		echo "ERROR: no trained checkpoint at $(TRAINED_RUN)/final_model. Run 'make train' with the same variables first."; exit 1; }
ifneq ($(filter-out with,$(NORMALIZATION)),)
	$(RUN) --step $(STEP) --eval_split $(EVAL_SPLIT) $(REUSE)
endif
ifeq ($(SYSTEM),baseline)
ifneq ($(filter-out without,$(NORMALIZATION)),)
	$(RUN) --step preprocessing --eval_split $(EVAL_SPLIT) $(REUSE)
endif
endif

predict-content-words: require-checkpoints ## Run the seed-42 clean checkpoints on the nine content-word phrases
	@test -d "$(PREDICTIONS_DIR)/crossed" || { \
		echo "ERROR: $(PREDICTIONS_DIR)/crossed is missing; the clean-accuracy gate reads it"; exit 1; }
	cd "$(ROOT)" && DATA_DIR="$(DATA_DIR)" PREDICTIONS_DIR="$(PREDICTIONS_DIR)" \
		CHECKPOINTS_DIR="$(CHECKPOINTS_DIR)" $(PYTHON) analysis/content_words/run_inference.py \
		$(if $(strip $(MODEL)),--models $(MODEL))

predict-transfer: require-checkpoints ## Run the seed-42 clean checkpoints on SICK and ANLI (TRAIN=mnli or snli)
	@test -f "$(DATA_DIR)/external/sick_test.jsonl" || { \
		echo "ERROR: $(DATA_DIR)/external is missing. Run 'make download-external' (dataprep)."; exit 1; }
	cd "$(ROOT)" && DATA_DIR="$(DATA_DIR)" PREDICTIONS_DIR="$(PREDICTIONS_DIR)" \
		CHECKPOINTS_DIR="$(CHECKPOINTS_DIR)" $(PYTHON) analysis/case_study/transfer/run_inference.py \
		--train $(or $(TRAIN),mnli) $(if $(filter snli,$(TRAIN)),--datasets sick) \
		$(if $(strip $(MODEL)),--models $(MODEL))

smoke: ## Tiny CPU run: ELECTRA-small, one epoch, TRAIN_SIZE=200, a few evaluation pairs (offline)
	cd "$(ROOT)" && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		$(PYTHON) src/run_all_v2.py --config "$(ROOT)/configs/experiment_v6.json" \
		--model electra --dataset snli --step baseline --seed $(SEED) \
		--smoke_test --train_size $(SMOKE_TRAIN_SIZE) \
		--smoke_train_samples $(SMOKE_TRAIN_SIZE) --smoke_eval_samples $(SMOKE_EVAL_PAIRS) \
		$(if $(wildcard $(DATA_DIR)/release_manifest.json),--data_release "$(DATA_DIR)") \
		--out "$(SMOKE_DIR)"
	cd "$(ROOT)" && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		$(PYTHON) src/run_all_v2.py --config "$(ROOT)/configs/experiment_v6.json" \
		--model electra --dataset snli --step preprocessing --seed $(SEED) \
		--smoke_test --train_size $(SMOKE_TRAIN_SIZE) \
		--smoke_train_samples $(SMOKE_TRAIN_SIZE) --smoke_eval_samples $(SMOKE_EVAL_PAIRS) \
		$(if $(wildcard $(DATA_DIR)/release_manifest.json),--data_release "$(DATA_DIR)") \
		--out "$(SMOKE_DIR)"

clean: ## Remove the smoke-run outputs; trained runs in RESULTS_DIR are kept
	@test -n "$(SMOKE_DIR)" || { echo "ERROR: SMOKE_DIR is empty"; exit 1; }
	rm -rf "$(SMOKE_DIR)"
