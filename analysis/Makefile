# Analysis: from saved predictions to the numbers, tables, and figures of the paper.

SHELL := /bin/bash
.DEFAULT_GOAL := help

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON ?= python3

PREDICTIONS_DIR ?= $(ROOT)/predictions
DATA_DIR ?= $(ROOT)/data
OUTPUT_DIR ?= $(ROOT)/outputs
CHECKPOINTS_DIR ?=
TOKENIZER_MODE ?= off
SENSITIVITY_INPUT ?= $(ROOT)/analysis/primary/inputs/seed_sensitivity_compact_inputs.json

export PREDICTIONS_DIR DATA_DIR OUTPUT_DIR CHECKPOINTS_DIR
A := $(ROOT)/analysis
CROSSED_INPUTS := --source-workspace "$(PREDICTIONS_DIR)/crossed/workspace" \
	--corrected-predictions "$(PREDICTIONS_DIR)/crossed/bertweet_corrected" \
	--archive-root "$(PREDICTIONS_DIR)/fold_study"

.PHONY: help all primary run-sensitivity emoji-structure crossed crossed-validate \
	content-words case-study training-data-account figures test clean \
	require-fold-study require-crossed require-content-words require-case-study

help: ## List the targets of this module
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

define require_dir
	@test -d "$(1)" || { echo "ERROR: required input $(1) is missing. $(2)"; exit 1; }
endef
define require_file
	@test -f "$(1)" || { echo "ERROR: required input $(1) is missing. $(2)"; exit 1; }
endef
LAYOUT := See the layout of PREDICTIONS_DIR in analysis/README.md.
DATA_HINT := Run 'make data' (dataprep) or set DATA_DIR.

require-fold-study:
	$(call require_dir,$(PREDICTIONS_DIR)/fold_study,$(LAYOUT))
	$(call require_dir,$(PREDICTIONS_DIR)/deberta,$(LAYOUT))
	$(call require_file,$(DATA_DIR)/eval_sets/dataset_manifest.json,$(DATA_HINT))

require-crossed:
	$(call require_dir,$(PREDICTIONS_DIR)/fold_study,$(LAYOUT))
	$(call require_dir,$(PREDICTIONS_DIR)/crossed/workspace/predictions,$(LAYOUT))
	$(call require_dir,$(PREDICTIONS_DIR)/crossed/bertweet_corrected,$(LAYOUT))
	$(call require_dir,$(PREDICTIONS_DIR)/crossed/deberta,$(LAYOUT))
	$(call require_dir,$(DATA_DIR)/crossed,$(DATA_HINT))

require-content-words: require-crossed
	$(call require_dir,$(PREDICTIONS_DIR)/content_words/predictions,$(LAYOUT))
	$(call require_file,$(OUTPUT_DIR)/flip_rates/flip_rates_long.csv,Run 'make crossed' first.)

require-case-study: require-crossed
	$(call require_dir,$(PREDICTIONS_DIR)/transfer/predictions,$(LAYOUT))
	$(call require_file,$(DATA_DIR)/external/chaosnli_mnli.jsonl,Run 'make download-external' (dataprep).)
	$(call require_dir,$(DATA_DIR)/source,$(DATA_HINT))

all: primary emoji-structure crossed content-words case-study figures ## Run every analysis in order (without run-sensitivity and crossed-validate)

primary: require-fold-study ## Four prespecified contrasts, DeBERTa family, and Table 2
	$(PYTHON) $(A)/primary/run_primary.py --tokenizer-mode $(TOKENIZER_MODE)

run-sensitivity: ## Sensitivity to the training run; needs SENSITIVITY_INPUT, which is not distributed
	$(call require_file,$(SENSITIVITY_INPUT),This input file is not distributed with the repository. Set SENSITIVITY_INPUT to a compact input file to run this target.)
	$(PYTHON) $(A)/primary/seed_sensitivity_analysis.py \
		--input "$(SENSITIVITY_INPUT)" \
		--output-dir "$(OUTPUT_DIR)/run_sensitivity" --force

emoji-structure: require-fold-study ## Emoji loss by replacement structure and noun (Table 3)
	$(PYTHON) $(A)/emoji_structure/emoji_structure.py

crossed: require-crossed ## Crossed study: effects, component-word contrasts, flip rates
	@mkdir -p "$(OUTPUT_DIR)/crossed"
	$(PYTHON) $(A)/crossed/analyze_v2.py $(CROSSED_INPUTS) --output-dir "$(OUTPUT_DIR)/crossed"
	$(PYTHON) $(A)/crossed/flip_rates.py
	$(PYTHON) $(A)/crossed/commitment_markers.py

crossed-validate: require-crossed ## Independent recomputation of the crossed-study outputs (slow)
	$(call require_file,$(OUTPUT_DIR)/crossed/validation_v2.json,Run 'make crossed' first.)
	$(PYTHON) $(A)/crossed/validate_v2.py $(CROSSED_INPUTS) \
		--analysis-dir "$(OUTPUT_DIR)/crossed" \
		--output "$(OUTPUT_DIR)/crossed/independent_validation_v2.json"

content-words: require-content-words ## Content-word phrases and the content-word coding
	$(PYTHON) $(A)/content_words/analyze.py
	$(PYTHON) $(A)/content_words/content_word_coding.py

case-study: require-case-study training-data-account ## All-five failure, ChaosNLI subset, SICK and ANLI transfer, training-data account
	$(call require_file,$(OUTPUT_DIR)/crossed/cofailure_results_v2.csv,Run 'make crossed' first.)
	$(PYTHON) $(A)/case_study/chaosnli_subset.py
	$(PYTHON) $(A)/case_study/transfer/analyze.py --train mnli --datasets sick anli
	$(PYTHON) $(A)/case_study/transfer/analyze.py --train snli --datasets sick

training-data-account: require-crossed ## Training-data account of the on-god effect (part of case-study)
	$(call require_dir,$(DATA_DIR)/source,$(DATA_HINT))
	$(PYTHON) $(A)/case_study/training_data_account/m1_training_counts.py
	$(PYTHON) $(A)/case_study/training_data_account/m1b_small_cell_exact.py
	$(PYTHON) $(A)/case_study/training_data_account/m2_hypothesis_only_probe.py
	$(PYTHON) $(A)/case_study/training_data_account/m2b_probe_gap_bootstrap.py
	$(PYTHON) $(A)/case_study/training_data_account/m3_tokenization.py
	$(PYTHON) $(A)/case_study/training_data_account/m4_genre_breakdown.py
	$(PYTHON) $(A)/case_study/training_data_account/m4b_premise_god_sources.py

figures: ## Figures 1 and 2 (after crossed and content-words)
	$(call require_file,$(OUTPUT_DIR)/flip_rates/flip_rates_table.csv,Run 'make crossed' first.)
	$(call require_file,$(OUTPUT_DIR)/crossed/cofailure_results_v2.csv,Run 'make crossed' first.)
	$(call require_file,$(OUTPUT_DIR)/content_words/results.csv,Run 'make content-words' first.)
	$(PYTHON) $(A)/figures/make_phrase_figure.py

test: ## Unit tests of the analysis code
	cd "$(ROOT)" && $(PYTHON) -m pytest -q -p no:cacheprovider tests/test_phase6_statistics.py \
		tests/test_publication_analysis.py tests/test_deberta_ext_statistics.py \
		tests/test_seed_sensitivity_reporting.py
	cd "$(A)/crossed" && $(PYTHON) -m unittest test_analyze_v2.py test_validate_v2.py

clean: ## Remove OUTPUT_DIR
	@test -n "$(OUTPUT_DIR)" || { echo "ERROR: OUTPUT_DIR is empty"; exit 1; }
	rm -rf "$(OUTPUT_DIR)"
