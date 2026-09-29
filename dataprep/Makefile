# Data preparation: source datasets and edited evaluation sets.

SHELL := /bin/bash
.DEFAULT_GOAL := help

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON ?= python3

DATA_DIR ?= $(ROOT)/data
SEED ?= 42
CONFIG ?= $(ROOT)/configs/experiment_v6.json
WORKERS ?= 0
EDITS ?=
DATASETS ?=
SPLITS ?=
SNLI_REVISION ?= cdb5c3d5eed6ead6e5a341c8e56e669bb666725b
MNLI_REVISION ?= da70db2af9d09693783c3320c4249840212ee221

DATAPREP := $(PYTHON) $(ROOT)/dataprep/dataprep.py --data-dir "$(DATA_DIR)" --config "$(CONFIG)"
DATA_TESTS := tests/test_phase1_design.py tests/test_phase3_challenge_sets.py \
	tests/test_data_release.py tests/test_psych_variant.py tests/test_dataprep.py

.PHONY: help download download-external modify verify clean distclean require-source

help: ## List the targets of this module
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

require-source:
	@test -f "$(DATA_DIR)/source_manifest.json" || { \
		echo "ERROR: no source data in DATA_DIR=$(DATA_DIR). Run 'make download' first."; exit 1; }

download: ## Fetch SNLI and MultiNLI into DATA_DIR (needs network)
	@test ! -e "$(DATA_DIR)" || { \
		echo "ERROR: DATA_DIR=$(DATA_DIR) already exists. Choose a new directory or run 'make distclean'."; exit 1; }
	$(DATAPREP) download --workers $(WORKERS) \
		--revision snli=$(SNLI_REVISION) --revision multi_nli=$(MNLI_REVISION)

download-external: require-source ## Fetch ChaosNLI, SICK, and ANLI for the case study (needs network)
	$(DATAPREP) external

modify: require-source ## Build the edited evaluation sets (EDITS, DATASETS, SPLITS limit the work)
	$(DATAPREP) modify --seed $(SEED) --workers $(WORKERS) \
		$(if $(strip $(EDITS)),--edits $(EDITS)) \
		$(if $(strip $(DATASETS)),--datasets $(DATASETS)) \
		$(if $(strip $(SPLITS)),--splits $(SPLITS))

verify: require-source ## Run the unit tests for the data code and check the generated data
	cd "$(ROOT)" && $(PYTHON) -m pytest -q -p no:cacheprovider $(DATA_TESTS)
	$(DATAPREP) verify

clean: ## Remove the generated evaluation sets; keep the downloaded source data
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR is empty"; exit 1; }
	rm -rf "$(DATA_DIR)/eval_sets" "$(DATA_DIR)/crossed" "$(DATA_DIR)/content_words" \
		"$(DATA_DIR)/release_manifest.json"

distclean: ## Remove DATA_DIR entirely, including the downloaded source data
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR is empty"; exit 1; }
	rm -rf "$(DATA_DIR)"
