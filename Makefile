# Top-level entry point. Each module has its own Makefile and README.

SHELL := /bin/bash
.DEFAULT_GOAL := help

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
MODULES := dataprep training analysis

.PHONY: help data train analysis test clean

help: ## List every target of every module
	@echo "Top level (make <target>):"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(ROOT)/Makefile
	@for module in $(MODULES); do \
		echo; echo "$$module (make -C $$module <target>):"; \
		awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(ROOT)/$$module/Makefile; \
	done

data: ## dataprep: download, modify, verify
	$(MAKE) -C $(ROOT)/dataprep download
	$(MAKE) -C $(ROOT)/dataprep modify
	$(MAKE) -C $(ROOT)/dataprep verify

train: ## training: fine-tune one system (MODEL, DATASET, SYSTEM, FOLD, SEED, TRAIN_SIZE)
	$(MAKE) -C $(ROOT)/training train

analysis: ## analysis: every analysis target, in order (PREDICTIONS_DIR)
	$(MAKE) -C $(ROOT)/analysis all

test: ## Run all unit tests
	cd "$(ROOT)" && $(PYTHON) -m pytest -q -p no:cacheprovider tests
	cd "$(ROOT)/analysis/crossed" && $(PYTHON) -m unittest test_analyze_v2.py test_validate_v2.py

clean: ## Remove generated evaluation sets, smoke outputs, and analysis outputs
	$(MAKE) -C $(ROOT)/dataprep clean
	$(MAKE) -C $(ROOT)/training clean
	$(MAKE) -C $(ROOT)/analysis clean
