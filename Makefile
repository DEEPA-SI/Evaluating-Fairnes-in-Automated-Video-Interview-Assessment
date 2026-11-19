PYTHON ?= python
OUTPUT_DIR ?= build/reproduction
export MPLBACKEND := Agg
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1

.PHONY: all verify reproduce test lint

all: verify reproduce

verify:
	$(PYTHON) -m scripts.publication_manifest --check

reproduce:
	$(PYTHON) -m src.publication --output-dir "$(OUTPUT_DIR)"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests
