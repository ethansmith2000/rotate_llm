PYTHON ?= python3
VENV := .venv
VENV_PY := $(VENV)/bin/python
MAX_WORDS ?= 500

.PHONY: setup smoke run help clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

smoke: setup
	$(VENV_PY) run_experiment.py --smoke --max-words $(MAX_WORDS)

run: setup
	$(VENV_PY) run_experiment.py --max-words $(MAX_WORDS)

help: setup
	$(VENV_PY) run_experiment.py --help

clean:
	rm -rf __pycache__ results_smoke
