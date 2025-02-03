#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = company_success_prediction
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 text_based_company_success_prediction
	isort --check --diff --profile black text_based_company_success_prediction
	black --check --config pyproject.toml text_based_company_success_prediction

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml text_based_company_success_prediction

## Set up python interpreter environment (macOS/Linux compatible)
.PHONY: create_environment
create_environment:
	@echo ">>> Creating virtual environment: venv"
	$(PYTHON_INTERPRETER) -m venv venv
	@echo ">>> Virtual environment created. Activate with:\nsource venv/bin/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) text_based_company_success_prediction/dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
