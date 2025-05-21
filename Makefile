# Signifies our desired python version
PYTHON := python

# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY: help install install-package correct lint type-check security test unit integration all

# Defines the default target that `make` will to try to make, or in the case of a phony target, execute the specified commands
# This target is executed whenever we just type `make`
.DEFAULT_GOAL := all

# The @ makes sure that the command itself isn't echoed in the terminal
# This target parses the comments on each target to create a help menu
help: # Print help on Makefile
	@echo "---------------HELP-----------------"
	@grep '^[^.#]\+:\s\+.*#' Makefile | \
	sed "s/\(.\+\):\s*\(.*\) #\s*\(.*\)/`printf "\033[93m"`\1`printf "\033[0m"`	\3/" | \
	expand -t20
	@echo "------------------------------------"

install: # Install ci dependencies
	${PYTHON} -m pip install --upgrade setuptools wheel
	${PYTHON} -m pip install -r ci-requirements.txt

install-package: check-env # Install package
	${PYTHON} -m pip install --extra-index-url $(INDEX_URL) .

correct: # Auto format code
	${PYTHON} -m black --line-length=120 xsell_dental_exemplo
	${PYTHON} -m black --line-length=120 tests
	${PYTHON} -m isort xsell_dental_exemplo

lint: correct # Lint and static-check
	${PYTHON} -m flake8 xsell_dental_exemplo
	${PYTHON} -m pylint xsell_dental_exemplo
	${PYTHON} -m interrogate -v xsell_dental_exemplo

type-check: # Type check
	${PYTHON} -m mypy xsell_dental_exemplo --ignore-missing-imports

security: # Security check
	${PYTHON} -m bandit -r xsell_dental_exemplo

test: unit integration # Run all tests

unit: # Run unit tests
	${PYTHON} -m pytest tests/unit/ --junitxml=junit/test-results-unit.xml --cov-report xml --cov=xsell_dental_exemplo/ --cov-branch
	${PYTHON} -m coverage report -m

integration: # Run integration tests
	${PYTHON} -m pytest tests/integration/ --junitxml=junit/test-results-integration.xml

all: correct lint security test # Run all linters and tests