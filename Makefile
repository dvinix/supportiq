.PHONY: help setup lint format test clean data baseline serve docker-build

PYTHON ?= 3.12
UV ?= uv

help:
	@echo "SupportIQ Development Commands:"
	@echo "  make setup        - Install dependencies using uv"
	@echo "  make lint         - Run linter (ruff) and type/format checks"
	@echo "  make format       - Format code with ruff"
	@echo "  make test         - Run pytest suite"
	@echo "  make clean        - Remove caches and build artifacts"
	@echo "  make data         - Run data pipeline"
	@echo "  make baseline     - Run baseline models"
	@echo "  make serve        - Start API server"
	@echo "  make docker-build - Build API container"

setup:
	$(UV) python pin $(PYTHON)
	$(UV) sync --all-groups
	$(UV) run pre-commit install

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:
	$(UV) run pytest

clean:
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
