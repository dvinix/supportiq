# Phase 0 Report — Foundation

## Overview
Phase 0 establishes a clean, reproducible, and CI-enabled repository skeleton for SupportIQ in accordance with `AGENT_GUIDE.md` and `docs/SupportIQ_End_to_End_Project_Plan.md`.

## What Was Built
1. **Repository Layout & Version Control**:
   - Git repository initialized on `master` branch.
   - Pinned Python runtime to 3.12 using Astral's `uv`.
   - Structured directory hierarchy: `configs/`, `data/` (`raw/`, `interim/`, `processed/`, `splits/`), `docs/` (`phase_reports/`), `notebooks/kaggle/`, `reports/`, `schemas/`, `scripts/`, `src/supportiq/` (`core/`, `data/`, `training/`, `evaluation/`, `inference/`, `monitoring/`), `app/`, `tests/`, `infrastructure/terraform/`, and `docker/`.
   - `.gitignore` configured to guard against committing large datasets, secrets, binary models, and cache directories.
   - Gitkeep files added to protect data split directory structure.

2. **Dependency Management**:
   - `pyproject.toml` with `hatchling` build system and dependency groups:
     - `dev`: `pytest`, `pytest-cov`, `ruff`, `pre-commit`, `detect-secrets`
     - `data`: `polars`, `pyarrow`, `datasets`, `scikit-learn`, `presidio-analyzer`, `presidio-anonymizer`, `spacy`, `datasketch`, `dvc`
     - `train`: `torch`, `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `mlflow`
     - `serve`: `fastapi`, `uvicorn`, `httpx`, `prometheus-client`
   - `.env.example` created documenting all environment variables without committing secrets.

3. **Core Library Utilities**:
   - `src/supportiq/core/config.py`: YAML loader with Pydantic v2 `BaseModel` / `BaseSettings` schema validation and environment overrides.
   - `src/supportiq/core/logger.py`: Structured JSON logger producing timestamped, machine-readable logs.
   - `src/supportiq/core/seed.py`: Deterministic seed utility setting seeds across `random`, `numpy`, and `torch`.

4. **Configurations**:
   - `configs/data.yaml`: Dataset specifications, split ratios (80/10/10), MinHash deduplication parameters, and PII entities.
   - `configs/training.yaml`: Qwen base model hyperparameters, LoRA target modules, 4-bit NF4 quantization settings, and training parameters.
   - `configs/evaluation.yaml`: Quality gates, metric targets, and evaluation paths.

5. **Code Quality & CI**:
   - `.pre-commit-config.yaml`: Pre-commit hooks for trailing whitespace, EOF formatting, YAML/TOML validation, `ruff`, `ruff-format`, and `detect-secrets`.
   - `.secrets.baseline`: Generated via `detect-secrets scan`.
   - `.github/workflows/ci.yml`: GitHub Actions CI pipeline running linting, formatting, and unit tests with coverage on PR/push.
   - `Makefile`: Standard targets (`setup`, `lint`, `format`, `test`, `clean`).
   - `tests/`: 9 unit tests covering configuration validation, JSON logging, and seed determinism.

6. **Documentation**:
   - `README.md`: Project summary, architecture, 12-phase status checklist, quickstart, and directory tour.
   - `docs/DECISIONS.md`: Initial Architecture Decision Records (ADR-001 for Python 3.12 & `uv`, ADR-002 for Structured JSON Logging).

## Commands to Reproduce
```bash
# Set up dependencies and git hooks
make setup

# Run linting and format checks
make lint

# Run unit tests
make test

# Run pre-commit hooks
uv run pre-commit run --all-files
```

## Real Computed Results
- **Unit Test Suite:** 9 passed in 0.19s (100% pass rate).
- **Linter & Formatter:** 0 errors across all 11 files (`ruff check` clean).
- **Pre-commit Hooks:** 9/9 hooks passed (`detect-secrets`, `ruff`, `check-yaml`, etc.).

## Problems Found & Remediations
1. *Host Python Version (3.14)*: The system default Python is 3.14.7, which lacks binary wheels for PyTorch, TRL, and related ML libraries.
   - *Resolution*: Pinned Python to 3.12 (`.python-version`) using `uv` which isolates our environment with pre-built CPython 3.12.
2. *Hatchling Editable Build Requires README*: Editable installation initially failed because `README.md` and package init had not yet been created.
   - *Resolution*: Created `README.md` and `src/supportiq/__init__.py`, allowing `hatchling` to build smoothly.
3. *PEP 695 Type Parameters*: Ruff flagged Python 3.12 modern type parameter syntax (`UP047`).
   - *Resolution*: Adopted PEP 695 generic function syntax `def load_config[T: BaseModel](...)`.

## Architecture Decisions Made
- **ADR-001:** Python 3.12 & `uv` Package Manager.
- **ADR-002:** Structured JSON Logging & Shared Logger.

## AWS Spend So Far
- **$0.00** (All Phase 0 work executed locally on CPU).
