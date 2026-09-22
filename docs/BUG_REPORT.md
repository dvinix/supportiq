# SupportIQ — Bug & Error Log

This document records every error, failing test, and bug encountered during development, along with root causes and remediations. Documenting errors honestly is a core engineering practice for portfolio transparency and interview preparation.

---

## Bug 01 — Missing `pip` in `uv`-Generated Virtual Environment
- **Stage / Context:** Environment Setup & Dependency Verification
- **Error Encountered:**
  ```bash
  bash: line 1: pip: command not found (exit code 127)
  ```
- **Where It Happened:** Running `source .venv/bin/activate && pip install -r requirements.txt`.
- **Root Cause:** By design, Astral's `uv` creates extremely minimal virtual environments without standard `pip` pre-installed to keep creation under 10ms.
- **Remediation:** Executed `uv pip install pip` inside `.venv` to ensure developers and CLI scripts that rely on standard `pip` can execute without failure.
- **Prevention:** Documented both `pip install -r requirements.txt` and `uv sync` in `requirements.txt` and `README.md`.

---

## Bug 02 — Relative Path Failure When Notebook Executed Inside `notebooks/`
- **Stage / Context:** Subtask 1.2 (`02_profile.ipynb` execution)
- **Error Encountered:**
  ```python
  FileNotFoundError: Raw dataset not found at data/raw/bitext_raw.parquet. Run ingest_raw_dataset first.
  ```
- **Where It Happened:** In `load_raw_dataframe("data/raw")` called from `notebooks/02_profile.ipynb`.
- **Root Cause:** When Jupyter / `nbconvert` executes a notebook inside the `notebooks/` directory, the kernel's working directory is set to `/home/dvinix/Projects/supportiq/notebooks`, not the project root. Therefore, relative path `data/raw` resolved to `notebooks/data/raw`, which does not exist.
- **Remediation:** Enhanced `src/supportiq/data/load.py` in `load_raw_dataframe()`:
  ```python
  parquet_path = Path(data_dir) / "bitext_raw.parquet"
  if not parquet_path.exists():
      alt_path = Path("..") / data_dir / "bitext_raw.parquet"
      if alt_path.exists():
          parquet_path = alt_path
      else:
          raise FileNotFoundError(...)
  ```
  The loader now resolves data paths seamlessly whether called from the project root CLI or from inside the `notebooks/` subfolder.

---

## Bug 03 — `TypeError` in Structured Logger Due to Keyword Arguments
- **Stage / Context:** Subtask 1.5 (`src/supportiq/data/profile.py` unit testing)
- **Error Encountered:**
  ```python
  TypeError: Logger._log() got an unexpected keyword argument 'model'
  ```
- **Where It Happened:** In `compute_token_length_stats()` at line 187:
  `logger.info("Loading tokenizer for sequence length profiling", model=tokenizer_name)`.
- **Root Cause:** Python standard library `logging.Logger` methods (`info`, `warning`, `error`) only accept positional arguments for message formatting (`%s`), keyword arguments for standard options (`exc_info`, `stack_info`), and custom dictionary data via `extra={"extra": ...}`. Arbitrary keyword arguments like `model=...` are rejected by `_log()`.
- **Remediation:** Changed logger invocations in `load.py` and `profile.py` to standard string formatting:
  `logger.info("Loading tokenizer for sequence length profiling: %s", tokenizer_name)`.

---

## Bug 04 — Syntax Error in `profile.py` (`lines.extend`)
- **Stage / Context:** Subtask 1.5 (`src/supportiq/data/profile.py` testing)
- **Error Encountered:**
  ```python
  SyntaxError: '(' was never closed
  ```
- **Where It Happened:** In `generate_markdown_report()`: a closing bracket was `]` instead of `])` on `lines.extend([...])`.
- **Root Cause:** Typographical error during file modification.
- **Remediation:** Closed the bracket properly as `lines.extend([...])`. Verified immediately with `python -m py_compile` and `ruff check`.

---

## Bug 05 — Assertion Failure in `test_generate_markdown_and_run_profile`
- **Stage / Context:** Subtask 1.5 (`tests/data/test_profile.py`)
- **Error Encountered:**
  ```python
  FAILED tests/data/test_profile.py::test_generate_markdown_and_run_profile - AssertionError: assert 'Order Number' in md
  ```
- **Where It Happened:** Line 98 of `tests/data/test_profile.py`: `assert "Order Number" in md`.
- **Root Cause:** The profiling markdown generator originally printed the numeric count of placeholders (`2`), but did not print the actual extracted entity names (e.g. `Order Number`, `Invoice ID`).
- **Remediation:** Updated `generate_markdown_report()` to render a sample of detected entity names into markdown table/bullet points:
  `lines.append("  - Sample entities: " + ", ".join(f"`{{{{{p}}}}}`" for p in report.placeholders.unique_placeholders[:10]))`.
  The test now passes, and the generated markdown report is significantly more informative for stakeholders.

---

## Bug 06 — `ModuleNotFoundError: No module named 'schemas'` During Pytest Collection
- **Stage / Context:** Subtask 2.1 (`tests/data/test_validate.py`)
- **Error Encountered:**
  ```python
  ImportError while importing test module 'tests/data/test_validate.py'
  ModuleNotFoundError: No module named 'schemas'
  ```
- **Where It Happened:** Importing `from schemas.dataset import CustomerSupportRecord` in test files.
- **Root Cause:** In Python packaging, only directories inside `src/supportiq` are installed in editable site-packages by hatchling. Root-level directories (like `schemas/`) are not in `sys.path` by default when pytest executes unless explicitly configured.
- **Remediation:**
  1. Created `schemas/__init__.py` to make `schemas` a proper Python package.
  2. Added `pythonpath = ["."]` under `[tool.pytest.ini_options]` in `pyproject.toml`.
  Pytest now discovers and imports root schemas seamlessly in local and CI environments.

---

## Bug 07 — `ModuleNotFoundError: No module named 'schemas'` When Running `02_profile.ipynb`
- **Stage / Context:** Subtask 1.2 (`notebooks/02_profile.ipynb` execution from Jupyter / IDE)
- **Error Encountered:**
  ```python
  ModuleNotFoundError: No module named 'schemas'
  ```
- **Where It Happened:** Cell 1 of `notebooks/02_profile.ipynb` when executing `from supportiq.data.load import load_raw_dataframe`.
- **Root Cause:** Importing `supportiq.data.load` triggers `supportiq.data.__init__.py`, which in turn imports `validate.py`. In `validate.py`, `from schemas.dataset import ...` looked for `schemas` in `sys.path`. When Jupyter runs inside the `notebooks/` directory, Python sets the working directory and first search path to `notebooks/`, so the project root (where `schemas/` lives) was absent from `sys.path`.
- **Remediation:**
  1. Added a fallback in `src/supportiq/data/validate.py` to dynamically discover and inject the project root into `sys.path` if `schemas` is not found.
  2. Added an explicit `sys.path` bootstrap guard in Cell 1 of `notebooks/02_profile.ipynb`.
  3. Registered the virtual environment kernel via `python -m ipykernel install --user --name supportiq-env --display-name "Python (.venv supportiq)"` so that VS Code and Jupyter bind to the correct kernel.

---

## Bug 08 — Harmless Stderr Warning: `[transformers] PyTorch was not found...` on Import
- **Stage / Context:** Stage 1 & Stage 2 Data Pipeline Notebooks (`01_load_and_inspect.ipynb`, `03_validate_and_quarantine.ipynb`)
- **Error / Message Encountered:**
  ```text
  [transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
  ```
- **Where It Happened:** Block 1 in data processing notebooks when importing from `supportiq.data`.
- **Root Cause:**
  1. In Stage 1 & 2 (Data Engineering), we only need lightweight BPE tokenization (Qwen tokenizer) to profile token distributions, so PyTorch (a 2GB+ deep learning framework) is omitted to keep the environment lightweight until Phase 3 (Fine-Tuning).
  2. In `src/supportiq/data/__init__.py`, `profile.py` was imported. In `profile.py`, `from transformers import AutoTokenizer` was placed at the top level.
  3. Whenever `transformers` is imported without PyTorch installed, it prints an advisory notification to `stderr`. In Jupyter notebooks, `stderr` messages appear highlighted or reddish, confusing beginners into thinking the cell crashed.
- **Remediation:**
  1. Converted `AutoTokenizer` into a **lazy import** inside `compute_token_length_stats()` in `src/supportiq/data/profile.py`, so `transformers` is never imported unless token length profiling is explicitly invoked.
  2. Set `os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"` before importing, suppressing the advisory notice entirely.
  3. Notebook cell 1 now runs completely clean with zero stderr warnings.
