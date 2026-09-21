# Stage 1 Report — Explore the Data (Notebook-First)

## 1. Overview
Stage 1 explored and characterized the raw Bitext customer-support dataset following the notebook-first loop specified in `AGENT_GUIDE_v2.md`. All findings were verified in notebooks, extracted into reusable Python modules, tested with unit tests, and documented with computed metrics.

---

## 2. What Was Built
1. **Exploratory Notebooks**:
   - `notebooks/01_load_and_inspect.ipynb`: Ingestion of raw dataset from Hugging Face Hub, commit revision pinning, raw preservation in `data/raw/`, and generation of `data/raw/METADATA.json`.
   - `notebooks/02_profile.ipynb`: Full profiling including class distributions, text and token length distributions with the Qwen tokenizer, placeholder pattern detection, and MinHash-LSH near-duplicate clustering.
2. **Modular Library Code**:
   - `src/supportiq/data/load.py`: `ingest_raw_dataset()`, `load_raw_dataframe()`, and `compute_file_sha256()`.
   - `src/supportiq/data/profile.py`: Statistical percentiles, class distributions, deduplication metrics, placeholder detector, and markdown report generator.
3. **Unit Test Suite**:
   - `tests/data/test_load.py`: 4 tests covering checksum verification, metadata serialization, and DataFrame loading.
   - `tests/data/test_profile.py`: 5 tests covering distributions, length statistics, deduplication, placeholder detection, and report generation.
4. **Reports & Documentation**:
   - `reports/data_profile.json` & `reports/data_profile.md`: Real computed metrics.
   - `docs/DATA_CARD.md`: Dataset card covering provenance, structure, profiling metrics, templating analysis, and limitations.

---

## 3. Real Computed Results
- **Dataset Size:** 26,872 rows, 5 columns (`flags`, `instruction`, `category`, `intent`, `response`).
- **Nulls / Missing:** 0 nulls, 0 empty strings across all fields.
- **Revision SHA:** `430d1a89bd93bd1fa23c16f29dd53e73f0087443` (CDLA-Sharing-1.0 license).
- **Class Balance:** 11 categories (`ACCOUNT`: 22.28% down to `CANCEL`: 3.54%), 27 intents (~950 to 1,000 samples per intent, median 995).
- **Sequence Lengths (Qwen Tokenizer):**
  - Instructions: median 10 tokens, P95 15 tokens, max 24 tokens.
  - Responses: median 104 tokens, P95 256 tokens, max 478 tokens.
  - Total tokens: median 115 tokens, P95 267 tokens, max 490 tokens.
  - Truncation rate at `max_seq_length = 512`: **0.00%**.
- **Duplicates & Templating:**
  - Exact duplicate instructions: 2,237 (8.32%).
  - Exact duplicate (instruction, response) pairs: 0 (0.00%).
  - Instructions with placeholders (`{{...}}`): 6,670 (24.82%).
  - Responses with placeholders: 13,006 (48.40%).
  - 391 unique placeholder entity types detected.
  - MinHash-LSH (threshold=0.80) shows template clusters averaging 4.7 paraphrases per seed.

---

## 4. Architectural & Scientific Decisions
- **ADR-003: Grouped Splitting by Cluster ID**:
  Because Bitext instructions are synthetic paraphrases of common seed templates, a random split would cause template leakage between train and test sets. Splitting must be grouped by MinHash cluster ID.
- **Preservation of Slot Placeholders**:
  Expressions matching `{{...}}` are intentional slot variables in customer service prompts. They must be preserved during normalization and not stripped or redacted by PII cleaners.

---

## 5. Commands to Reproduce
```bash
# Run unit tests
uv run pytest tests/data/ -v

# Run linting and formatting checks
uv run ruff check .
uv run ruff format --check .

# Re-run profiling pipeline
uv run python -c "from supportiq.data.load import load_raw_dataframe; from supportiq.data.profile import run_full_profile; run_full_profile(load_raw_dataframe('data/raw'))"
```

---

## 6. Cloud Spend So Far
- **$0.00** (All Stage 1 data loading, tokenization, and profiling executed locally on CPU).

---

## 7. Check Your Understanding
Answer these questions in your own words in `docs/LEARNING_LOG.md`:
1. What is class imbalance, does it exist in the Bitext dataset, and why does intent balance look unusually uniform here?
2. Why does a naive random train/test split threaten the validity of evaluation on a templated dataset like Bitext?
3. How do the measured Qwen token lengths (median ~115, max 490) justify setting `max_seq_length = 512` for fine-tuning?
4. Why must dataset placeholders like `{{Order Number}}` be treated differently from real customer PII?
