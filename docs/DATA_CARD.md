# SupportIQ — Dataset Card (Bitext Customer Support Dataset)

## 1. Dataset Summary & Provenance

- **Dataset Name:** Bitext Customer Service Tagged Training Dataset for LLM-based Virtual Assistants
- **Hugging Face Hub Identifier:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- **Revision SHA:** `430d1a89bd93bd1fa23c16f29dd53e73f0087443`
- **License:** CDLA-Sharing-1.0 (Community Data License Agreement – Sharing, Version 1.0)
- **Primary Domain:** E-commerce / Retail Customer Service
- **Ingestion Date (UTC):** 2026-09-21
- **Preserved Raw Artifacts:**
  - `data/raw/bitext_raw.parquet` (SHA-256: `0c8ae53ede333a008aa821ee287c93a91d166b6f80cb81a3d665fa55d9a48782`, 3.19 MB)
  - `data/raw/bitext_raw.csv` (SHA-256: `6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434`, 19.20 MB)
  - `data/raw/METADATA.json`

---

## 2. Dataset Structure & Fields

The raw dataset contains **26,872** rows and 5 string columns:

| Field | Type | Description |
| :--- | :--- | :--- |
| `flags` | string | Bitext generation metadata flags (e.g. `'B'`, `'Q'`, etc.) |
| `instruction` | string | Customer query or message |
| `category` | string | High-level domain category (11 distinct values) |
| `intent` | string | Specific customer action or request intent (27 distinct values) |
| `response` | string | Agent/assistant response text |

---

## 3. Data Profile & Real Computed Metrics

All statistics below are computed directly from the raw dataset via `src/supportiq/data/profile.py` (see [reports/data_profile.md](file:///home/dvinix/Projects/supportiq/reports/data_profile.md)):

### Quality & Missing Values
- **Null values:** 0 across all columns (0.00%).
- **Empty or whitespace-only text:** 0 rows (0.00%).
- **Exact duplicate instructions:** 2,237 (8.32%).
- **Exact duplicate (instruction, response) pairs:** 0 (0.00%).

### Taxonomy & Class Balance
- **11 Categories:**
  `ACCOUNT` (22.28%), `ORDER` (14.84%), `REFUND` (11.13%), `CONTACT` (7.44%), `INVOICE` (7.44%), `PAYMENT` (7.44%), `FEEDBACK` (7.43%), `DELIVERY` (7.42%), `SHIPPING` (7.33%), `SUBSCRIPTION` (3.72%), `CANCEL` (3.54%).
- **27 Intents:**
  Intents are synthetically balanced, ranging between 950 and 1,000 samples per intent (median: ~995 rows).

### Sequence Lengths & Token Statistics

| Metric | Min | P25 | Median | Mean | P95 | P99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Instruction Chars** | 6 | 40 | 48 | 46.9 | 61 | 71 | 92 |
| **Instruction Words** | 1 | 7 | 9 | 8.7 | 13 | 14 | 16 |
| **Response Chars** | 57 | 427 | 540 | 634.1 | 1,295 | 1,837 | 2,472 |
| **Response Words** | 9 | 72 | 90 | 104.8 | 206 | 300 | 402 |
| **Instruction Tokens (Qwen)** | 1 | 8 | 10 | 10.1 | 15 | 19 | 24 |
| **Response Tokens (Qwen)** | 12 | 83 | 104 | 125.2 | 256 | 358 | 478 |
| **Total Tokens (Inst + Resp)** | 18 | 93 | 115 | 135.4 | 267 | 370 | 490 |

> [!TIP]
> **Sequence Length Selection:** With a sequence length limit of `max_seq_length = 512`, the truncation rate across all 26,872 examples is **0.00%** (max observed token length is 490 tokens).

---

## 4. Templating & Near-Duplicate Analysis

- **Slot Placeholders:**
  - 391 distinct placeholder entities detected (e.g. `{{Order Number}}`, `{{Account ID}}`, `{{Email}}`).
  - Found in **24.82%** of customer instructions and **48.40%** of assistant responses.
  - **Decision:** Placeholders represent intentional slots and must be preserved during normalization and PII scrubbing, never over-redacted.
- **Paraphrase Clusters:**
  - Instructions are generated via templated paraphrasing around common root intents.
  - MinHash-LSH clustering (threshold = 0.80, 128 permutations) reveals an average of **4.7** matching paraphrases per instruction (P95: 9, max: 17).
  
> [!IMPORTANT]
> **Leakage Hazard & Grouped Splitting:**
> If a standard random train/test split is applied, near-duplicate paraphrases will leak into both sets, artificially inflating baseline and model metrics.
> In Stage 2, splits **must be grouped by `cluster_id`** to guarantee that every template family is isolated to either train, validation, or test.

---

## 5. Limitations & Ethical Considerations

1. **Synthetic Nature:** Bitext is synthetically generated and template-balanced. It does not reflect real-world user distributions, non-standard slang, heavy typos, or multi-turn dialogues.
2. **Evaluation Boundary:** Bitext will serve for training and in-domain evaluation. Out-of-domain robustness will be tested in later stages using BANKING77 as an external evaluation benchmark (never mixed into training).
