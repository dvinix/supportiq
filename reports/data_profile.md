# SupportIQ — Data Profile Report

- **Total Records:** 26,872
- **Categories:** 11
- **Intents:** 27

## 1. Category Distribution

| Category | Count | Percentage |
| :--- | :--- | :--- |
| `ACCOUNT` | 5,986 | 22.28% |
| `ORDER` | 3,988 | 14.84% |
| `REFUND` | 2,992 | 11.13% |
| `CONTACT` | 1,999 | 7.44% |
| `INVOICE` | 1,999 | 7.44% |
| `PAYMENT` | 1,998 | 7.44% |
| `FEEDBACK` | 1,997 | 7.43% |
| `DELIVERY` | 1,994 | 7.42% |
| `SHIPPING` | 1,970 | 7.33% |
| `SUBSCRIPTION` | 999 | 3.72% |
| `CANCEL` | 950 | 3.54% |

## 2. Text Length Statistics

| Metric | Min | P25 | Median | Mean | P95 | P99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| instruction_chars | 6 | 40 | 48 | 46.9 | 61 | 71 | 92 |
| instruction_words | 1 | 7 | 9 | 8.7 | 13 | 14 | 16 |
| response_chars | 57 | 427 | 540 | 634.1 | 1295 | 1837 | 2472 |
| response_words | 9 | 72 | 90 | 104.8 | 206 | 300 | 402 |

## 3. Qwen Tokenizer Sequence Lengths

| Token Metric | Min | P25 | Median | Mean | P95 | P99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| instruction_tokens | 1 | 8 | 10 | 10.1 | 15 | 19 | 24 |
| response_tokens | 12 | 83 | 104 | 125.2 | 256 | 358 | 478 |
| total_tokens | 18 | 93 | 115 | 135.4 | 267 | 370 | 490 |

## 4. Placeholders & Synthetic Templating

- **Instructions with Placeholders:** 6,670 (24.82%)
- **Responses with Placeholders:** 13,006 (48.40%)
- **Unique Placeholders Detected:** 391
  - Sample entities: `{{Access Key}}`, `{{Access Key Recovery}}`, `{{Access Key Reset Page URL}}`, `{{Access Key Retrieval}}`, `{{Account}}`, `{{Account Access Key Reset}}`, `{{Account Category}}`, `{{Account Change}}`, `{{Account Closure Process}}`, `{{Account Closure Timeframe}}`

## 5. Deduplication Analysis

- **Unique Instructions:** 24,635 (2,237 exact duplicates, 8.32%)
- **Unique (Instruction, Response) Pairs:** 26,872 (0 duplicate pairs)

> [!NOTE]
> While exact pair duplication is 0%, near-duplicate instructions share underlying semantic templates.
> Data splits MUST be grouped by cluster ID to avoid train-test data leakage.
