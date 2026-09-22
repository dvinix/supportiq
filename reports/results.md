# SupportIQ — Master Model Benchmark Results

This table tracks all models evaluated in the project on the exact same frozen test set (`data/processed/test.parquet`, 2,688 rows, and the fixed 500-row evaluation subset).

| Model / Approach | Description | Intent Macro-F1 | Category Macro-F1 | JSON Validity | Latency (p50) | Training Cost |
|---|---|---|---|---|---|---|
| **Baseline A** | TF-IDF (word+char) + Logistic Regression | **99.93%** | **99.91%** | N/A (Classification only) | **0.15 ms** | **$0.00** |
| **Baseline B** | Qwen2.5-0.5B / Qwen3-4B (Zero/Few-Shot) | *TBD* | *TBD* | *TBD* | *TBD* | $0.00 |
| **Baseline C** | Qwen2.5-0.5B / Qwen3-4B (Structured Prompt) | *TBD* | *TBD* | *TBD* | *TBD* | $0.00 |
| **Candidate SFT** | Qwen + QLoRA Adapter (Fine-Tuned) | *TBD* | *TBD* | *TBD* | *TBD* | < $1.00 |

*Note: Per our scientific rules, all uncomputed rows are marked as `TBD` until official runs finish.*
