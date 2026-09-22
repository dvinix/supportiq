# SupportIQ — Baseline A Evaluation Report

**Model:** TF-IDF (Word n-grams [1, 2] + Character n-grams [3, 5]) + Logistic Regression ($C=1.0$)  
**Date:** 2026-09-23  
**Training Set:** `data/processed/train.parquet` (21,497 rows, grouped by `cluster_id`)  
**Test Set:** `data/processed/test.parquet` (2,688 rows, frozen test set)  
**Execution Environment:** CPU (x86_64, Linux) — zero GPU required  

---

## 1. Summary of Quantitative Results

| Task | Test Accuracy | Macro-F1 | Weighted-F1 | Inference Latency (p50) |
|---|---|---|---|---|
| **Intent Classification (27 classes)** | **99.93%** | **99.93%** | **99.93%** | ~0.15 ms / request |
| **Category Classification (11 classes)** | **99.93%** | **99.91%** | **99.93%** | ~0.12 ms / request |

- **Total Test Samples:** 2,688
- **Errors on Test Set:** Only 2 out of 2,688 records misclassified.
- **Model Size:** 18.2 MB (`artifacts/baseline_tfidf.joblib`)

---

## 2. Scientific & Architectural Insights

### Why does a simple linear model achieve 99.9% accuracy?
1. **Keyword Distinctiveness:** The Bitext customer support dataset has clean, distinct lexical indicators for each intent (e.g. *"cancel order"*, *"forgot password"*, *"change address"*).
2. **Feature Coverage:** Sub-word character n-grams (3-5 chars) combined with word n-grams (1-2 words) create ~30,000 distinct signals, making linear separation straightforward in-domain.

### The Critical Trade-off: Why Fine-Tune an LLM If TF-IDF Scores 99.9%?
This finding provides a powerful senior engineering talking point for portfolios and interviews:

| Capability | Baseline A (TF-IDF + Logistic Regression) | Candidate (Fine-Tuned Qwen LLM) |
|---|---|---|
| **Intent & Category Classification** | 99.9% (Near-perfect in-domain) | High expected accuracy |
| **Conversational Response Generation** | **CANNOT DO** (Static lookup only) | **YES** (Context-aware, empathetic, helpful answers) |
| **Structured Output Generation** | Requires hardcoded string templating | Native JSON with slot filling |
| **Out-of-Domain Generalization** | **Brittle** (Fails if phrasing changes completely) | **High** (Understands semantics and intent nuances) |
| **Compute Cost** | $0 (Runs on cheap CPU) | Requires GPU for training; can be served quantized |

---

## 3. The Next Benchmark: Baselines B & C
Now that Baseline A has set the classification ceiling for in-domain data, our subsequent evaluations will measure:
- **Baseline B:** Zero/Few-shot prompted base Qwen.
- **Baseline C:** Structured JSON prompted base Qwen.
- **Candidate:** Qwen + SFT + QLoRA fine-tuned adapter.
