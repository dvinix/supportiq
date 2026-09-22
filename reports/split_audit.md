# SupportIQ — Split & Data Leakage Audit Report

**Date:** 2026-09-22  
**Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`  
**Total Raw Rows:** 26,872  
**Clustering Algorithm:** MinHash LSH (3-word shingles, Jaccard threshold = 0.80, num_perm = 64)  
**Total Unique Template Clusters:** 23,297  

---

## 1. Split Distribution

| Split | Rows | Row Share | Unique Clusters | Target Ratio |
|---|---|---|---|---|
| **Train** | 21,497 | 80.0% | 18,637 | 80.0% |
| **Validation** | 2,687 | 10.0% | 2,330 | 10.0% |
| **Test** | 2,688 | 10.0% | 2,330 | 10.0% |
| **Test Eval Subset** | 500 | - | - | Fixed benchmark |
| **Total** | **26,872** | **100.0%** | **23,297** | **100.0%** |

---

## 2. Mathematical Leakage Audit

To prevent the LLM from scoring artificially high by memorizing templated paraphrases, splits are strictly partitioned by `cluster_id`:

- **Train $\cap$ Validation Overlap:** 0 clusters
- **Train $\cap$ Test Overlap:** 0 clusters
- **Validation $\cap$ Test Overlap:** 0 clusters
- **Status:** **PASS (100% Leakage-Free)**

---

## 3. Test Set Integrity & Checksum
The evaluation test set has been cryptographically frozen to guarantee benchmark reproducibility:
- **File:** `data/splits/test.parquet`
- **SHA-256:** `03eba06c98db4047ac264c9aee8cd27b56f18f670b0fc8473c4413278b5d1edd`

---

## 4. Fine-Tuning Artifacts
All splits are exported in standard conversational format for Qwen SFT fine-tuning:
- `data/processed/train.jsonl` (21,497 dialogues, 22.41 MB)
- `data/processed/val.jsonl` (2,687 dialogues, 2.80 MB)
- `data/processed/test.jsonl` (2,688 dialogues, 2.83 MB)
- `data/processed/test_eval_500.jsonl` (500 dialogues, 0.53 MB)
