# SupportIQ

> **Production-grade LLM fine-tuning system for customer support triage with strict JSON schema outputs.**

---

## 🎯 Goal

Given a customer support message, output strict JSON `{"category", "intent", "response"}`. SupportIQ compares 4 approaches to rigorously prove (or disprove) that LLM fine-tuning delivers measurable business value over classical or prompting baselines:

1. **Baseline A:** TF-IDF + Logistic Regression (intent & category)
2. **Baseline B:** Qwen3-4B-Base, zero/few-shot
3. **Baseline C:** Qwen3-4B-Base + structured output prompt
4. **Candidate:** Qwen3-4B-Base + SFT + QLoRA

---

## 📋 Implementation Status

- [x] **Phase 0: Foundation (laptop)**
  - Clean repository skeleton, dependency management (`uv`), pre-commit hooks, CI workflow, and base testing.
- [ ] **Phase 1: Data Pipeline (laptop, CPU only)**
  - Ingestion, data profiling, Pydantic schema validation, Presidio PII anonymization, normalization, MinHash near-duplicate clustering, taxonomy derivation, leakage-safe grouped split, SFT formatting, and DVC pipeline.
- [ ] **Phase 2: Classical Baseline (laptop, CPU)**
  - TF-IDF + Logistic Regression baseline, shared evaluation harness, bootstrap confidence intervals, and grouped vs. random split leakage audit.
- [ ] **Phase 3: LLM Code & Free-GPU Development (Kaggle/Colab)**
  - TRL SFTTrainer + PEFT QLoRA pipeline, resumable checkpointing, prompt engineering for baselines B/C, and smoke test on small model.
- [ ] **Phase 4: AWS Foundation (Terraform)**
  - Infrastructure-as-code for S3 artifacts/checkpoints, ECR repositories, SageMaker execution IAM role, and billing alerts.
- [ ] **Phase 5: SageMaker Training & Evaluation**
  - Managed Spot training jobs, MLflow experiment tracking, and cost auditing.
- [ ] **Phase 6: Evaluation, Robustness & Error Analysis**
  - 4-way comparative evaluation, human rating audit, BANKING77 out-of-distribution stress test, error taxonomy, and model card.
- [ ] **Phase 7: MLflow Tracking & Model Registry**
  - Full model lineage, quality gating script (`promote.py`), and deployment manifest inspector.
- [ ] **Phase 8: Serving Code & Container**
  - FastAPI service with backend abstractions, Pydantic schema validation, fallback recovery, Docker container, and GGUF demo.
- [ ] **Phase 9: SageMaker Endpoint Deployment & Monitoring**
  - Terraform endpoint deployment, load testing with latency histograms, CloudWatch metrics/alarms, and documented teardown.
- [ ] **Phase 10: CI/CD**
  - GitHub Actions CI (PR validation) and CD (AWS OIDC workflow_dispatch training trigger).
- [ ] **Phase 11: Feedback Loop & Retraining (v2)**
  - Failure logging, human labeling workflow, DVC v2 dataset, retraining, gate evaluation, and rollback verification.
- [ ] **Phase 12: Portfolio Packaging**
  - Comprehensive documentation, interview Q&A guide, reproducible benchmarks, and clean release.

---

## 🛠️ Quickstart

### Prerequisites
- Python 3.11 or 3.12
- [`uv`](https://github.com/astral-sh/uv) package manager
- `make`

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd supportiq

# Set up environment and install dependencies
make setup
```

### Verification
```bash
# Run linting and formatting checks
make lint

# Run unit tests
make test
```

---

## 🏛️ Repository Layout
```
supportiq/
├── configs/            # data.yaml, training.yaml, evaluation.yaml
├── data/               # raw/, interim/, processed/, splits/ (DVC-tracked)
├── docs/               # Architecture Decision Records (ADRs), reports, model cards
├── infrastructure/     # Terraform definitions (S3, ECR, IAM, SageMaker)
├── docker/             # Dockerfiles for API service and training
├── notebooks/          # Exploratory notebooks & Kaggle runbooks
├── reports/            # Data profiling, split audit, and evaluation reports
├── schemas/            # Pydantic data & API schemas
├── scripts/            # CLI launcher scripts (training, eval, promotion, deploy)
├── src/supportiq/      # Core package (data, training, eval, inference, monitoring)
├── app/                # FastAPI serving application
├── tests/              # Pytest test suite
└── Makefile            # Standard development targets
```
