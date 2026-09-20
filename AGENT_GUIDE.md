# SupportIQ — Agent Execution Guide

> **Read this whole file before doing anything.**
> This is your operating manual for building SupportIQ, a production-grade LLM fine-tuning
> project for a portfolio. The full design lives in `docs/SupportIQ_End_to_End_Project_Plan.md`
> (the "Plan"). This guide tells you **what to do, in what order, and when to stop and ask
> the human.**

---

## 0. How to work

1. Work **one phase at a time**, in order. Do not start Phase N+1 until every acceptance
   criterion of Phase N passes.
2. Inside a phase, work **one task at a time**. After each task: run tests, run the linter,
   commit with a clear message (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
3. At the end of each phase, write `docs/phase_reports/phase_<N>.md` containing: what was
   built, commands to reproduce, real computed results, problems found, decisions made,
   AWS spend so far (if any).
4. Tasks are tagged:
   - **[AGENT]** you do it.
   - **[HUMAN]** the human must do it (credentials, AWS console, manual Kaggle/Colab GPU
     runs, paid actions). Prepare everything so the human's part is copy-paste, then **stop
     and wait**.
   - **[GATE]** stop and ask for explicit approval before continuing.
5. If blocked, write the blocker to `docs/BLOCKERS.md`, then continue with any unblocked
   task. Never silently skip or fake a step.
6. Library APIs (TRL, PEFT, transformers, bitsandbytes, SageMaker SDK, vLLM) change often.
   **Check current official docs before writing code that depends on them**, and pin the
   versions you verified in `pyproject.toml` / `requirements-*.txt`.

---

## 1. Project context (short)

- **Goal:** Given a customer message, output strict JSON `{category, intent, response}`.
  Compare 4 approaches and prove (or disprove) that fine-tuning is worth it:
  - Baseline A: TF-IDF + Logistic Regression
  - Baseline B: Qwen3-4B-Base, zero/few-shot
  - Baseline C: Qwen3-4B-Base + structured prompt
  - Candidate: Qwen3-4B-Base + SFT + QLoRA
- **Data:** Bitext customer-support dataset (train/val/test). BANKING77 is **external
  evaluation only**, never mixed into training.
- **Deliverable:** a portfolio-grade repo: reproducible data pipeline, tracked experiments,
  model registry, Dockerized FastAPI service, AWS deployment via Terraform, monitoring,
  feedback loop, honest documentation.

### Environment constraints (important)

- The developer machine has **no GPU**. Never try to train or run the 4B model locally.
  Everything except training/LLM inference must run on CPU.
- GPU sources:
  1. **Kaggle/Colab free GPU** (usually a T4 16 GB) for debugging and cheap iteration.
     Assume **no bf16, no FlashAttention-2**; use fp16 + SDPA. The human runs these.
  2. **AWS SageMaker** for official runs, using about **$200 of credits**. Assume
     `ml.g5.xlarge` (A10G, 24 GB) for training.
- Small dev models for debugging: `Qwen3-0.6B-Base` / `Qwen3-1.7B-Base`. Final model:
  `Qwen3-4B-Base`. Switching must be a **config change only**.

### Budget and safety rules (non-negotiable)

- **Never** launch a GPU training job, create an endpoint, or run `terraform apply` on
  billable resources without a **[GATE]** approval that includes a cost estimate.
- Every launcher script must have a `--dry-run` mode that prints instance type, max
  runtime, and estimated cost.
- Every SageMaker job must set `max_run`, and use Managed Spot with checkpoints to S3
  where supported.
- Inference endpoints exist only during demo/testing windows. Terraform must expose
  `enable_endpoint` (default `false`) and a documented teardown command.
- Avoid credit-eating services: NAT gateways, idle notebooks/Studio apps, always-on GPU
  endpoints, large unused EBS/S3 data.
- Tag all AWS resources: `project=supportiq`, `env=dev`.
- **Never commit secrets** (AWS keys, HF tokens, Kaggle keys). Use env vars / `.env`
  (git-ignored) and document required variables in `.env.example`.

### Scientific rules

- No training before the data is profiled and understood.
- No fine-tuning claims without baselines.
- **Never tune on the test set.** Test is touched only for final reporting.
- **No fake metrics.** Anything not yet computed is written as `TBD`. All reported numbers
  come from scripts that write result files.
- Fixed random seeds everywhere; record them.
- Bitext is synthetic/templated. Never describe it as real production traffic.

---

## 2. Conventions

**Tooling:** Python 3.11+, `uv` (or Poetry), `ruff` (lint+format), `pytest`, `pre-commit`,
Polars/PyArrow, Pydantic v2, DVC, MLflow, Docker, Terraform, GitHub Actions.

**Repo layout** (create only what is needed per phase; grow toward this):

```
supportiq/
├── configs/            # data.yaml training.yaml evaluation.yaml deployment.yaml
├── data/               # raw/ interim/ processed/ splits/  (DVC-tracked, not in git)
├── docs/               # plan, phase_reports/, DATA_CARD.md, MODEL_CARD.md, COST.md, DECISIONS.md
├── infrastructure/terraform/
├── docker/
├── notebooks/          # exploratory + kaggle/ thin wrappers around src/
├── reports/            # generated eval/profiling reports (versioned)
├── schemas/            # Pydantic schemas
├── scripts/            # CLI launchers (sagemaker, deploy, teardown, promote)
├── src/supportiq/      # data/ training/ evaluation/ inference/ monitoring/
├── app/                # FastAPI service
├── tests/
├── .github/workflows/
├── dvc.yaml  pyproject.toml  Makefile  README.md  .env.example
```

**Rules:**
- All behavior is **config-driven** (YAML), no hard-coded paths or hyperparameters.
- Notebooks are thin wrappers; **all logic lives in `src/`** and is unit-tested.
- Provide `make` targets: `setup`, `lint`, `test`, `data`, `baseline`, `train-smoke`,
  `eval`, `docker-build`, `serve`, etc.
- Structured JSON logging via a shared logger.
- Record decisions worth explaining to a recruiter in `docs/DECISIONS.md` as short ADRs
  (context → decision → alternatives → consequences).

---

## PHASE 0 — Foundation (laptop)

**Goal:** a clean, testable, CI-enabled repo skeleton.

Tasks
- [AGENT] Init git repo, `pyproject.toml`, dependency groups (`dev`, `data`, `train`,
  `serve`), `.gitignore` (data, secrets, checkpoints), `.env.example`, `Makefile`.
- [AGENT] Config loader (YAML → Pydantic settings), shared logger, seed utility.
- [AGENT] `pre-commit` (ruff, trailing whitespace, secrets scan such as gitleaks/detect-secrets).
- [AGENT] `pytest` setup with a couple of real tests.
- [AGENT] GitHub Actions `ci.yml`: lint + unit tests on PR/push.
- [AGENT] `README.md` stub with project summary and a "status" checklist.
- [AGENT] Copy the Plan into `docs/`.

Acceptance
- `make setup && make lint && make test` pass on a fresh clone.
- CI workflow file present and valid.

---

## PHASE 1 — Data pipeline (laptop, CPU only)

**Goal:** a reproducible, tested, versioned, leakage-safe dataset (Plan §8–15, 18).
Each stage is an independent module with a CLI entry point and unit tests.

### 1.1 Ingestion (`src/supportiq/data/ingest.py`)
- Download Bitext from Hugging Face (record dataset **revision**), save raw files
  **untouched** in `data/raw/`.
- Write `data/raw/METADATA.json`: source URL, revision, license, download date, row count,
  SHA-256 checksum.
- Initialize DVC and track `data/`. (DVC remote: ask the human; a Google Drive folder or
  local path is fine for now.)

### 1.2 Profiling (`profile.py`)
- Generate `reports/data_profile.md` + `.json` with **computed** values: rows, missing
  values, category/intent distributions, instruction/response length stats (median, P95,
  max), duplicate counts (exact instruction; instruction+response), outliers, invalid
  labels, placeholder patterns (e.g. `{{Order Number}}`), token-length estimates using the
  Qwen tokenizer (to choose max sequence length later).
- Also **manually inspect** ~50–100 random rows and record qualitative notes (templating,
  paraphrase style, weird responses) in the report.

### 1.3 Schema validation (`schemas/dataset.py`, `validate.py`)
- Pydantic record schema. Invalid/malformed rows go to `data/interim/quarantine.jsonl`
  with a reason; never silently dropped. Report counts.

### 1.4 PII handling (`pii.py`)
- Use Microsoft Presidio (with a small spaCy model). Detect emails, phones, card numbers,
  names, addresses, etc.
- **Treat dataset placeholders (`{{...}}`) as intentional**, not PII. Do not over-redact
  them.
- Output a PII findings report (entity counts, examples masked) and anonymized text using
  tokens like `<EMAIL>`. Unit-test with synthetic PII strings, including false-positive
  checks on placeholders.
- Note: the data is synthetic, so findings may be near zero. Report that honestly.

### 1.5 Normalization (`normalize.py`)
- Unicode normalization, whitespace cleanup, encoding fixes. Preserve casing/punctuation
  unless justified (document why). Keep original text alongside normalized text.

### 1.6 Deduplication (`deduplicate.py`)
- Exact duplicates (instruction; instruction+response).
- **Near-duplicate / template clustering**: e.g. MinHash-LSH or TF-IDF cosine on
  normalized instructions; assign each row a `cluster_id`. Report cluster-size
  distribution and threshold used.
- Do **not** blindly delete near-duplicates. Keep them but ensure the split keeps each
  cluster in one partition (see 1.8). Log the decision in `DECISIONS.md`.

### 1.7 Taxonomy (`taxonomy.py`, `configs/taxonomy.yaml`)
- Derive category → intent taxonomy **from the data**, write it to YAML, validate every
  row against it. Include a mapping layer (source label → internal label) for future
  datasets.

### 1.8 Leakage-safe split (`split.py`)
- 80/10/10 train/val/test, **grouped by `cluster_id`**, stratified by intent as far as
  groups allow, seeded.
- **Leakage audit** (`reports/split_audit.md`): no exact-duplicate overlap across splits;
  max similarity of any test item to train reported; label distributions per split.
- Freeze the test set: write its checksum to `data/splits/TEST_SET.sha256`. Never modify
  it afterwards.
- Also create a **fixed stratified evaluation subset** (about 500 test rows, seeded) for
  cheap LLM evaluation, saved as its own file.

### 1.9 SFT formatting (`format_sft.py`)
- Output conversational JSONL: user message = customer text; assistant message = compact
  JSON string `{"category","intent","response"}`.
- Verify the Qwen3-4B-Base tokenizer chat template. If absent or unsuitable, define a
  documented plain prompt format with an explicit EOS token. Confirm loss will be computed
  on assistant tokens only.
- Check token lengths against the chosen `max_seq_length`; report truncation rate.

### 1.10 Versioning and documentation
- `dvc.yaml` pipeline connecting stages (`dvc repro` rebuilds everything).
- Tag dataset versions (`data-v1.0`) in git; DVC tracks the artifacts.
- Write `docs/DATA_CARD.md`: source, license, synthetic nature, limitations, pipeline,
  splits, known issues.
- Data tests in `tests/data/` (schema, no split overlap, label validity, determinism).

Acceptance
- `dvc repro` rebuilds the dataset from raw with identical checksums.
- Profile, PII, dedup, and split-audit reports exist and contain computed numbers.
- Test set frozen and hashed. Data tests pass in CI.

**[GATE]** Present the profiling + split audit summary to the human before Phase 2.
The findings about templating and near-duplicates decide the rest of the evaluation.

---

## PHASE 2 — Classical baseline (laptop, CPU)

**Goal:** Baseline A and the shared evaluation code (Plan §20–22).

Tasks
- [AGENT] `src/supportiq/evaluation/classification.py`: accuracy, macro/weighted F1,
  per-intent F1, confusion matrix, bootstrap confidence intervals. Reusable for every model.
- [AGENT] `evaluation/schema.py`: JSON validity, schema validity, field-level accuracy
  (used later for LLM outputs).
- [AGENT] TF-IDF (word + char n-grams) + Logistic Regression for **intent** and
  **category**. Tune **on validation only**. Save the model and predictions.
- [AGENT] **Leakage demonstration:** also train/evaluate the same baseline on a *random*
  (non-grouped) split and report both. This quantifies how much naive splitting inflates
  results, a strong portfolio point.
- [AGENT] Write `reports/baseline_a_report.md` and a first entry in the results table
  (`reports/results.md`).

Acceptance
- Baseline A metrics on validation and test are reproducible via `make baseline`.
- Random-split vs grouped-split comparison is documented.

---

## PHASE 3 — LLM code + free-GPU development (Kaggle/Colab)

**Goal:** working, resumable QLoRA code and LLM baselines, proven on a small model
before spending AWS credits (Plan §16–20, 23).

Tasks
- [AGENT] `training/config.py` + `configs/training.yaml`: model name + **revision**,
  dataset version, LoRA r/alpha/dropout/target modules, learning rate, epochs, batch size,
  grad accumulation, max_seq_length, quantization config, seed, output/checkpoint paths.
  Suggested starting points (tune on **validation only**): r=16, alpha=32, dropout=0.05,
  all linear projection modules, lr about 2e-4, 1–3 epochs, 4-bit NF4, gradient
  checkpointing.
- [AGENT] `training/train.py` (TRL SFTTrainer + PEFT + bitsandbytes):
  - config-driven, CLI entry point
  - checkpoints every N steps, **resume from checkpoint**
  - optional push of checkpoints to HF Hub (private) or a mounted path
  - logs params/metrics to MLflow (file backend by default)
  - works with fp16 on T4 and is switchable to bf16 for A10G via config
- [AGENT] `inference/predictor.py` (batch generation) + `evaluation/generation.py`:
  generate predictions JSONL on a given split, parse JSON, compute schema-validity and
  classification metrics. Response quality: ROUGE-L and BERTScore (CPU-friendly).
- [AGENT] LLM baselines B and C: prompts in `configs/prompts/` (zero-shot, few-shot using
  **train-set examples only**, structured-output prompt). Same fixed 500-row subset for
  every model.
- [AGENT] Prepare Kaggle/Colab notebooks in `notebooks/kaggle/` that only: install pinned
  deps, clone the repo, pull the dataset version, run the CLI. Include a
  `docs/KAGGLE_RUNBOOK.md` with exact steps and expected outputs.
- [HUMAN] Run notebooks on Kaggle/Colab:
  1. Smoke test: Qwen3-0.6B-Base, about 100 steps, verify loss decreases, checkpoint and
     resume work.
  2. Baselines B and C on Qwen3-4B-Base (if it fits) or on the small model first.
  3. Copy result files back into `reports/` (or push to HF Hub) and tell the agent.
- [AGENT] Ingest results and update `reports/results.md`.

Acceptance
- Smoke test trains, checkpoints, resumes, and evaluates end to end on the small model.
- Baseline B/C predictions and metrics exist for the fixed subset.
- Unit tests cover config parsing, prompt building, JSON parsing/validation and metrics.

---

## PHASE 4 — AWS foundation (Terraform)

**Goal:** reproducible cloud base with cost guardrails (Plan §27–28, 34).

Tasks
- [HUMAN] Verify the AWS account plan (Free vs Paid) and credit balance/expiry in
  Billing. Request Service Quota for **ml.g5.xlarge for training job usage** (and for
  endpoint usage if desired). Choose a region with good g5 availability and stick to it.
  Configure credentials locally (SSO or a least-privilege IAM user). Set AWS Budgets
  alerts at $50 / $100 / $150.
- [AGENT] `infrastructure/terraform/`: `main.tf`, `variables.tf`, `outputs.tf`, `s3.tf`,
  `ecr.tf`, `iam.tf` (least-privilege SageMaker execution role), optional `budget.tf`.
  - S3 bucket(s) with versioning, encryption, public access blocked, lifecycle rules
    (expire checkpoints/old artifacts).
  - ECR repos for training and inference images with lifecycle policy.
  - All resources tagged. Variables for region, project name, `enable_endpoint`.
- [AGENT] Run `terraform fmt`, `validate`, `plan`; save the plan summary to
  `docs/phase_reports/phase_4.md`.
- **[GATE]** Human approves `terraform apply` (these resources should cost about pennies).
- [AGENT] Script to sync DVC/data artifacts and configs to S3
  (`scripts/sync_to_s3.py`).

Acceptance
- S3 + ECR + IAM exist and are managed by Terraform; no manual console resources.
- Budget alarms confirmed by the human.

---

## PHASE 5 — SageMaker training and evaluation jobs

**Goal:** official baseline and fine-tuning runs on SageMaker (Plan §17, 29).

Tasks
- [AGENT] Decide the container approach (SageMaker Hugging Face DLC vs custom ECR image)
  after reading current docs. Record an ADR. Prefer the simplest thing that works.
- [AGENT] `scripts/launch_sagemaker_training.py`:
  - `--dry-run` prints instance type, `max_run`, `max_wait`, estimated cost
  - Managed Spot + `checkpoint_s3_uri`; resumes after interruption
  - reads dataset from S3, writes adapter, logs, and metrics back to S3
  - passes the config file, git commit SHA, and dataset version as job metadata
  - the same `train.py` as Phase 3, unchanged
- [AGENT] Equivalent launcher for evaluation jobs (baselines B/C and the fine-tuned model
  on the fixed subset and the full test set).
- **[GATE]** Present cost estimates. After approval, run in this order:
  1. **Tiny validation job** (about 10–15 min): confirms permissions, quotas, paths.
  2. Baseline B/C evaluation on Qwen3-4B-Base (official numbers).
  3. **Main QLoRA run** on Qwen3-4B-Base.
  4. Optional ablation (LoRA rank, or training-set size learning curve) if budget allows.
- [AGENT] Pull artifacts to `artifacts/`, log runs to MLflow, and update `COST.md` with
  actual job durations and spend per job (from job metadata/billing data provided by the
  human).

Acceptance
- Adapter for the main run stored in S3 with config + dataset version + git SHA recorded.
- Training is resumable after Spot interruption (verified or documented).
- `COST.md` contains real numbers.

---

## PHASE 6 — Evaluation, robustness and error analysis

**Goal:** honest evidence of whether fine-tuning helped (Plan §21–22, 6).

Tasks
- [AGENT] Evaluate all four systems on the same test data: accuracy, macro/weighted F1,
  per-intent F1, confusion matrices, JSON/schema validity, field accuracy, latency,
  ROUGE-L/BERTScore for responses, with bootstrap confidence intervals.
- [AGENT] **Manual review:** sample about 50 responses per system (blinded and shuffled)
  into a CSV for the human to rate for relevance, correctness, helpfulness. Keep
  subjective results separate from objective metrics.
- [AGENT] **BANKING77 robustness protocol:**
  - Write `configs/banking77_mapping.yaml` proposing a coarse mapping from BANKING77
    intents to Bitext categories/intents **only where genuinely equivalent**; mark
    unmappable intents as out-of-taxonomy.
  - **[GATE]** Human reviews and approves the mapping.
  - Report: accuracy on mapped subset, behavior on out-of-taxonomy inputs (JSON validity,
    hallucinated labels, fallback rate), qualitative failures.
  - Never train on BANKING77 in v1.
- [AGENT] **Error analysis:** classify failures using the Plan §22 categories; produce
  top failure modes with examples.
- [AGENT] Generate `reports/evaluation_report.md` with the comparison table, charts
  (matplotlib), the key finding stated plainly, and caveats (synthetic data, subset
  sizes). If TF-IDF matches the LLM on classification, **say so** and highlight where the
  LLM does add value (structured output, response generation, robustness).
- [AGENT] Draft `docs/MODEL_CARD.md` (intended use, training data, metrics, limitations,
  out-of-scope use).

Acceptance
- Every number in the report is traceable to a results file.
- Error-analysis section names concrete failure modes with examples.

---

## PHASE 7 — MLflow tracking and model registry

**Goal:** lineage: "exactly what produced the deployed model?" (Plan §23–24).

Tasks
- [AGENT] Ensure each run logs: model name and revision, dataset version, git commit,
  full hyperparameters (LR, batch, grad accumulation, epochs, seq length, LoRA r/alpha/
  dropout, quantization), train/val loss, intent/category F1, JSON/schema validity,
  latency, and links to S3 artifacts.
- [AGENT] Tracking backend: local file store by default; if the human provides DagsHub
  (or similar) credentials, support it via env vars. Do not require a paid server.
- [AGENT] Registry: MLflow Model Registry (or, if impractical for adapter-only artifacts,
  a versioned `registry/models.json` manifest plus MLflow tags). Each version references
  base model + revision, adapter S3 URI, dataset version, config, eval report, git SHA,
  MLflow run ID.
- [AGENT] `scripts/promote.py`: reads quality gates from `configs/evaluation.yaml`
  (e.g. minimum intent macro-F1, minimum schema validity, max latency, must beat the
  currently promoted model). Promotion fails loudly if gates are not met.
- [AGENT] `scripts/whats_deployed.py`: prints full lineage of the promoted model.

Acceptance
- Running `whats_deployed.py` answers the lineage question with real references.
- Promotion is blocked when gates fail (tested).

---

## PHASE 8 — Serving code and container

**Goal:** a tested inference service with structured-output guarantees (Plan §19, 25–26).

Tasks
- [AGENT] Decide the serving stack via an ADR after reading current docs: options include
  SageMaker LMI/vLLM container with LoRA adapter support, TGI, or merging the adapter and
  serving the merged weights. Choose the simplest option that works on the target
  instance.
- [AGENT] `app/` FastAPI service:
  - `POST /v1/support/analyze`, `GET /health`, `GET /ready`, `GET /model`, `GET /metrics`
  - Pydantic request/response schemas; input length and content validation
  - **Backend abstraction:** `MockBackend` (for tests), `SageMakerBackend` (boto3),
    `LocalGGUFBackend` (llama.cpp, for the free CPU demo)
  - Output validation against the schema. On invalid output: one bounded retry, then a
    safe fallback response and an incremented failure counter. Invalid output must never
    silently reach the caller.
  - Structured JSON request logs (request ID, model version, latency, validity flag, no raw
    PII beyond what is needed) written to a JSONL file
  - Prometheus-format metrics: request count, errors, latency histogram, schema-failure
    rate, fallback rate
- [AGENT] Tests: API tests with `MockBackend`, schema tests, failure/fallback tests.
- [AGENT] `docker/Dockerfile.api` (multi-stage, non-root, pinned base). Build and run the
  image locally with the mock backend and confirm health checks pass.
- [AGENT] Merge adapter and produce a 4-bit GGUF for a free always-on CPU demo
  (Hugging Face Space). Where a step needs a GPU/Kaggle or SageMaker run, prepare the
  script and mark it [HUMAN]. Measure and report CPU latency honestly.
- [HUMAN] Publish the Hugging Face Space and provide the link.

Acceptance
- `make serve` runs the API locally; `docker build` succeeds; tests pass in CI.
- Space demo reachable, with real (not mocked) predictions.

---

## PHASE 9 — SageMaker endpoint deployment and monitoring

**Goal:** a real cloud deployment, demonstrated then torn down (Plan §27–31).

Tasks
- [AGENT] Terraform module for SageMaker model, endpoint config, and endpoint, gated by
  `enable_endpoint`. Include CloudWatch log groups, dashboard, and alarms (errors, p95
  latency, endpoint health).
- [AGENT] `scripts/deploy_endpoint.py` and `scripts/teardown_endpoint.py`, with the
  estimated hourly cost printed before deploying.
- [AGENT] `scripts/load_test.py` (Locust or a simple async client): measure p50/p95/p99
  latency, throughput, and error rate at a few concurrency levels.
- **[GATE]** Human approves a **time-boxed deployment window** (state the hours and
  estimated cost).
- [HUMAN] With the agent's scripts: deploy, run the load test, capture CloudWatch
  screenshots, record a short demo video, then **run teardown immediately** and confirm no
  endpoint remains (`aws sagemaker list-endpoints`).
- [AGENT] Write `COST.md`: actual endpoint cost per hour, cost per 1,000 requests at
  measured throughput, cost of training runs, and a "what production would cost" estimate
  (24/7, autoscaling options, cheaper CPU/GGUF alternative).

Acceptance
- Screenshots, latency table, and demo recording are in `docs/assets/`.
- Endpoint torn down; Terraform state shows no leftover billable resources.

---

## PHASE 10 — CI/CD

**Goal:** automated quality checks and controlled model workflows (Plan §33).

Tasks
- [AGENT] `ci.yml` (PR): lint, unit tests, data tests, API tests, Docker build (no push).
- [AGENT] `train.yml` (**`workflow_dispatch` only**): launches a SageMaker training job
  using AWS OIDC role assumption (no long-lived keys in GitHub secrets), then evaluation,
  then the quality gate from Phase 7. Stops on failure. **No automatic deployment.**
- [AGENT] `docs/CICD.md` with a diagram of the flow. Add CI status badges to the README.
- [HUMAN] Create the OIDC role/trust policy using the Terraform/IAM snippets the agent
  provides, and add required repository variables.

Acceptance
- PR workflow green. Training workflow validated in dry-run mode without spending money.

---

## PHASE 11 — Feedback loop and retraining (v2)

**Goal:** demonstrate the full model lifecycle (Plan §32).

Tasks
- [AGENT] Prediction logging schema and `scripts/export_failures.py` (invalid JSON,
  low-confidence, user-flagged, and errors found in validation/BANKING77 runs).
- [AGENT] Human-review workflow: export a review CSV (input, prediction, suggested fix,
  blank `correct_category`, `correct_intent`, `correct_response`, `reviewer_notes`).
- [HUMAN] Review and label a sample (even 100–200 rows is fine).
- [AGENT] Validate reviewed rows (schema, taxonomy), then build **dataset v2**:
  - **Never add any test-set item, or near-duplicate of one, to training.** Check
    programmatically against the frozen test set and log the check.
  - Version with DVC, tag `data-v2.0`, update the DATA_CARD.
- **[GATE]** Approve the v2 retraining cost. Retrain via the SageMaker launcher, then
  evaluate on the **same untouched test set** and compare v1 vs v2.
- [AGENT] Run promotion through the quality gate. Document a **rollback** procedure and
  script: switch the registry alias to the previous version and redeploy the previous
  adapter. Test rollback in dry-run.
- [AGENT] Write `reports/v1_vs_v2.md`. If v2 does not improve, report that honestly.

Acceptance
- Full loop documented and demonstrated: failures → review → dataset v2 → retrain →
  evaluate → gate → promote/rollback.

---

## PHASE 12 — Portfolio packaging

**Goal:** a repo a recruiter can understand in five minutes and verify in thirty.

Tasks
- [AGENT] **README.md**, in this order: one-paragraph pitch, headline result (table),
  architecture diagram (Mermaid), live demo link and demo video, key engineering
  decisions, how to reproduce (`make` commands), repo tour, cost summary, limitations and
  honest notes, "what I would do next".
- [AGENT] Finalize `DATA_CARD.md`, `MODEL_CARD.md`, `COST.md`, `DECISIONS.md`,
  `docs/CICD.md`, `docs/KAGGLE_RUNBOOK.md`.
- [AGENT] Add answers (with links to evidence in the repo) for the Plan §38 interview
  questions in `docs/INTERVIEW_QA.md`: why fine-tune, data prep, leakage prevention, why
  QLoRA, why this model, proof fine-tuning helped, how to know what's in production,
  retraining, failure handling, inference cost, rollback.
- [AGENT] Repo hygiene pass: remove dead code and stray notebooks outputs, check no
  secrets in history (run a secrets scanner), verify `LICENSE`, pin versions, confirm
  `make setup && make test` works from a clean clone.
- [HUMAN] Tag the release (`v1.0.0`), pin the repo on GitHub, add a short LinkedIn/CV
  bullet list (the agent can draft it from the real results).

Acceptance
- A stranger can clone, read the README, and reproduce data + baseline + tests without
  asking anything.
- Every performance claim in the README links to a report file.

---

## Appendix A — Definition of "done" for any task

- Code is typed where reasonable, linted, and has tests.
- Behavior is configurable, not hard-coded.
- Results are produced by scripts and saved to files, never typed by hand.
- Documentation updated; commit made.
- Cost impact (if any) recorded.

## Appendix B — Things the agent must never do

- Train or run the 4B model locally on CPU.
- Launch billable AWS resources without a [GATE] approval.
- Modify or peek at the test set for tuning; add test items to training.
- Merge BANKING77 into training in v1.
- Invent metrics, costs, or results.
- Commit secrets, raw large data, or model weights to git.
- Leave GPU endpoints running.
