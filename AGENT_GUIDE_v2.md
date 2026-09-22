# SupportIQ — Agent Execution Guide (v2: staged, notebook-first)

> Supersedes `SUPPORTIQ_AGENT_GUIDE.md`. Read this whole file before starting.
> Design reference: `docs/SupportIQ_End_to_End_Project_Plan.md` (the "Plan").

**What we are building:** a customer-support model that reads a message and returns strict
JSON `{category, intent, response}`. We compare TF-IDF + Logistic Regression, Qwen3-4B-Base
prompted, and Qwen3-4B-Base fine-tuned with QLoRA, then package the winner as a Dockerized
FastAPI service deployed on AWS, with tracking, monitoring, and a feedback loop.

**How this guide differs from a "do everything" spec:** the project is built as a series of
**small vertical stages**. In each stage the human learns by *seeing* the thing work in a
notebook, then it is hardened into code, then wrapped in the next tool. Tools like YAML
configs, pytest, FastAPI, Docker, CI, MLflow, and Terraform are introduced **at the stage
that first needs them**, not up front.

---

## 0. Working agreement

### 0.1 The stage loop (applies to every stage)

For each subtask, follow this order:

1. **[NB] Explore in a notebook.** Small cells, each with a markdown cell above it saying
   *what this does and why*. Show outputs (shapes, samples, plots). No hidden logic.
2. **[PY] Extract to `src/supportiq/`.** Move the working notebook logic into clean,
   typed functions. The notebook then *imports* them, so the notebook still runs.
3. **[TEST] Add tests** (`tests/`) for the extracted logic. Start with a few meaningful
   tests, not exhaustive ones.
4. **[DOC] Update docs** briefly (see 0.3).
5. **Commit** with a clear message (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).

Every stage ends with:
- a **stage report** `docs/stage_reports/stage_<N>.md` (what was built, commands to
  reproduce, computed results, issues, decisions, cloud spend so far), and
- a **short "check your understanding" list** at the end of the stage report: 3–5
  questions the human should be able to answer before moving on (e.g. "why does a random
  split inflate the score on this dataset?"). The human answers them in
  `docs/LEARNING_LOG.md` in their own words. The agent must not write those answers.

### 0.2 Task tags

- **[NB]** notebook work · **[PY]** module code · **[TEST]** tests · **[DOC]** docs
- **[HUMAN]** the human must do it (credentials, AWS console, running Kaggle/Colab GPU
  notebooks, labeling). Prepare it so it's copy-paste, then **stop and wait**.
- **[GATE]** stop and get explicit approval before continuing.

Do not start Stage N+1 until every acceptance item of Stage N passes **and** the human has
confirmed the stage. If blocked, write to `docs/BLOCKERS.md` and continue with any
unblocked work; never silently skip or fake a step.

### 0.3 Introducing a new tool

The first time a tool appears (pytest, YAML configs, pre-commit, DVC, FastAPI, Docker,
GitHub Actions, MLflow, Terraform, SageMaker), create `docs/concepts/<tool>.md` of **at
most ~25 lines** covering: what problem it solves here, the 3–5 core ideas, the minimal
example from *this* repo, and how to debug it when it breaks. Keep the tone technical and
concise, not condescending. Then use the tool.

### 0.4 Environment constraints

- The developer machine has **no GPU**. Never train or run the 4B model locally.
  Everything except training and LLM inference must run on CPU.
- GPU sources:
  1. **Kaggle/Colab free GPU** (assume a T4 16 GB: fp16, no bf16, no FlashAttention-2) for
     development and smoke tests. The human runs these.
  2. **AWS SageMaker** with about **$200 of credits** for official runs (assume
     `ml.g5.xlarge`, A10G 24 GB).
- Dev models: `Qwen3-0.6B-Base` / `Qwen3-1.7B-Base`. Final: `Qwen3-4B-Base`. Switching
  models must be a config change only.

### 0.5 Budget and safety rules (non-negotiable)

- **No billable action** (SageMaker job, endpoint, `terraform apply` of billable
  resources) without a **[GATE]** that includes a cost estimate.
- Launcher scripts must have `--dry-run` (prints instance type, max runtime, estimated
  cost). Every SageMaker job sets `max_run`; use Managed Spot + S3 checkpoints where
  supported.
- Endpoints exist only during time-boxed demo windows. Terraform exposes
  `enable_endpoint` (default `false`) and there is a documented teardown command.
- Avoid credit-eaters: NAT gateways, idle notebooks/Studio apps, always-on GPU endpoints,
  unused large S3/EBS data. Tag every resource `project=supportiq`.
- **Never commit secrets.** Use env vars / a git-ignored `.env`; keep `.env.example`.

### 0.6 Scientific rules

- No training before the data is profiled and understood.
- No fine-tuning claims without baselines. **Never tune on the test set.**
- **No fake metrics:** anything not computed is `TBD`. Numbers come from scripts that write
  result files.
- Fixed seeds; record them. Bitext is synthetic/templated, never call it real traffic.
- BANKING77 is external evaluation only, never training data in v1.

### 0.7 Repo growth (do not create everything on day 1)

Start with:
```
supportiq/
├── notebooks/
├── src/supportiq/
├── data/            # git-ignored
├── docs/
└── README.md
```
Add folders only when a stage needs them: `tests/` (Stage 2), `configs/` (Stage 2),
`app/` + `docker/` + `.github/workflows/` (Stage 4), `scripts/` (Stage 6),
`infrastructure/terraform/` (Stage 9).

---

## STAGE 0 — Project setup

**Goal:** a clean repo and a working environment; nothing ML yet.

Subtasks
- 0.1 [AGENT] Create the repo, `.gitignore` (data, `.env`, checkpoints, `.ipynb_checkpoints`),
  Python environment with `uv` (or venv), `pyproject.toml` with dependency groups, and
  an installable `src/supportiq` package (`pip install -e .`).
- 0.2 [AGENT] Add `README.md` stub, `docs/LEARNING_LOG.md` (empty template),
  `docs/DECISIONS.md` (ADR template), copy the Plan into `docs/`.
- 0.3 [AGENT] `pre-commit` with ruff (lint + format). Write `docs/concepts/pre-commit.md`.
- 0.4 [HUMAN] Create the GitHub repo and push. Create the Hugging Face and Kaggle accounts
  (needed in later stages).
- 0.5 [HUMAN, do now, needs lead time] Check the AWS plan (Free vs Paid) and credit
  balance/expiry under Billing. Request Service Quota for **ml.g5.xlarge for training job
  usage**. Set AWS Budgets alerts at $50 / $100 / $150. Nothing is launched yet.

Acceptance: fresh clone → install → `import supportiq` works; ruff passes; first commit
pushed.

---

## STAGE 1 — Explore the data (notebook only)

**Goal:** understand Bitext well enough to make every later decision from evidence.

Subtasks
- 1.1 [NB] `01_load_and_inspect.ipynb`: load the Hugging Face dataset, record dataset
  **revision**, save raw files untouched to `data/raw/`, write `data/raw/METADATA.json`
  (source, revision, license, date, row count, SHA-256). Look at columns, dtypes, 20
  random rows.
- 1.2 [NB] `02_profile.ipynb`: category/intent counts and balance (bar charts), instruction
  and response length distributions (median, P95, max), missing values, exact
  duplicates, invalid/odd labels, placeholder patterns like `{{Order Number}}`, token
  lengths using the Qwen tokenizer.
- 1.3 [NB] Template analysis: find near-duplicate instructions (TF-IDF cosine or MinHash),
  estimate how many rows are paraphrases of the same seed, show example clusters.
- 1.4 [NB] Read 50–100 random rows by hand and write qualitative notes (tone, oddities,
  label problems) in a markdown cell.
- 1.5 [PY] Extract only the reusable pieces: `data/load.py` (load + save raw + metadata)
  and `data/profile.py` (functions that compute the profile numbers). Write the profile
  to `reports/data_profile.md/.json`.
- 1.6 [DOC] Start `docs/DATA_CARD.md` with source, license, synthetic nature, and what the
  profile revealed.

Acceptance: every number in the profile comes from code; the near-duplicate finding is
quantified; notes on templating written.

**[GATE]** Present the profile summary. The templating finding shapes Stage 2 and 3.

Check-your-understanding topics: what class imbalance is and whether it exists here; why
templated data threatens evaluation; how token length relates to `max_seq_length`.

---

## STAGE 2 — Clean, validate, split (notebook → module → tests)

**Goal:** a reproducible, leakage-safe train/val/test split.

Subtasks
- 2.1 [NB] Schema and validation: define a Pydantic record model, validate all rows, and
  show how invalid rows are quarantined (never silently dropped).
  [PY] `schemas/dataset.py`, `data/validate.py` → `data/interim/quarantine.jsonl`.
  [TEST] valid/invalid record cases. *(Write `docs/concepts/pytest.md`.)*
- 2.2 [NB] PII check with Microsoft Presidio: run on a sample, inspect detections, treat
  `{{...}}` placeholders as intentional (not PII), measure false positives. Findings may
  be near zero since the data is synthetic; report that. [PY] `data/pii.py`.
  [TEST] synthetic PII strings + placeholder false-positive check.
- 2.3 [NB] Normalization (Unicode, whitespace); keep original + normalized text.
  [PY] `data/normalize.py`. [TEST] a few edge cases.
- 2.4 [NB] Deduplication and **cluster ids**: exact duplicates, then near-duplicate
  clusters from Stage 1.3 → a `cluster_id` per row. Do not delete near-duplicates blindly.
  [PY] `data/deduplicate.py`. Record the threshold and reasoning in `DECISIONS.md`.
- 2.5 [NB] Taxonomy: derive category → intent map from the data, save as YAML, validate
  every row. *(First YAML/config in the project → write `docs/concepts/yaml-configs.md`.)*
  [PY] `data/taxonomy.py`; introduce `configs/data.yaml` (paths, seed, split ratios,
  thresholds) and a small loader in `config.py`.
- 2.6 [NB] **Grouped split**: 80/10/10 by `cluster_id`, stratified by intent where
  possible, seeded. Leakage audit: no exact overlap across splits, maximum similarity of
  each test item to train, label distributions per split.
  [PY] `data/split.py` → `data/splits/{train,val,test}.parquet`, plus a **fixed
  stratified ~500-row evaluation subset** of test.
  [TEST] no cluster spans two splits; deterministic given the seed.
  Write `reports/split_audit.md`; freeze the test set (`data/splits/TEST_SET.sha256`).
- 2.7 [NB] SFT formatting: produce the chat/JSONL format where the assistant turn is
  compact JSON `{"category","intent","response"}`. Verify the Qwen3-4B-Base tokenizer
  chat template (if absent, define a documented prompt format with an explicit EOS token),
  confirm loss will apply to assistant tokens only, report the truncation rate at the
  chosen max length. [PY] `data/format_sft.py`. [TEST] round-trip parse of formatted
  examples.
- 2.8 [PY] Introduce DVC: track `data/` and add `dvc.yaml` stages for
  validate → normalize → dedupe → split → format so `dvc repro` rebuilds everything.
  *(Write `docs/concepts/dvc.md`.)* Ask the human for a remote (a Google Drive folder or a
  local path is fine). Tag `data-v1.0`.
- 2.9 [PY] Add `Makefile` targets (`setup`, `lint`, `test`, `data`), finish `DATA_CARD.md`.

Acceptance: `dvc repro` rebuilds the dataset with identical checksums; tests pass; split
audit shows zero cross-split cluster leakage; test set is frozen.

Check-your-understanding topics: why group by cluster; what "leakage" means for your later
numbers; what DVC tracks vs what git tracks.

---

## STAGE 3 — Classical baseline (Baseline A)

**Goal:** the number every later model must justify itself against, and the shared
evaluation code.

Subtasks
- 3.1 [NB] `03_tfidf_baseline.ipynb`: TF-IDF (word + char n-grams) + Logistic Regression
  predicting **intent**, then **category**. Tune on **validation only**. Inspect
  the top features per intent.
- 3.2 [NB] Metrics: accuracy, macro/weighted F1, per-intent F1, confusion matrix.
  Explain in markdown why macro-F1 and not only accuracy. Add bootstrap confidence
  intervals.
  [PY] `evaluation/classification.py` (reused by every later model). [TEST] metrics on a
  toy example with known answers.
- 3.3 [NB] **Leakage experiment:** retrain on a *random* (non-grouped) split and compare
  with the grouped result. Quantify the inflation.
- 3.4 [PY] `models/tfidf.py` (train / save / load / predict with confidence),
  `configs/baseline.yaml`, `make baseline`. Save predictions and metrics to `reports/`.
- 3.5 [DOC] `reports/baseline_a_report.md` and the first row of `reports/results.md`.

Acceptance: `make baseline` reproduces the metrics; the grouped-vs-random comparison is
documented.

Check-your-understanding topics: what the baseline's ceiling implies for the LLM's value;
where the baseline fails (which intents, why).

---

## STAGE 4 — Serve the baseline: FastAPI, Docker, CI

**Goal:** learn API + container + CI on the small model you already understand. The LLM
will later plug into the *same* API.

Subtasks
- 4.1 [NB] `04_predict_function.ipynb`: load the saved baseline and call a `predict(text)`
  function returning `{category, intent, confidence}`. Show edge cases (empty text, very
  long text, non-English).
- 4.2 [PY] `inference/base.py`: a `Predictor` interface with `predict(text) -> Prediction`;
  `inference/tfidf_predictor.py` implementing it. This interface is what lets the LLM be
  swapped in later.
- 4.3 [PY] FastAPI app in `app/`: Pydantic request/response schemas, `POST
  /v1/support/analyze`, `GET /health`, `GET /model`. Input validation (length limits).
  *(Write `docs/concepts/fastapi.md`.)* [NB] A short notebook or `curl` walk-through
  calling the running API.
- 4.4 [PY] Structured JSON logging per request (request id, latency, model version,
  prediction) to a JSONL file, plus a `MockPredictor` for tests.
- 4.5 [TEST] API tests with FastAPI `TestClient`: valid request, invalid request (422),
  health, error path.
- 4.6 [PY] `docker/Dockerfile.api` (multi-stage, non-root, pinned base image) and
  `.dockerignore`. Build and run locally; confirm `/health` works from inside the
  container. *(Write `docs/concepts/docker.md`.)* Add `make serve`, `make docker-build`.
- 4.7 [PY] `.github/workflows/ci.yml`: install → ruff → pytest → docker build (no push).
  *(Write `docs/concepts/github-actions.md`: what a workflow, job, step are; explain each
  block of this file.)* Add a README status badge.
- 4.8 [DOC] Stage report + a short architecture sketch (Mermaid) of "client → FastAPI →
  Predictor".

Acceptance: `make serve` works; container runs; CI is green.

Check-your-understanding topics: request → validation → prediction → response flow;
why a `Predictor` interface; what the Dockerfile layers do; what CI is protecting.
EOF
---

## STAGE 5 — LLM baselines (Colab/Kaggle GPU)

**Goal:** Baselines B and C, plus a robust way to get structured JSON out of an LLM.

Subtasks
- 5.1 [NB] `05_prompting.ipynb` (CPU, tiny model or mocked outputs): design the prompt
  formats: zero-shot, few-shot (examples from **train only**), and structured-output
  prompt that requests the exact JSON schema. Show how parsing succeeds/fails.
- 5.2 [PY] `inference/prompts.py` and `configs/prompts/*.yaml`;
  `inference/parse.py` (extract JSON, validate against the Pydantic schema, classify
  failures: not JSON / wrong keys / invalid label). [TEST] parser on good, malformed, and
  truncated outputs.
- 5.3 [PY] `inference/hf_predictor.py`: batch generation with Hugging Face `transformers`
  (4-bit optional), implementing the same `Predictor` interface from Stage 4.
- 5.4 [PY] `evaluation/generation.py`: run a predictor over a split, save predictions JSONL,
  compute classification metrics (reusing Stage 3 code), JSON/schema validity, field
  accuracy, latency, and ROUGE-L / BERTScore for responses.
- 5.5 [DOC] `docs/KAGGLE_RUNBOOK.md` and a thin notebook `notebooks/kaggle/run_baselines.ipynb`
  that only installs pinned deps, clones the repo, pulls the dataset version, and calls
  the CLI. Pin verified versions (check current docs; Qwen3 needs a recent
  `transformers`).
- 5.6 [HUMAN] Run on Colab/Kaggle: first Qwen3-0.6B-Base to prove the pipeline, then
  Qwen3-4B-Base (Baselines B and C) on the **fixed 500-row subset**. Return the result
  files.
- 5.7 [AGENT] Ingest results into `reports/results.md`; write an analysis of how often the
  base model produces valid JSON and where it goes wrong.

Acceptance: predictions and metrics for B and C exist on the same subset used later for the
fine-tuned model; parsing and prompt code are tested.

Check-your-understanding topics: why a base model needs prompting tricks; what JSON
validity measures that accuracy does not.

---

## STAGE 6 — QLoRA fine-tuning (free GPU first)

**Goal:** understand and run parameter-efficient fine-tuning on the free GPU, resumably.

Subtasks
- 6.1 [NB] `06_lora_concepts.ipynb`: load the small model, count parameters, apply a LoRA
  adapter, show trainable vs frozen parameters, and inspect one tokenized SFT example
  (including which tokens contribute to the loss).
- 6.2 [PY] `configs/training.yaml`: model name + revision, dataset version, LoRA
  (r=16, alpha=32, dropout=0.05, all linear projections as starting points), lr about
  2e-4, epochs 1–3, 4-bit NF4, gradient checkpointing, max_seq_length from Stage 2,
  seed, checkpoint paths. Tune on **validation only**.
- 6.3 [PY] `training/train.py` (TRL `SFTTrainer` + PEFT + bitsandbytes), CLI entry point:
  checkpoints every N steps, **resume from checkpoint**, fp16 on T4 / bf16 configurable
  for A10G, logs metrics to a local JSON file (MLflow comes in Stage 8).
  Always check current TRL/PEFT docs; APIs change.
- 6.4 [TEST] Config parsing and data-collation tests (CPU, tiny fake model where possible).
- 6.5 [DOC] Extend `KAGGLE_RUNBOOK.md`; thin notebook `notebooks/kaggle/train.ipynb`.
- 6.6 [HUMAN] **Smoke test** on Kaggle/Colab: Qwen3-0.6B-Base, ~100 steps. Confirm loss
  decreases, a checkpoint is written, and resuming works after killing the session.
- 6.7 [HUMAN] Optional but useful: a first full run of the small model (0.6B/1.7B) to
  see the whole loop and get preliminary numbers. Check whether Qwen3-4B fits on the
  free T4 with your settings; if it does, a 4B run here is a fallback in case the AWS
  quota is delayed.
- 6.8 [AGENT] Run the Stage 5 evaluation on the fine-tuned adapter's predictions; add a row
  to `reports/results.md`.

Acceptance: training runs, checkpoints, resumes, and produces an adapter that the Stage 5
evaluator can score.

Check-your-understanding topics: what LoRA/QLoRA actually change and why memory drops;
what the loss curve tells you; why validation, not test, guides tuning.

---

## STAGE 7 — Evaluation, robustness, error analysis

**Goal:** honest evidence of whether fine-tuning helped, and where it fails.

Subtasks
- 7.1 [NB] `07_compare_models.ipynb`: all four systems on the same test data — accuracy,
  macro/weighted F1, per-intent F1, JSON/schema validity, latency, ROUGE-L/BERTScore, with
  bootstrap confidence intervals. Plot the comparison.
- 7.2 [NB] Error analysis: categorize failures (Plan §22), find the top failure modes,
  show concrete examples. Include confusion between similar intents.
- 7.3 [PY] Export the notebook logic into `evaluation/report.py` → `reports/evaluation_report.md`
  and charts.
- 7.4 [AGENT + HUMAN] Manual review sheet: ~50 blinded, shuffled responses per system in a
  CSV; the human rates relevance/correctness/helpfulness. Keep subjective results separate.
- 7.5 [NB] **BANKING77 robustness:** propose `configs/banking77_mapping.yaml` (coarse mapping
  only where intents are genuinely equivalent; others marked out-of-taxonomy).
  **[GATE]** human approves the mapping. Report accuracy on the mapped subset, and, for
  out-of-taxonomy inputs, JSON validity, invented labels, and fallback behavior.
- 7.6 [DOC] `docs/MODEL_CARD.md` draft (intended use, data, metrics, limitations,
  out-of-scope use). State the key finding plainly. If TF-IDF matches the LLM on
  classification, say so and show where the LLM adds value.

Acceptance: every number traces to a results file; failure modes are concrete.

---

## STAGE 8 — Experiment tracking and model registry (MLflow)

**Goal:** answer "exactly what produced this model?" with evidence.

Subtasks
- 8.1 [NB] `08_mlflow_intro.ipynb`: log a toy run (params, metrics, artifacts) and open the
  MLflow UI. *(Write `docs/concepts/mlflow.md`.)*
- 8.2 [PY] `tracking/mlflow_utils.py`: standard logging helper (model + revision, dataset
  version, git SHA, all hyperparameters, metrics, artifact links). Local file backend by
  default; support DagsHub via env vars if the human provides it.
- 8.3 [PY] **Backfill** runs for Baseline A, B, C and the fine-tuned model from stored result
  files, so the UI shows the whole story.
- 8.4 [PY] Registry: MLflow Model Registry, or a versioned `registry/models.json` manifest
  if adapter-only artifacts fit poorly. Each version links: base model + revision,
  adapter location, dataset version, config, eval report, git SHA, run ID.
- 8.5 [PY] `scripts/promote.py` with quality gates in `configs/evaluation.yaml` (min intent
  macro-F1, min schema validity, max latency, must beat the current model). Fails loudly.
  `scripts/whats_deployed.py` prints full lineage. [TEST] gate pass/fail cases.

Acceptance: lineage is answerable with one command; promotion is blocked when gates fail.

---

## STAGE 9 — AWS: infrastructure, SageMaker training, LLM serving

**Goal:** the cloud story, done in small verifiable steps with cost guardrails.

Subtasks
- 9.1 [NB] `09_aws_basics.ipynb`: with boto3, list S3 buckets, upload and download a small
  file, describe quotas. Confirms credentials work. *(Write `docs/concepts/aws-basics.md`:
  IAM roles vs users, S3, ECR, SageMaker jobs vs endpoints, what costs money.)*
- 9.2 [PY] `infrastructure/terraform/`: S3 (versioned, encrypted, public access blocked,
  lifecycle rules), ECR (lifecycle policy), least-privilege SageMaker execution role, tags,
  variables (`region`, `enable_endpoint`). Run `fmt`, `validate`, `plan`.
  *(Write `docs/concepts/terraform.md`.)* **[GATE]** human reviews the plan before
  `apply` (should cost about pennies).
- 9.3 [PY] `scripts/sync_to_s3.py` to upload the dataset version and configs.
- 9.4 [PY] Container decision ADR (SageMaker Hugging Face DLC vs custom ECR image;
  prefer the simplest thing that works). `scripts/launch_sagemaker_training.py` with
  `--dry-run`, Managed Spot, `checkpoint_s3_uri`, `max_run`/`max_wait`, running the
  **same** `train.py` from Stage 6.
- 9.5 **[GATE]** cost estimate approved, then in order: (a) 10–15 min validation job;
  (b) official baseline evaluation on Qwen3-4B-Base; (c) **main QLoRA run**;
  (d) optional ablation (LoRA rank or training-set size).
- 9.6 [PY] Pull artifacts, register the model (Stage 8), record real job durations and spend
  in `docs/COST.md`.
- 9.7 [PY] **LLM serving:** ADR for the serving stack (SageMaker LMI/vLLM with LoRA, TGI, or
  merged weights). Implement `SageMakerPredictor` (boto3) and `LocalGGUFPredictor`
  (llama.cpp) behind the **same `Predictor` interface** from Stage 4. Add output
  validation: one bounded retry, then safe fallback + failure counter; invalid output
  never reaches the caller. [TEST] using `MockPredictor`.
- 9.8 [PY] Merge adapter → 4-bit GGUF for a free CPU demo on a Hugging Face Space.
  [HUMAN] publishes the Space and shares the link. Measure real CPU latency.
- 9.9 [PY] Terraform module for SageMaker model/endpoint config/endpoint behind
  `enable_endpoint`, CloudWatch log groups/dashboard/alarms, `scripts/deploy_endpoint.py`
  and `scripts/teardown_endpoint.py` (print hourly cost before deploying), and
  `scripts/load_test.py` (p50/p95/p99 latency, throughput, errors).
- 9.10 **[GATE]** a time-boxed deployment window with estimated cost. [HUMAN]: deploy →
  load test → CloudWatch screenshots → demo recording → **teardown immediately** and
  verify with `aws sagemaker list-endpoints`.
- 9.11 [DOC] Finish `COST.md`: cost per training run, endpoint cost per hour, cost per
  1,000 requests at measured throughput, and a 24/7 production estimate with cheaper
  alternatives.

Acceptance: adapter in S3 with config + dataset version + git SHA recorded; endpoint
demonstrated and torn down; screenshots/latency table/demo in `docs/assets/`; `COST.md`
has real numbers.

---

## STAGE 10 — Monitoring, feedback loop, CI/CD

**Goal:** show the model lifecycle, not just a one-off model.

Subtasks
- 10.1 [PY] Prediction log schema, Prometheus-style `/metrics` (request count, errors,
  latency histogram, schema-failure rate, fallback rate), a small report script summarizing
  a log file.
- 10.2 [NB] `10_failure_analysis.ipynb`: collect failures (invalid JSON, low confidence,
  test/BANKING77 errors); export a review CSV with blank `correct_category`,
  `correct_intent`, `correct_response`.
- 10.3 [HUMAN] Label a sample (100–200 rows is enough).
- 10.4 [PY] Validate reviewed rows, build **dataset v2**. **Never add any test-set item or
  near-duplicate of one**; check programmatically against the frozen test set and log
  it. Version with DVC, tag `data-v2.0`, update the DATA_CARD.
- 10.5 **[GATE]** approve v2 retraining cost, retrain, evaluate on the **same untouched
  test set**, compare v1 vs v2 in `reports/v1_vs_v2.md` (report honestly if no gain).
- 10.6 [PY] Promotion via the quality gate; a **rollback script** (switch registry alias to
  the previous version and redeploy the previous adapter), tested in dry-run.
- 10.7 [PY] `train.yml` GitHub Actions workflow (`workflow_dispatch` only) that launches a
  SageMaker job via **AWS OIDC role assumption** (no long-lived keys), then evaluation
  and the quality gate. **No automatic deployment.** [HUMAN] creates the OIDC role from
  the provided snippets. Write `docs/CICD.md` with a diagram.

Acceptance: the loop failures → review → v2 → retrain → evaluate → gate → promote/rollback
is documented and demonstrated.

---

## STAGE 11 — Portfolio packaging

**Goal:** a recruiter understands it in five minutes and verifies it in thirty.

Subtasks
- 11.1 [DOC] `README.md`: pitch, headline results table, architecture diagram (Mermaid),
  live demo link + video, key engineering decisions, reproduce steps (`make` commands),
  repo tour, cost summary, limitations, next steps.
- 11.2 [DOC] Finalize `DATA_CARD.md`, `MODEL_CARD.md`, `COST.md`, `DECISIONS.md`,
  `CICD.md`, `KAGGLE_RUNBOOK.md`.
- 11.3 [DOC] `docs/INTERVIEW_QA.md`: answers (with evidence links) to the Plan §38 questions:
  why fine-tune, data prep, leakage prevention, why QLoRA, why this model, proof it
  helped, how to know what's in production, retraining, failure handling, inference cost,
  rollback. **The human rewrites these in their own words.**
- 11.4 [AGENT] Hygiene: remove dead code and notebook noise, run a secrets scan (including
  git history), verify `LICENSE`, pinned versions, and that `make setup && make test`
  works from a clean clone.
- 11.5 [HUMAN] Tag `v1.0.0`, pin the repo on GitHub, and draft CV/LinkedIn bullets from
  the real results.

Acceptance: a stranger can clone, read the README, and reproduce data + baseline + tests;
every performance claim links to a report file.

---

## Appendix A — Definition of done for any subtask

- Notebook runs top to bottom; extracted code is typed, linted, and tested.
- Behavior is configurable (no hard-coded paths or hyperparameters).
- Results come from scripts and are saved to files, never typed by hand.
- Docs updated, commit made, cloud cost impact recorded.

## Appendix B — Things the agent must never doD

- Train or run the 4B model locally on CPU.
- Launch billable AWS resources without a [GATE] approval.
- Peek at the test set for tuning, or add test items to training.
- Put BANKING77 into training in v1.
- Invent metrics, costs, or results; write the human's learning-log answers.
- Commit secrets, raw data, or model weights.
- Leave GPU endpoints running.
