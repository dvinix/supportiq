# SupportIQ --- End-to-End Production LLM Fine-Tuning Platform

## 1. Project Overview

**SupportIQ** is an end-to-end production-oriented machine learning
platform for fine-tuning an open-weight Large Language Model (LLM) for
customer-support intelligence.

The project is intentionally designed around a realistic production
problem rather than a simple "fine-tune a model and generate text" demo.

The system will:

1.  Ingest public customer-support data.
2.  Profile and validate the raw data.
3.  Detect and anonymize PII.
4.  Normalize and deduplicate examples.
5.  Build a controlled support taxonomy.
6.  Create leakage-safe train/validation/test splits.
7.  Establish non-LLM and LLM baselines.
8.  Fine-tune an open-weight LLM using Supervised Fine-Tuning (SFT) +
    QLoRA.
9.  Evaluate the fine-tuned model against the baselines.
10. Track experiments and model versions.
11. Package the model for inference.
12. Serve the model through an API.
13. Deploy the system to AWS.
14. Monitor production performance.
15. Collect human-reviewed failures.
16. Feed validated failures back into future dataset versions.
17. Retrain and evaluate new model versions.

The central engineering question is not:

> "Can we fine-tune an LLM?"

It is:

> "Does a specialized fine-tuned model provide a measurable production
> advantage over simpler alternatives for a defined customer-support
> task?"

------------------------------------------------------------------------

# 2. Business / ML Problem

Customer-support organizations receive large volumes of conversations
and tickets.

A support platform needs to convert unstructured conversations into
structured information that can be consumed by downstream systems.

For example:

### Input

``` text
Customer:

How can I cancel my order? I placed it a few minutes ago
but I no longer need it.
```

### Expected output

``` json
{
  "category": "ORDER",
  "intent": "cancel_order",
  "response": "I can help you with cancelling your order..."
}
```

The output can then be used by:

-   CRM systems
-   ticket routing
-   agent-assistance systems
-   support analytics
-   workflow automation
-   escalation systems

------------------------------------------------------------------------

# 3. Core ML Task

The first version of SupportIQ will focus on three related capabilities:

## 3.1 Intent Classification

Determine what the customer is trying to accomplish.

Example:

``` text
Input:
"I want to cancel my order."

Output:
cancel_order
```

## 3.2 Category Classification

Determine the broad support category.

Example:

``` text
ORDER
```

## 3.3 Response Generation

Generate a suitable support response.

Example:

``` text
"I can help you cancel your order. Please provide your
order number so I can check whether it is still eligible
for cancellation."
```

The model will produce these outputs in a controlled structured format.

------------------------------------------------------------------------

# 4. Why Fine-Tuning?

SupportIQ will explicitly compare multiple approaches.

## Baseline A --- Traditional ML

A TF-IDF + Logistic Regression classifier.

Purpose:

-   establish a simple baseline
-   determine whether an LLM is necessary for classification

## Baseline B --- Base LLM

Qwen3-4B-Base without fine-tuning.

Purpose:

-   measure zero/few-shot behavior

## Baseline C --- Prompted LLM

Qwen3-4B-Base with carefully designed prompting and structured-output
instructions.

Purpose:

-   measure how much prompting alone can accomplish

## Candidate --- Fine-Tuned LLM

Qwen3-4B-Base fine-tuned with SFT + QLoRA.

Purpose:

-   determine whether specialization improves the task enough to justify
    the additional training and operational complexity

The final project will not assume beforehand that fine-tuning is
superior.

The evaluation will determine that.

------------------------------------------------------------------------

# 5. Primary Dataset

## Bitext Customer Support LLM Training Dataset

Source:

https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

The dataset contains approximately 26,872 customer-support examples and
covers 27 intents across 11 categories.

Typical fields include:

-   instruction
-   category
-   intent
-   response

Example conceptual record:

``` json
{
  "instruction": "How can I cancel my order?",
  "category": "ORDER",
  "intent": "cancel_order",
  "response": "..."
}
```

### Why this is the primary dataset

It directly supports the intended supervised fine-tuning task:

``` text
Customer request
        +
Intent/category
        +
Expected response
```

It is therefore more appropriate as the initial training foundation than
combining unrelated datasets simply to increase dataset size.

### Important limitation

The Bitext dataset is not equivalent to naturally occurring enterprise
support logs. Its provenance and synthetic/hybrid generation
characteristics must be documented in the project.

We will therefore treat it as a controlled research/training dataset,
not claim that it represents real production traffic.

------------------------------------------------------------------------

# 6. Secondary Dataset

## BANKING77

Source:

https://huggingface.co/datasets/PolyAI/banking77

BANKING77 contains approximately 13,083 banking customer-service queries
covering 77 fine-grained intents.

It will **not** be blindly merged into the Bitext training dataset.

Instead, it will initially be used as an external
robustness/generalization evaluation source.

Purpose:

``` text
Training distribution
        ↓
Bitext
        ↓
Fine-tuned model
        ↓
External evaluation
        ↓
BANKING77
```

This helps investigate whether the model's learned support behavior
generalizes beyond the exact training distribution.

Because the label taxonomies differ, the evaluation protocol will be
designed explicitly rather than pretending that the two taxonomies are
identical.

------------------------------------------------------------------------

# 7. Future Data Sources

Additional datasets may be introduced only when a measurable gap is
identified.

For every new source, the project will document:

-   source
-   license
-   domain
-   task
-   label taxonomy
-   data provenance
-   synthetic vs human-generated content
-   PII risk
-   duplicate/overlap risk
-   quality
-   intended usage

The project will not add data merely because it increases the number of
examples.

------------------------------------------------------------------------

# 8. Data Architecture

``` text
                  PUBLIC DATA
                      |
          +-----------+-----------+
          |                       |
       Bitext                  BANKING77
          |                       |
          v                       |
      Raw Storage                 |
          |                       |
          v                       |
     Data Profiling              |
          |                       |
          v                       |
   Schema Validation             |
          |                       |
          v                       |
      PII Handling               |
          |                       |
          v                       |
     Normalization               |
          |                       |
          v                       |
    Deduplication                |
          |                       |
          v                       |
    Quality Filtering            |
          |                       |
          v                       |
   Taxonomy Validation           |
          |                       |
          v                       |
 Train / Validation / Test       |
          |                       |
          v                       |
     SFT Formatting              |
          |                       |
          v                       |
   Versioned Training Data       |
                                  |
                                  v
                          External Evaluation
```

------------------------------------------------------------------------

# 9. Data Pipeline

The data pipeline will be implemented as independent, testable stages.

``` text
Raw Dataset
    |
    v
Ingestion
    |
    v
Profiling
    |
    v
Schema Validation
    |
    v
PII Detection / Anonymization
    |
    v
Normalization
    |
    v
Exact Deduplication
    |
    v
Near-Duplicate Analysis
    |
    v
Quality Filtering
    |
    v
Taxonomy Mapping
    |
    v
Leakage Prevention
    |
    v
Train / Validation / Test Split
    |
    v
SFT Formatting
    |
    v
Versioned Dataset
```

------------------------------------------------------------------------

# 10. Data Profiling

Before modifying the data, the system will generate a profiling report.

The report should include:

-   total number of rows
-   missing values
-   duplicate counts
-   category distribution
-   intent distribution
-   instruction length distribution
-   response length distribution
-   outliers
-   invalid labels
-   possible data leakage
-   source metadata

Example:

``` text
Dataset: Bitext
Rows: <computed>
Categories: <computed>
Intents: <computed>

Missing instructions: <computed>
Missing responses: <computed>

Duplicate instructions: <computed>
Duplicate instruction-response pairs: <computed>

Median instruction length: <computed>
P95 instruction length: <computed>

Median response length: <computed>
P95 response length: <computed>
```

All values must be computed from the actual dataset.

------------------------------------------------------------------------

# 11. Data Quality

The pipeline will implement automated quality checks.

## Missing Values

Reject or quarantine records with missing required fields.

## Invalid Labels

Check that:

``` text
category ∈ approved taxonomy
intent ∈ approved taxonomy
```

## Duplicate Detection

Detect:

1.  exact instruction duplicates
2.  exact instruction + response duplicates
3.  near duplicates
4.  highly similar templates

## Length Filtering

Identify examples that are:

-   empty
-   unusually short
-   unusually long
-   malformed

## Content Quality

Flag:

-   malformed records
-   irrelevant responses
-   contradictory metadata
-   broken encoding
-   suspicious examples

------------------------------------------------------------------------

# 12. PII Protection

Customer-support data can contain sensitive information.

Before training, the pipeline will inspect text for PII.

Potential entities include:

-   email addresses
-   phone numbers
-   payment card numbers
-   account identifiers
-   names where appropriate
-   addresses
-   other identifiable information

Example:

``` text
Before:

My email is user@example.com and my card is 1234...

After:

My email is <EMAIL> and my card is <CARD_NUMBER>.
```

The project will use a dedicated PII detection/anonymization component
rather than relying only on regexes.

Potential technology:

``` text
Microsoft Presidio
```

PII handling will be tested independently.

------------------------------------------------------------------------

# 13. Taxonomy

The project will maintain a controlled taxonomy.

Example:

``` text
ORDER
  - cancel_order
  - track_order
  - modify_order

PAYMENT
  - payment_failed
  - payment_pending

ACCOUNT
  - account_access
  - account_update
```

The exact taxonomy will be derived from the source dataset and
documented rather than invented arbitrarily.

A mapping layer will allow source-specific labels to map into the
SupportIQ internal representation.

------------------------------------------------------------------------

# 14. Data Splitting

The final dataset will be divided into:

``` text
Train       ~80%
Validation  ~10%
Test        ~10%
```

The exact distribution can be adjusted based on the final data.

Splitting must account for:

-   label distribution
-   duplicates
-   near duplicates
-   template overlap
-   leakage

The test set must remain isolated from training decisions.

------------------------------------------------------------------------

# 15. Dataset Versioning

DVC will be used for dataset versioning.

Conceptual lifecycle:

``` text
v0.1
Raw data

v0.2
Cleaned data

v0.3
PII processed

v0.4
Deduplicated

v0.5
Taxonomy mapped

v1.0
Training-ready dataset

v1.1
Production failure samples

v2.0
Retraining dataset
```

Git will track:

-   code
-   configuration
-   schemas
-   metadata

DVC will track:

-   datasets
-   large artifacts

------------------------------------------------------------------------

# 16. Model

## Primary Model

### Qwen3-4B-Base

Source:

https://huggingface.co/Qwen/Qwen3-4B-Base

The model is a 4B-parameter causal language model with a 32K context
window and Apache-2.0 licensing according to its model card.

The Base model is intentionally selected for the supervised fine-tuning
experiment rather than using an already instruction-tuned model.

------------------------------------------------------------------------

# 17. Fine-Tuning Method

The primary training approach will be:

``` text
Supervised Fine-Tuning
        +
LoRA
        +
4-bit Quantization
        =
QLoRA-based SFT
```

Conceptual architecture:

``` text
Qwen3-4B-Base
       |
       v
4-bit Quantization
       |
       v
Frozen Base Model
       |
       +
       |
LoRA Adapters
       |
       v
Supervised Fine-Tuning
       |
       v
SupportIQ Adapter
```

Core libraries:

-   PyTorch
-   Transformers
-   TRL
-   PEFT
-   bitsandbytes
-   Accelerate

------------------------------------------------------------------------

# 18. Training Data Format

Training records will be converted into conversational SFT examples.

Conceptual format:

``` json
{
  "messages": [
    {
      "role": "user",
      "content": "How can I cancel my order?"
    },
    {
      "role": "assistant",
      "content": "{\"category\":\"ORDER\",\"intent\":\"cancel_order\",\"response\":\"...\"}"
    }
  ]
}
```

The final formatting will follow the selected model's
tokenizer/chat-template requirements.

------------------------------------------------------------------------

# 19. Output Contract

The model's generated output will follow a strict schema.

Example:

``` json
{
  "category": "ORDER",
  "intent": "cancel_order",
  "response": "..."
}
```

The serving layer will validate the output against a Pydantic schema.

Invalid output should not silently enter downstream systems.

------------------------------------------------------------------------

# 20. Baseline Experiments

Before fine-tuning:

## Baseline 1

``` text
TF-IDF + Logistic Regression
```

Measures whether simple supervised ML is sufficient for intent
classification.

## Baseline 2

``` text
Qwen3-4B-Base
```

Measures base-model performance.

## Baseline 3

``` text
Qwen3-4B-Base + structured prompt
```

Measures prompt-only performance.

## Candidate

``` text
Qwen3-4B-Base + SFT + QLoRA
```

Measures the actual benefit of fine-tuning.

------------------------------------------------------------------------

# 21. Evaluation

Evaluation will contain multiple layers.

## Classification Metrics

-   Accuracy
-   Macro F1
-   Weighted F1
-   Per-intent F1
-   Confusion matrix

## Structured Output Metrics

-   JSON validity
-   Schema validity
-   Field-level accuracy
-   Category accuracy
-   Intent accuracy

## Generation Quality

Human/LLM-assisted evaluation can measure:

-   relevance
-   correctness
-   helpfulness
-   consistency

Subjective evaluation will be clearly separated from objective metrics.

------------------------------------------------------------------------

# 22. Error Analysis

The system will classify model failures.

Possible categories:

``` text
Wrong intent
Wrong category
Invalid JSON
Invalid schema
Hallucinated information
Incomplete response
Irrelevant response
Instruction-following failure
Ambiguous request
Out-of-distribution request
```

For every model version, an error analysis report should identify
recurring failure modes.

------------------------------------------------------------------------

# 23. Experiment Tracking

MLflow will track training and evaluation runs.

Each experiment should record:

``` text
model_name
base_model_revision
dataset_version
git_commit
learning_rate
batch_size
gradient_accumulation
epochs
sequence_length
LoRA rank
LoRA alpha
LoRA dropout
quantization configuration

training_loss
validation_loss
intent_f1
category_f1
json_validity
schema_validity
latency
```

This enables reproducibility.

------------------------------------------------------------------------

# 24. Model Registry

Models will be versioned.

Example:

``` text
SupportIQ
|
+-- v0.1
+-- v0.2
+-- v1.0
+-- v1.1
```

Each model version should reference:

-   base model
-   adapter
-   dataset version
-   training configuration
-   evaluation results
-   Git commit
-   MLflow run

The project should be able to answer:

> "Exactly what produced the model currently deployed?"

------------------------------------------------------------------------

# 25. Inference Architecture

The serving architecture will be:

``` text
Client
  |
  v
FastAPI
  |
  v
Input Validation
  |
  v
Model Server
  |
  v
Qwen3-4B + LoRA
  |
  v
Structured Output Validation
  |
  v
Response
```

vLLM can be evaluated as the model serving layer.

------------------------------------------------------------------------

# 26. API

Example:

``` http
POST /v1/support/analyze
```

Request:

``` json
{
  "conversation": [
    {
      "role": "customer",
      "content": "How can I cancel my order?"
    }
  ]
}
```

Response:

``` json
{
  "category": "ORDER",
  "intent": "cancel_order",
  "response": "...",
  "model_version": "supportiq-v1.0"
}
```

Additional endpoints:

``` text
GET /health
GET /ready
GET /model
GET /metrics
POST /v1/support/analyze
```

------------------------------------------------------------------------

# 27. Cloud Deployment

## Cloud Provider

### AWS

The target architecture:

``` text
                    AWS
                     |
       +-------------+--------------+
       |             |              |
       v             v              v
      S3            ECR         CloudWatch
       |             |              |
       |        Docker Images     Logs/
       |                          Metrics
       |
       v
   SageMaker
       |
   +---+---+
   |       |
Training  Endpoint
   |       |
   v       v
QLoRA    Inference
```

------------------------------------------------------------------------

# 28. AWS Services

## Amazon S3

Store:

-   raw dataset artifacts
-   processed datasets
-   DVC artifacts
-   model adapters
-   evaluation reports

## Amazon ECR

Store:

-   training container
-   inference container

## Amazon SageMaker

Use for:

-   GPU training jobs
-   model deployment
-   inference endpoints

## CloudWatch

Monitor:

-   logs
-   errors
-   latency
-   endpoint health
-   infrastructure metrics

## IAM

Implement least-privilege access.

## VPC

Where required, isolate production resources and control network access.

------------------------------------------------------------------------

# 29. Training on Cloud

The GPU should not run continuously.

Training flow:

``` text
Dataset version
      |
      v
S3
      |
      v
SageMaker Training Job
      |
      v
GPU
      |
      v
QLoRA
      |
      v
Evaluation
      |
      v
Model Artifact
      |
      v
S3 / MLflow
      |
      v
Training Job Terminates
```

This keeps training resources ephemeral.

------------------------------------------------------------------------

# 30. Inference Deployment

The fine-tuned model will be packaged and deployed behind an inference
endpoint.

Conceptually:

``` text
Internet
   |
   v
API Layer
   |
   v
FastAPI
   |
   v
SageMaker Endpoint
   |
   v
Qwen3-4B + SupportIQ Adapter
```

The deployment configuration should support:

-   health checks
-   logging
-   versioning
-   rollback
-   controlled rollout

------------------------------------------------------------------------

# 31. Production Monitoring

Infrastructure metrics:

``` text
CPU
GPU
Memory
Request count
Errors
Latency
```

Application metrics:

``` text
P50 latency
P95 latency
P99 latency
JSON validity
Schema failure rate
Fallback rate
Human escalation rate
```

Model-quality metrics can be computed when feedback labels become
available.

------------------------------------------------------------------------

# 32. Production Feedback Loop

The system will support continuous improvement.

``` text
Production Requests
        |
        v
Predictions
        |
        v
Logging / Monitoring
        |
        +--------------------+
        |                    |
      Good                  Failure
                             |
                             v
                       Human Review
                             |
                             v
                       Correct Label
                             |
                             v
                       Dataset v2
                             |
                             v
                         Retraining
                             |
                             v
                          Evaluation
                             |
                             v
                       Candidate Model
                             |
                             v
                       Deployment
```

This creates a complete model lifecycle.

------------------------------------------------------------------------

# 33. CI/CD

GitHub Actions will handle:

## Pull Request

``` text
Lint
  |
Unit Tests
  |
Data Tests
  |
Integration Tests
  |
Docker Build
```

## Model Training Workflow

``` text
Dataset Version
      |
      v
Training Job
      |
      v
Evaluation
      |
      v
Quality Gate
      |
      +---- Fail ----> Stop
      |
      v
Model Registry
      |
      v
Deployment
```

A model should not automatically reach production just because training
completed.

------------------------------------------------------------------------

# 34. Infrastructure as Code

Terraform will define cloud infrastructure where practical.

Example:

``` text
infrastructure/
└── terraform/
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    ├── s3.tf
    ├── ecr.tf
    ├── iam.tf
    └── sagemaker.tf
```

This makes the environment reproducible.

------------------------------------------------------------------------

# 35. Final Repository Structure

``` text
supportiq/
|
+-- data/
|   +-- raw/
|   +-- interim/
|   +-- processed/
|   +-- splits/
|   +-- README.md
|
+-- src/
|   +-- data/
|   |   +-- ingest.py
|   |   +-- profile.py
|   |   +-- validate.py
|   |   +-- pii.py
|   |   +-- normalize.py
|   |   +-- deduplicate.py
|   |   +-- taxonomy.py
|   |   +-- split.py
|   |   +-- format_sft.py
|   |
|   +-- training/
|   |   +-- train.py
|   |   +-- qlora.py
|   |   +-- config.py
|   |
|   +-- evaluation/
|   |   +-- classification.py
|   |   +-- generation.py
|   |   +-- schema.py
|   |   +-- robustness.py
|   |   +-- report.py
|   |
|   +-- inference/
|   |   +-- predictor.py
|   |
|   +-- monitoring/
|       +-- metrics.py
|
+-- schemas/
|   +-- dataset.py
|   +-- prediction.py
|
+-- configs/
|   +-- data.yaml
|   +-- training.yaml
|   +-- evaluation.yaml
|   +-- deployment.yaml
|
+-- app/
|   +-- main.py
|   +-- routes/
|
+-- tests/
|
+-- infrastructure/
|   +-- terraform/
|
+-- docker/
|
+-- notebooks/
|   +-- exploratory/
|
+-- .github/
|   +-- workflows/
|
+-- dvc.yaml
+-- pyproject.toml
+-- README.md
```

------------------------------------------------------------------------

# 36. Implementation Milestones

## Milestone 1 --- Foundation

-   repository
-   Python environment
-   dependency management
-   configuration
-   logging
-   testing
-   pre-commit/linting

## Milestone 2 --- Data Ingestion

-   download Bitext
-   preserve raw data
-   record dataset metadata
-   compute dataset checksum
-   document license/source

## Milestone 3 --- Data Profiling

-   statistics
-   label distributions
-   length distributions
-   missing values
-   duplicates
-   anomalies

## Milestone 4 --- Data Quality

-   schema validation
-   malformed record handling
-   duplicate detection
-   PII detection
-   normalization

## Milestone 5 --- Dataset Construction

-   taxonomy
-   leakage prevention
-   train/validation/test split
-   SFT formatting
-   dataset versioning

## Milestone 6 --- Baselines

-   TF-IDF classifier
-   Qwen base inference
-   prompted Qwen

## Milestone 7 --- Fine-Tuning

-   cloud GPU environment
-   QLoRA
-   SFT
-   checkpointing
-   resume support
-   configuration-driven training

## Milestone 8 --- Evaluation

-   classification metrics
-   structured-output metrics
-   generation evaluation
-   external robustness evaluation
-   error analysis

## Milestone 9 --- MLOps

-   MLflow
-   dataset/model lineage
-   model registry
-   reproducible experiments

## Milestone 10 --- Serving

-   model packaging
-   vLLM evaluation
-   FastAPI
-   Pydantic validation
-   health/readiness endpoints

## Milestone 11 --- Cloud

-   AWS S3
-   ECR
-   SageMaker
-   CloudWatch
-   IAM
-   Terraform

## Milestone 12 --- Production Loop

-   monitoring
-   human review
-   failure dataset
-   dataset v2
-   retraining workflow
-   model promotion
-   rollback

------------------------------------------------------------------------

# 37. Definition of Done

The project is complete only when we can demonstrate:

### Data

-   reproducible ingestion
-   data profiling
-   quality checks
-   PII handling
-   deduplication
-   leakage prevention
-   versioned dataset

### ML

-   baseline models
-   QLoRA fine-tuning
-   reproducible training
-   tracked experiments
-   evaluation
-   error analysis

### MLOps

-   dataset lineage
-   model lineage
-   model registry
-   model promotion

### Engineering

-   tested API
-   Docker image
-   structured outputs
-   health checks
-   CI/CD

### Cloud

-   cloud training
-   model artifact storage
-   deployed inference
-   monitoring
-   infrastructure as code

### Continuous Learning

-   production feedback
-   human review
-   retraining dataset
-   new model evaluation
-   controlled redeployment

------------------------------------------------------------------------

# 38. Recruiter Demonstration

The final demonstration should tell a complete engineering story:

``` text
Problem
  ↓
Data
  ↓
Data Quality
  ↓
Baseline
  ↓
Fine-Tuning
  ↓
Evaluation
  ↓
Improvement
  ↓
Model Registry
  ↓
Deployment
  ↓
Production Monitoring
  ↓
Failure Analysis
  ↓
Retraining
```

The recruiter should be able to ask:

> Why did you fine-tune?

> How did you prepare the data?

> How did you prevent leakage?

> Why QLoRA?

> Why this model?

> How did you prove fine-tuning helped?

> How do you know which model is in production?

> How would you retrain it?

> What happens when the model fails?

> How much does inference cost?

> How would you roll back a bad model?

And the repository should contain evidence for each answer.

------------------------------------------------------------------------

# 39. Final Technology Stack

  Layer                 Technology
  --------------------- -------------------------------------
  Language              Python
  Base model            Qwen3-4B-Base
  Training              PyTorch
  Fine-tuning           SFT + QLoRA
  PEFT                  Hugging Face PEFT
  Training framework    Hugging Face TRL
  Quantization          bitsandbytes
  Dataset               Bitext Customer Support
  External evaluation   BANKING77
  Data processing       Polars / PyArrow
  Validation            Pydantic
  PII                   Microsoft Presidio
  Dataset versioning    DVC
  Experiment tracking   MLflow
  Model registry        MLflow
  Model artifacts       S3 / Hugging Face where appropriate
  Inference             vLLM
  API                   FastAPI
  Database              PostgreSQL
  Cache/queue           Redis where needed
  Containers            Docker
  CI/CD                 GitHub Actions
  Cloud                 AWS
  Training              SageMaker
  Storage               S3
  Container registry    ECR
  Monitoring            CloudWatch + Prometheus/Grafana
  Infrastructure        Terraform

------------------------------------------------------------------------

# 40. Guiding Principles

1.  **No training before the data pipeline is understood.**
2.  **No fine-tuning without a baseline.**
3.  **No test-set optimization.**
4.  **No fake metrics.**
5.  **No arbitrary dataset mixing.**
6.  **No synthetic labels presented as ground truth.**
7.  **No unnecessary infrastructure.**
8.  **Every model version must have dataset and configuration lineage.**
9.  **Every production model must have an evaluation report.**
10. **Production failures should become potential training data only
    after validation/human review.**
11. **Fine-tuning must have a measurable reason to exist.**
12. **The final project should demonstrate engineering decisions, not
    just model training.**

------------------------------------------------------------------------

## Project Success Criteria

The final SupportIQ system should demonstrate that a fine-tuned
open-weight model can perform the defined customer-support task with
measurable improvements or operational advantages over the selected
baselines, while being reproducibly trained, evaluated, versioned,
deployed, monitored, and continuously improved.

The project should be presented as an **LLM/ML engineering system**, not
merely as a fine-tuning experiment.
