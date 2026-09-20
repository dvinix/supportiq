# Architecture Decision Records (ADRs)

This document tracks technical decisions made across the SupportIQ project lifecycle, following the format: **Context → Decision → Alternatives Considered → Consequences**.

---

## ADR-001: Python 3.12 and `uv` Package Manager

- **Status:** Accepted
- **Date:** 2026-09-21
- **Deciders:** SupportIQ Team

### Context
SupportIQ requires modern Python features (Pydantic v2, structured typing, Polars 1.0+), fast and deterministic dependency resolution, and lightweight CI setup. Python 3.14 (system default on host) is currently too new for several core ML wheels (PyTorch, TRL, bitsandbytes), while Python 3.11/3.12 have mature pre-built wheels.

### Decision
Use Python 3.12 as the target runtime and Astral's `uv` as the single package manager for virtual environment management, lockfile generation, and tool execution.

### Alternatives Considered
- **Poetry:** Feature-rich but noticeably slower dependency resolution and environment setup in CI.
- **pip-tools / standard venv:** Requires multiple fragmented tools (`pip-compile`, `venv`, `pip`) without built-in Python version pinning.
- **Python 3.14:** Fails to build or install pre-compiled binary packages for foundational ML libraries.

### Consequences
- Fast installs (sub-second resolution and installation).
- Deterministic cross-platform builds in GitHub Actions using `astral-sh/setup-uv`.
- Clear dependency groups (`dev`, `data`, `train`, `serve`).

---

## ADR-002: Structured JSON Logging & Shared Logger

- **Status:** Accepted
- **Date:** 2026-09-21
- **Deciders:** SupportIQ Team

### Context
In production triage systems and ML training pipelines, logs must be machine-readable, ingestible by CloudWatch/Datadog, and traceable across request and training lifecycles without relying on unstructured console printouts.

### Decision
Standardize on a structured JSON logger using Python's standard library `logging` and a custom `JSONFormatter` providing ISO 8601 UTC timestamps, level, module, line numbers, error stack traces, and arbitrary contextual extra keys.

### Alternatives Considered
- **Plain standard logging (`%(asctime)s - %(message)s`):** Difficult to parse in log aggregators.
- **External heavy logging libraries (`structlog`, `loguru`):** Adds third-party dependencies for functionality easily handled with Python's built-in logging.

### Consequences
- Clean JSON output compatible with cloud log aggregators and local CLI debugging.
- Zero extra dependencies for the core logger.
