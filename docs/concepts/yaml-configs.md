# Concept: Configuration-Driven Machine Learning

## 1. What Problem Does It Solve?
Hardcoding hyperparameters, file paths, or taxonomy lists directly inside Python scripts leads to severe production bugs:
- You cannot change the batch size, learning rate, or paths without editing source code.
- Teammates or Docker containers running with different directory layouts crash.
- Experiment tracking becomes impossible because you cannot version what settings produced which model.

---

## 2. The Core Solution: YAML Configs
SupportIQ strictly separates **Configuration** from **Code**:
- **Code (`src/supportiq/`):** Contains pure, reusable logic and algorithms (data loading, clustering, training loops).
- **Configs (`configs/`):** Contains the parameters that govern the run:
  - `configs/data.yaml`: Dataset paths, random seeds, split ratios (80/10/10), clustering threshold.
  - `configs/taxonomy.yaml`: The official list of 11 categories and 27 intents.
  - `configs/training.yaml`: LoRA hyperparameters ($r=16, \alpha=32$), learning rate, epochs.

---

## 3. How It Connects to Pydantic
When code executes, a config loader reads the YAML file and validates it against a strict **Pydantic schema**:
- If someone typos a setting (e.g., `train_ratio: "eighty"` instead of `0.80`), Pydantic fails fast at startup with a clear error message.
- The pipeline never crashes halfway through expensive GPU training due to an invalid setting.
