# Concept: Data Version Control (DVC)

## 1. What Problem Does It Solve?
Git was built for source code (small text files). If you commit 500 MB datasets or model weights into Git:
- Repository cloning slows down dramatically.
- GitHub rejects files larger than 100 MB.
- Git stores the entire file again on every minor edit, bloating the `.git` folder forever.

**DVC solves this by acting as "Git for Data".**

---

## 2. The Core Mechanism
- Large data files (`data/raw/`, `data/processed/`) live on your disk or cloud storage (S3).
- Git only tracks small 100-byte pointer files:
  - `data/raw.dvc` (contains the MD5 hash and file size).
  - `dvc.lock` (records the exact data versions used in the pipeline).
- When a teammate or CI runner clones the repository, they run `dvc pull` to fetch the exact matching dataset.

---

## 3. Pipeline Reproducibility (`dvc.yaml`)
`dvc.yaml` defines the execution stages with their dependencies (`deps`) and outputs (`outs`).
- When you run `dvc repro`:
  - DVC checks if any dependency has changed (e.g. `pipeline.py` or `configs/data.yaml`).
  - If nothing changed: DVC skips execution (`Data and pipelines are up to date`).
  - If code changed: DVC re-executes only the affected stages and updates `dvc.lock`.

---

## 4. Cheat Sheet
- `dvc init`: Initialize DVC in a git repo.
- `dvc add <path>`: Track a large data file/folder with DVC.
- `dvc repro`: Rebuild the data pipeline from end to end.
- `dvc push` / `dvc pull`: Sync data with remote storage (S3 / Google Drive).
