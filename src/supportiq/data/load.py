"""Data ingestion, raw artifact preservation, and metadata tracking."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from datasets import load_dataset
from huggingface_hub import HfApi
from pydantic import BaseModel, Field

from supportiq.core.logger import get_logger

logger = get_logger(__name__)


class RawArtifactInfo(BaseModel):
    """Metadata describing a single raw data artifact."""

    sha256: str
    size_bytes: int


class DatasetMetadata(BaseModel):
    """Metadata capturing the source, revision, and checksums of ingested raw data."""

    dataset_name: str
    source_url: str
    revision_sha: str
    license: str
    download_timestamp_utc: str
    row_count: int
    column_count: int
    columns: list[str]
    artifacts: dict[str, RawArtifactInfo] = Field(default_factory=dict)


def compute_file_sha256(filepath: Path | str, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        filepath: Path to the target file.
        chunk_size: Buffer chunk size in bytes.

    Returns:
        Hex-encoded SHA-256 checksum string.
    """
    path = Path(filepath)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def ingest_raw_dataset(
    dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    output_dir: Path | str = "data/raw",
    split: str = "train",
    revision: str | None = None,
) -> tuple[pl.DataFrame, DatasetMetadata]:
    """Download the raw dataset from Hugging Face, save untouched artifacts, and record metadata.

    Args:
        dataset_name: Hugging Face dataset identifier.
        output_dir: Directory where raw files and METADATA.json are saved.
        split: Hugging Face split to download (default: 'train').
        revision: Specific commit hash or branch name. If None, queries Hugging Face API.

    Returns:
        A tuple of (loaded Polars DataFrame, DatasetMetadata instance).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Hugging Face Hub for metadata: %s", dataset_name)
    api = HfApi()
    repo_info = api.dataset_info(dataset_name)
    resolved_revision = repo_info.sha if revision in (None, "main") else revision
    card_data: Any = repo_info.card_data or {}
    license_name = getattr(card_data, "license", "cdla-sharing-1.0")

    logger.info(
        "Downloading dataset split %s (split: %s, revision: %s)",
        dataset_name,
        split,
        resolved_revision,
    )
    hf_ds = load_dataset(dataset_name, split=split, revision=resolved_revision)
    df = pl.from_arrow(hf_ds.data.table)

    parquet_file = out_path / "bitext_raw.parquet"
    csv_file = out_path / "bitext_raw.csv"
    metadata_file = out_path / "METADATA.json"

    logger.info("Persisting raw artifacts: %s, %s", parquet_file, csv_file)
    df.write_parquet(parquet_file)
    df.write_csv(csv_file)

    parquet_sha = compute_file_sha256(parquet_file)
    csv_sha = compute_file_sha256(csv_file)

    metadata = DatasetMetadata(
        dataset_name=dataset_name,
        source_url=f"https://huggingface.co/datasets/{dataset_name}",
        revision_sha=resolved_revision,
        license=str(license_name),
        download_timestamp_utc=datetime.now(UTC).isoformat(),
        row_count=df.height,
        column_count=df.width,
        columns=df.columns,
        artifacts={
            "bitext_raw.parquet": RawArtifactInfo(
                sha256=parquet_sha,
                size_bytes=parquet_file.stat().st_size,
            ),
            "bitext_raw.csv": RawArtifactInfo(
                sha256=csv_sha,
                size_bytes=csv_file.stat().st_size,
            ),
        },
    )

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    logger.info(
        "Raw dataset ingestion complete (%d rows, %d columns, sha256=%s)",
        df.height,
        df.width,
        parquet_sha,
    )
    return df, metadata


def load_raw_dataframe(data_dir: Path | str = "data/raw") -> pl.DataFrame:
    """Load the preserved raw dataset directly from disk without re-downloading.

    Args:
        data_dir: Directory containing bitext_raw.parquet.

    Returns:
        Polars DataFrame of the raw dataset.
    """
    parquet_path = Path(data_dir) / "bitext_raw.parquet"
    if not parquet_path.exists():
        alt_path = Path("..") / data_dir / "bitext_raw.parquet"
        if alt_path.exists():
            parquet_path = alt_path
        else:
            msg = f"Raw dataset not found at {parquet_path} or {alt_path}. Run ingest_raw_dataset first."
            raise FileNotFoundError(msg)
    return pl.read_parquet(parquet_path)
