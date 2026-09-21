"""Unit tests for dataset ingestion utilities and metadata tracking."""

import hashlib
from pathlib import Path

import polars as pl
import pytest

from supportiq.data.load import (
    DatasetMetadata,
    RawArtifactInfo,
    compute_file_sha256,
    load_raw_dataframe,
)


def test_compute_file_sha256(tmp_path: Path) -> None:
    """Verify SHA-256 calculation matches expected digest."""
    test_file = tmp_path / "sample.txt"
    content = b"SupportIQ test content for hashing"
    test_file.write_bytes(content)

    expected_sha = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(test_file) == expected_sha


def test_dataset_metadata_schema() -> None:
    """Verify DatasetMetadata model validation and serialization."""
    metadata = DatasetMetadata(
        dataset_name="bitext/test",
        source_url="https://huggingface.co/datasets/bitext/test",
        revision_sha="abcdef1234567890",
        license="cdla-sharing-1.0",
        download_timestamp_utc="2026-09-22T00:00:00Z",
        row_count=100,
        column_count=5,
        columns=["flags", "instruction", "category", "intent", "response"],
        artifacts={
            "bitext_raw.parquet": RawArtifactInfo(
                sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                size_bytes=1024,
            )
        },
    )
    serialized = metadata.model_dump()
    assert serialized["row_count"] == 100
    assert "bitext_raw.parquet" in serialized["artifacts"]


def test_load_raw_dataframe(tmp_path: Path) -> None:
    """Verify loading raw DataFrame from an existing parquet file."""
    fake_df = pl.DataFrame(
        {
            "flags": ["B"],
            "instruction": ["cancel my order"],
            "category": ["ORDER"],
            "intent": ["cancel_order"],
            "response": ["Sure, let me cancel that."],
        }
    )
    fake_parquet = tmp_path / "bitext_raw.parquet"
    fake_df.write_parquet(fake_parquet)

    loaded_df = load_raw_dataframe(tmp_path)
    assert loaded_df.shape == (1, 5)
    assert loaded_df["intent"][0] == "cancel_order"


def test_load_raw_dataframe_missing(tmp_path: Path) -> None:
    """Verify FileNotFoundError when raw file is absent."""
    with pytest.raises(FileNotFoundError):
        load_raw_dataframe(tmp_path / "non_existent_directory")
