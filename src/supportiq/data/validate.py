"""Dataset validation, schema enforcement, and quarantine routing."""

import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import ValidationError

from schemas.dataset import CustomerSupportRecord, QuarantinedRecord, ValidationSummary
from supportiq.core.logger import get_logger

logger = get_logger(__name__)


def validate_single_record(
    record: dict[str, Any],
) -> tuple[bool, str | None, CustomerSupportRecord | None]:
    """Validate a single dictionary against CustomerSupportRecord schema.

    Args:
        record: Raw key-value dictionary representing one row.

    Returns:
        Tuple of (is_valid, error_reason, validated_model_or_none).
    """
    try:
        validated = CustomerSupportRecord(**record)
        return True, None, validated
    except ValidationError as e:
        error_msg = "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
        return False, error_msg, None


def validate_dataframe(
    df: pl.DataFrame,
    quarantine_path: Path | str = "data/interim/quarantine.jsonl",
) -> tuple[pl.DataFrame, list[QuarantinedRecord], ValidationSummary]:
    """Validate an entire Polars DataFrame, routing invalid rows to quarantine.

    Args:
        df: Input raw DataFrame.
        quarantine_path: Target JSONL path for invalid rows.

    Returns:
        Tuple of:
          - Valid rows as a new Polars DataFrame.
          - List of QuarantinedRecord instances.
          - ValidationSummary metrics.
    """
    logger.info("Starting schema validation for %d records", df.height)
    valid_rows: list[dict[str, Any]] = []
    quarantined_records: list[QuarantinedRecord] = []

    for idx, row in enumerate(df.iter_rows(named=True)):
        is_valid, reason, model_obj = validate_single_record(row)
        if is_valid and model_obj is not None:
            valid_rows.append(model_obj.model_dump())
        else:
            quarantined_records.append(
                QuarantinedRecord(
                    row_index=idx,
                    raw_record=row,
                    quarantine_reason=reason or "Unknown validation error",
                )
            )

    # Persist quarantine log if any failed records exist or ensure empty file exists
    q_file = Path(quarantine_path)
    q_file.parent.mkdir(parents=True, exist_ok=True)
    with open(q_file, "w", encoding="utf-8") as f:
        for q in quarantined_records:
            f.write(json.dumps(q.model_dump()) + "\n")

    rate = round(len(quarantined_records) / df.height * 100, 2) if df.height > 0 else 0.0
    summary = ValidationSummary(
        total_processed=df.height,
        valid_count=len(valid_rows),
        quarantined_count=len(quarantined_records),
        quarantine_rate=rate,
    )

    logger.info(
        "Validation complete: %d valid, %d quarantined (%.2f%% quarantine rate)",
        summary.valid_count,
        summary.quarantined_count,
        summary.quarantine_rate,
    )

    valid_df = pl.DataFrame(valid_rows) if valid_rows else pl.DataFrame(schema=df.schema)
    return valid_df, quarantined_records, summary
