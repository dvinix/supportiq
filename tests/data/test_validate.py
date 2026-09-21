"""Unit tests for dataset schema validation and quarantine mechanisms."""

import json
from pathlib import Path

import polars as pl

from schemas.dataset import CustomerSupportRecord
from supportiq.data.validate import validate_dataframe, validate_single_record


def test_validate_single_record_valid() -> None:
    """Verify that a well-formed record is validated and normalized."""
    record = {
        "flags": "B",
        "instruction": "how can I change my order {{Order Number}}?",
        "category": "order",  # test uppercase normalization
        "intent": "EDIT_ORDER",  # test lowercase normalization
        "response": "You can change your order by visiting the order details page.",
    }
    is_valid, reason, model_obj = validate_single_record(record)
    assert is_valid is True
    assert reason is None
    assert isinstance(model_obj, CustomerSupportRecord)
    assert model_obj.category == "ORDER"
    assert model_obj.intent == "edit_order"


def test_validate_single_record_empty_instruction() -> None:
    """Verify that whitespace-only instructions fail validation."""
    record = {
        "flags": "B",
        "instruction": "   \n\t  ",
        "category": "ORDER",
        "intent": "cancel_order",
        "response": "Order cancellation in progress.",
    }
    is_valid, reason, model_obj = validate_single_record(record)
    assert is_valid is False
    assert model_obj is None
    assert reason is not None
    assert "instruction" in reason


def test_validate_single_record_short_response() -> None:
    """Verify that responses under 5 characters are rejected."""
    record = {
        "flags": "B",
        "instruction": "where is my refund?",
        "category": "REFUND",
        "intent": "track_refund",
        "response": "No.",
    }
    is_valid, reason, model_obj = validate_single_record(record)
    assert is_valid is False
    assert model_obj is None
    assert reason is not None
    assert "response" in reason


def test_validate_dataframe_with_quarantine(tmp_path: Path) -> None:
    """Verify DataFrame validation separates valid records and writes quarantine JSONL."""
    df = pl.DataFrame(
        {
            "flags": ["B", "", "Q"],
            "instruction": [
                "valid query 1",
                "valid query 2",
                "   ",  # invalid whitespace
            ],
            "category": ["ORDER", "ORDER", "REFUND"],
            "intent": ["cancel_order", "cancel_order", "track_refund"],
            "response": [
                "Valid response text 1.",
                "Valid response text 2.",
                "Valid response text 3.",
            ],
        }
    )
    quarantine_file = tmp_path / "quarantine.jsonl"
    valid_df, quarantined_list, summary = validate_dataframe(df, quarantine_path=quarantine_file)

    assert summary.total_processed == 3
    assert summary.valid_count == 1
    assert summary.quarantined_count == 2
    assert len(quarantined_list) == 2
    assert valid_df.height == 1

    # Check quarantine log contents
    assert quarantine_file.exists()
    lines = quarantine_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    q_data_1 = json.loads(lines[0])
    assert q_data_1["row_index"] == 1
    assert "flags" in q_data_1["quarantine_reason"]

    q_data_2 = json.loads(lines[1])
    assert q_data_2["row_index"] == 2
    assert "instruction" in q_data_2["quarantine_reason"]
