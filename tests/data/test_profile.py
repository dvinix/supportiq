"""Unit tests for dataset profiling, statistical summaries, and placeholder detection."""

import polars as pl

from supportiq.data.profile import (
    compute_class_distributions,
    compute_dedup_stats,
    compute_length_stats,
    detect_placeholders,
    generate_markdown_report,
    run_full_profile,
)


def sample_dataframe() -> pl.DataFrame:
    """Provide a small synthetic DataFrame for profiling tests."""
    return pl.DataFrame(
        {
            "flags": ["B", "B", "Q", "B"],
            "instruction": [
                "cancel my order {{Order Number}}",
                "cancel my order {{Order Number}}",  # duplicate instruction
                "check invoice {{Invoice ID}}",
                "help me change address",
            ],
            "category": ["ORDER", "ORDER", "INVOICE", "ACCOUNT"],
            "intent": ["cancel_order", "cancel_order", "get_invoice", "change_address"],
            "response": [
                "Canceling order {{Order Number}} now.",
                "Order {{Order Number}} has been canceled.",
                "Here is your invoice {{Invoice ID}}.",
                "You can change your address in settings.",
            ],
        }
    )


def test_compute_class_distributions() -> None:
    """Verify category and intent distribution calculation."""
    df = sample_dataframe()
    categories, intents = compute_class_distributions(df)

    cat_map = {c.label: c.count for c in categories}
    assert cat_map["ORDER"] == 2
    assert cat_map["INVOICE"] == 1
    assert cat_map["ACCOUNT"] == 1

    intent_map = {i.label: i.count for i in intents}
    assert intent_map["cancel_order"] == 2
    assert intent_map["get_invoice"] == 1


def test_compute_length_stats() -> None:
    """Verify character and word count statistics."""
    df = sample_dataframe()
    stats = compute_length_stats(df)

    assert "instruction_chars" in stats
    assert "instruction_words" in stats
    assert stats["instruction_words"].min >= 4
    assert stats["instruction_words"].max <= 6


def test_compute_dedup_stats() -> None:
    """Verify exact duplicate detection."""
    df = sample_dataframe()
    dedup = compute_dedup_stats(df)

    assert dedup.total_rows == 4
    assert dedup.unique_instructions == 3
    assert dedup.exact_duplicate_instructions == 1
    assert dedup.unique_instruction_response_pairs == 4
    assert dedup.duplicate_pairs == 0


def test_detect_placeholders() -> None:
    """Verify regex detection of slot placeholders."""
    df = sample_dataframe()
    placeholders = detect_placeholders(df)

    assert "Order Number" in placeholders.unique_placeholders
    assert "Invoice ID" in placeholders.unique_placeholders
    assert placeholders.instruction_count == 3
    assert placeholders.response_count == 3


def test_generate_markdown_and_run_profile(tmp_path: pl.DataFrame) -> None:
    """Verify full profiling generation and report persistence."""
    df = sample_dataframe()
    report = run_full_profile(df, output_dir=tmp_path)

    assert (tmp_path / "data_profile.json").exists()
    assert (tmp_path / "data_profile.md").exists()

    md = generate_markdown_report(report)
    assert "# SupportIQ — Data Profile Report" in md
    assert "ORDER" in md
    assert "Order Number" in md
