"""Critical invariant tests for the SupportIQ data pipeline.

Tests mathematical invariants:
1. Zero cluster leakage across Train, Val, and Test splits.
2. Valid SFT conversational format with round-trip JSON parsing of assistant turn.
"""

import json

import polars as pl

from supportiq.data.pipeline import (
    assign_cluster_ids,
    audit_splits,
    format_sft_record,
    split_grouped_dataset,
)


def test_zero_cluster_leakage_in_grouped_split() -> None:
    """CRITICAL INVARIANT: No near-duplicate cluster may span across Train, Val, and Test splits."""
    # Synthetic dataset with 6 rows belonging to 3 distinct clusters
    df = pl.DataFrame({
        "instruction": [
            "cancel my order now",
            "please cancel my order now",  # Near-duplicate of row 0
            "track my package please",
            "where is my package tracking",  # Near-duplicate of row 2
            "change my delivery address",
            "update my shipping address",  # Near-duplicate of row 4
        ],
        "category": ["ORDER", "ORDER", "TRACKING", "TRACKING", "ACCOUNT", "ACCOUNT"],
        "intent": [
            "cancel_order",
            "cancel_order",
            "track_order",
            "track_order",
            "change_address",
            "change_address",
        ],
        "response": [
            "Order canceled.",
            "Order canceled.",
            "Tracking updated.",
            "Tracking updated.",
            "Address updated.",
            "Address updated.",
        ],
    })

    # 1. Assign cluster IDs
    clustered = assign_cluster_ids(df, threshold=0.70)
    assert "cluster_id" in clustered.columns

    # 2. Split grouped by cluster
    train_df, val_df, test_df = split_grouped_dataset(
        clustered, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42
    )

    # 3. Assert mathematically zero cluster overlap
    train_clusters = set(train_df["cluster_id"].to_list())
    val_clusters = set(val_df["cluster_id"].to_list())
    test_clusters = set(test_df["cluster_id"].to_list())

    assert train_clusters.isdisjoint(val_clusters), "Leakage between train and val!"
    assert train_clusters.isdisjoint(test_clusters), "Leakage between train and test!"
    assert val_clusters.isdisjoint(test_clusters), "Leakage between val and test!"

    # 4. Audit function reports leakage free
    audit = audit_splits(train_df, val_df, test_df)
    assert audit["is_leakage_free"] is True
    assert audit["leakage_overlap"] == 0


def test_sft_chat_format_roundtrip() -> None:
    """CRITICAL INVARIANT: Formatted SFT record conforms to Qwen chat template and valid JSON assistant turn."""
    formatted = format_sft_record(
        instruction="How do I change my billing address?",
        category="ACCOUNT",
        intent="change_shipping_address",
        response="To update your address, navigate to Settings > Profile.",
    )

    # Invariant: 3 messages (system, user, assistant)
    assert "messages" in formatted
    messages = formatted["messages"]
    assert len(messages) == 3
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]

    # Invariant: Assistant message must be strictly valid JSON
    assistant_content = messages[2]["content"]
    parsed = json.loads(assistant_content)

    assert parsed["category"] == "ACCOUNT"
    assert parsed["intent"] == "change_shipping_address"
    assert parsed["response"] == "To update your address, navigate to Settings > Profile."
