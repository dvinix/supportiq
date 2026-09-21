"""Pydantic schemas for SupportIQ dataset records and validation."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CustomerSupportRecord(BaseModel):
    """Schema for a single customer support interaction record."""

    flags: str = Field(..., min_length=1, description="Dataset generation flags")
    instruction: str = Field(
        ..., min_length=3, max_length=1000, description="Customer inquiry or instruction text"
    )
    category: str = Field(..., min_length=2, max_length=64, description="High-level category")
    intent: str = Field(..., min_length=2, max_length=64, description="Specific intent")
    response: str = Field(..., min_length=5, max_length=5000, description="Assistant response text")

    @field_validator("instruction", "response", mode="after")
    @classmethod
    def check_non_empty_whitespace(cls, v: str) -> str:
        """Ensure instruction and response are not empty or whitespace-only."""
        cleaned = v.strip()
        if not cleaned:
            msg = "Field cannot be empty or pure whitespace."
            raise ValueError(msg)
        return cleaned

    @field_validator("category", mode="after")
    @classmethod
    def check_category_format(cls, v: str) -> str:
        """Ensure category is non-empty and uppercase."""
        cleaned = v.strip().upper()
        if not cleaned:
            msg = "Category cannot be empty."
            raise ValueError(msg)
        return cleaned

    @field_validator("intent", mode="after")
    @classmethod
    def check_intent_format(cls, v: str) -> str:
        """Ensure intent is non-empty and lowercase."""
        cleaned = v.strip().lower()
        if not cleaned:
            msg = "Intent cannot be empty."
            raise ValueError(msg)
        return cleaned


class QuarantinedRecord(BaseModel):
    """Represents a record that failed validation and was routed to quarantine."""

    row_index: int
    raw_record: dict[str, Any]
    quarantine_reason: str


class ValidationSummary(BaseModel):
    """Statistical summary of dataset validation results."""

    total_processed: int
    valid_count: int
    quarantined_count: int
    quarantine_rate: float
