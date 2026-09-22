"""Text normalization module for cleaning and standardizing customer support dialogues.

Applies Unicode NFC normalization, whitespace collapsing, and trimming while preserving
raw text columns alongside normalized outputs.
"""

import re
import unicodedata

import polars as pl

from supportiq.core.logger import get_logger

logger = get_logger(__name__)

WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Normalize a single string using Unicode NFC and standardized whitespace.

    Args:
        text: Input string to normalize.

    Returns:
        Cleaned string with Unicode NFC normalization, collapsed whitespace, and stripped ends.
    """
    if not text:
        return ""
    # 1. Normalize Unicode to NFC (Canonical Decomposition + Canonical Composition)
    text = unicodedata.normalize("NFC", text)
    # 2. Collapse all tabs, newlines, and consecutive spaces to a single space
    text = WHITESPACE_REGEX.sub(" ", text)
    # 3. Strip leading and trailing whitespace
    return text.strip()


def normalize_dataframe(
    df: pl.DataFrame,
    instruction_col: str = "instruction",
    response_col: str = "response",
    output_instruction_col: str = "instruction_clean",
    output_response_col: str = "response_clean",
) -> pl.DataFrame:
    """Add normalized text columns to a DataFrame while strictly preserving original raw columns.

    Args:
        df: Input Polars DataFrame.
        instruction_col: Column name containing customer input.
        response_col: Column name containing agent response.
        output_instruction_col: New column name for normalized customer input.
        output_response_col: New column name for normalized agent response.

    Returns:
        Polars DataFrame containing both original and normalized columns.
    """
    logger.info("Normalizing %d records (Unicode NFC + whitespace standardization)", len(df))
    return df.with_columns(
        pl.col(instruction_col)
        .map_elements(normalize_text, return_dtype=pl.String)
        .alias(output_instruction_col),
        pl.col(response_col)
        .map_elements(normalize_text, return_dtype=pl.String)
        .alias(output_response_col),
    )
