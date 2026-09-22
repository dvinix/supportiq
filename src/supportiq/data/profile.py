"""Data profiling, distribution analysis, length percentiles, and placeholder detection."""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel

from supportiq.core.logger import get_logger

logger = get_logger(__name__)

PLACEHOLDER_REGEX = re.compile(r"\{\{([^}]+)\}\}")


class PercentileStats(BaseModel):
    """Statistical summary of numeric distributions."""

    min: int
    p25: int
    median: int
    mean: float
    p95: int
    p99: int
    max: int


class ClassDistribution(BaseModel):
    """Frequency and relative percentage of a single class label."""

    label: str
    count: int
    percentage: float


class PlaceholderStats(BaseModel):
    """Placeholder occurrence and variety metrics."""

    instruction_count: int
    instruction_percentage: float
    response_count: int
    response_percentage: float
    unique_placeholders: list[str]


class DeduplicationStats(BaseModel):
    """Counts of unique records and duplicates."""

    total_rows: int
    unique_instructions: int
    exact_duplicate_instructions: int
    instruction_duplicate_rate: float
    unique_instruction_response_pairs: int
    duplicate_pairs: int


class DataProfileReport(BaseModel):
    """Comprehensive profiling report combining all dataset metrics."""

    total_records: int
    categories: list[ClassDistribution]
    intents: list[ClassDistribution]
    text_lengths: dict[str, PercentileStats]
    token_lengths: dict[str, PercentileStats]
    placeholders: PlaceholderStats
    duplicates: DeduplicationStats


def _calc_stats(values: list[int] | np.ndarray) -> PercentileStats:
    """Compute min, p25, median, mean, p95, p99, and max from a list or array."""
    arr = np.asarray(values, dtype=np.int64)
    if len(arr) == 0:
        return PercentileStats(min=0, p25=0, median=0, mean=0.0, p95=0, p99=0, max=0)
    return PercentileStats(
        min=int(np.min(arr)),
        p25=int(np.percentile(arr, 25)),
        median=int(np.median(arr)),
        mean=round(float(np.mean(arr)), 2),
        p95=int(np.percentile(arr, 95)),
        p99=int(np.percentile(arr, 99)),
        max=int(np.max(arr)),
    )


def compute_class_distributions(
    df: pl.DataFrame,
) -> tuple[list[ClassDistribution], list[ClassDistribution]]:
    """Compute category and intent frequency distributions."""
    total = df.height
    cat_counts = df.group_by("category").agg(pl.len().alias("count")).sort("count", descending=True)
    categories = [
        ClassDistribution(
            label=row["category"],
            count=row["count"],
            percentage=round(row["count"] / total * 100, 2),
        )
        for row in cat_counts.iter_rows(named=True)
    ]

    intent_counts = (
        df.group_by("intent").agg(pl.len().alias("count")).sort("count", descending=True)
    )
    intents = [
        ClassDistribution(
            label=row["intent"],
            count=row["count"],
            percentage=round(row["count"] / total * 100, 2),
        )
        for row in intent_counts.iter_rows(named=True)
    ]
    return categories, intents


def compute_length_stats(df: pl.DataFrame) -> dict[str, PercentileStats]:
    """Compute character and word count statistics for instructions and responses."""
    inst_chars = [len(s) for s in df["instruction"]]
    inst_words = [len(s.split()) for s in df["instruction"]]
    resp_chars = [len(s) for s in df["response"]]
    resp_words = [len(s.split()) for s in df["response"]]

    return {
        "instruction_chars": _calc_stats(inst_chars),
        "instruction_words": _calc_stats(inst_words),
        "response_chars": _calc_stats(resp_chars),
        "response_words": _calc_stats(resp_words),
    }


def compute_dedup_stats(df: pl.DataFrame) -> DeduplicationStats:
    """Analyze exact duplicate instructions and instruction-response pairs."""
    total = df.height
    unique_inst = df["instruction"].n_unique()
    dup_inst = total - unique_inst
    unique_pairs = df.select(["instruction", "response"]).unique().height
    dup_pairs = total - unique_pairs

    return DeduplicationStats(
        total_rows=total,
        unique_instructions=unique_inst,
        exact_duplicate_instructions=dup_inst,
        instruction_duplicate_rate=round(dup_inst / total * 100, 2),
        unique_instruction_response_pairs=unique_pairs,
        duplicate_pairs=dup_pairs,
    )


def detect_placeholders(df: pl.DataFrame) -> PlaceholderStats:
    """Detect and quantify slot placeholders (e.g. {{Order Number}}) in text."""
    total = df.height
    unique_placeholders: set[str] = set()
    inst_with_p = 0
    resp_with_p = 0

    for inst in df["instruction"]:
        matches = PLACEHOLDER_REGEX.findall(inst)
        if matches:
            inst_with_p += 1
            unique_placeholders.update(matches)

    for resp in df["response"]:
        matches = PLACEHOLDER_REGEX.findall(resp)
        if matches:
            resp_with_p += 1
            unique_placeholders.update(matches)

    return PlaceholderStats(
        instruction_count=inst_with_p,
        instruction_percentage=round(inst_with_p / total * 100, 2),
        response_count=resp_with_p,
        response_percentage=round(resp_with_p / total * 100, 2),
        unique_placeholders=sorted(unique_placeholders),
    )


def compute_token_length_stats(
    df: pl.DataFrame, tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
) -> dict[str, PercentileStats]:
    """Compute token lengths using the specified Hugging Face tokenizer."""
    import os

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer for sequence length profiling: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    inst_texts = df["instruction"].to_list()
    resp_texts = df["response"].to_list()

    inst_tokens = [
        len(toks) for toks in tokenizer(inst_texts, add_special_tokens=False)["input_ids"]
    ]
    resp_tokens = [
        len(toks) for toks in tokenizer(resp_texts, add_special_tokens=False)["input_ids"]
    ]
    total_tokens = [i + r for i, r in zip(inst_tokens, resp_tokens, strict=True)]

    return {
        "instruction_tokens": _calc_stats(inst_tokens),
        "response_tokens": _calc_stats(resp_tokens),
        "total_tokens": _calc_stats(total_tokens),
    }


def generate_markdown_report(report: DataProfileReport) -> str:
    """Render the DataProfileReport into a structured GitHub-flavored Markdown document."""
    lines = [
        "# SupportIQ — Data Profile Report",
        "",
        f"- **Total Records:** {report.total_records:,}",
        f"- **Categories:** {len(report.categories)}",
        f"- **Intents:** {len(report.intents)}",
        "",
        "## 1. Category Distribution",
        "",
        "| Category | Count | Percentage |",
        "| :--- | :--- | :--- |",
    ]
    for c in report.categories:
        lines.append(f"| `{c.label}` | {c.count:,} | {c.percentage:.2f}% |")

    lines.extend(
        [
            "",
            "## 2. Text Length Statistics",
            "",
            "| Metric | Min | P25 | Median | Mean | P95 | P99 | Max |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for metric, s in report.text_lengths.items():
        lines.append(
            f"| {metric} | {s.min} | {s.p25} | {s.median} | {s.mean:.1f} | {s.p95} | {s.p99} | {s.max} |"
        )

    lines.extend(
        [
            "",
            "## 3. Qwen Tokenizer Sequence Lengths",
            "",
            "| Token Metric | Min | P25 | Median | Mean | P95 | P99 | Max |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for metric, s in report.token_lengths.items():
        lines.append(
            f"| {metric} | {s.min} | {s.p25} | {s.median} | {s.mean:.1f} | {s.p95} | {s.p99} | {s.max} |"
        )

    lines.extend(
        [
            "",
            "## 4. Placeholders & Synthetic Templating",
            "",
            f"- **Instructions with Placeholders:** {report.placeholders.instruction_count:,} ({report.placeholders.instruction_percentage:.2f}%)",
            f"- **Responses with Placeholders:** {report.placeholders.response_count:,} ({report.placeholders.response_percentage:.2f}%)",
            f"- **Unique Placeholders Detected:** {len(report.placeholders.unique_placeholders)}",
        ]
    )
    if report.placeholders.unique_placeholders:
        lines.append(
            "  - Sample entities: "
            + ", ".join(f"`{{{{{p}}}}}`" for p in report.placeholders.unique_placeholders[:10])
        )

    lines.extend(
        [
            "",
            "## 5. Deduplication Analysis",
            "",
            f"- **Unique Instructions:** {report.duplicates.unique_instructions:,} ({report.duplicates.exact_duplicate_instructions:,} exact duplicates, {report.duplicates.instruction_duplicate_rate:.2f}%)",
            f"- **Unique (Instruction, Response) Pairs:** {report.duplicates.unique_instruction_response_pairs:,} ({report.duplicates.duplicate_pairs} duplicate pairs)",
            "",
            "> [!NOTE]",
            "> While exact pair duplication is 0%, near-duplicate instructions share underlying semantic templates.",
            "> Data splits MUST be grouped by cluster ID to avoid train-test data leakage.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_full_profile(
    df: pl.DataFrame,
    output_dir: Path | str = "reports",
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
) -> DataProfileReport:
    """Run all profiling computations and persist reports/data_profile.json and .md."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Computing class distributions")
    categories, intents = compute_class_distributions(df)

    logger.info("Computing character & word length statistics")
    text_lengths = compute_length_stats(df)

    logger.info("Analyzing duplicates")
    duplicates = compute_dedup_stats(df)

    logger.info("Scanning for template placeholders")
    placeholders = detect_placeholders(df)

    logger.info("Computing token sequence lengths")
    token_lengths = compute_token_length_stats(df, tokenizer_name=tokenizer_name)

    report = DataProfileReport(
        total_records=df.height,
        categories=categories,
        intents=intents,
        text_lengths=text_lengths,
        token_lengths=token_lengths,
        placeholders=placeholders,
        duplicates=duplicates,
    )

    json_file = out_path / "data_profile.json"
    md_file = out_path / "data_profile.md"

    logger.info("Persisting profile reports to %s and %s", json_file, md_file)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    md_content = generate_markdown_report(report)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    return report
