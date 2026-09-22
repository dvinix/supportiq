"""SupportIQ data processing, profiling, validation, and splitting modules."""

from supportiq.data.load import (
    DatasetMetadata,
    compute_file_sha256,
    ingest_raw_dataset,
    load_raw_dataframe,
)
from supportiq.data.normalize import (
    normalize_dataframe,
    normalize_text,
)
from supportiq.data.pii import (
    PIISanitizer,
    get_pii_sanitizer,
    sanitize_text,
)
from supportiq.data.pipeline import (
    assign_cluster_ids,
    audit_splits,
    export_sft_jsonl,
    format_sft_record,
    run_full_pipeline,
    split_grouped_dataset,
)
from supportiq.data.profile import (
    DataProfileReport,
    compute_class_distributions,
    compute_dedup_stats,
    compute_length_stats,
    detect_placeholders,
    run_full_profile,
)
from supportiq.data.validate import (
    validate_dataframe,
    validate_single_record,
)

__all__ = [
    "DataProfileReport",
    "DatasetMetadata",
    "PIISanitizer",
    "assign_cluster_ids",
    "audit_splits",
    "compute_class_distributions",
    "compute_dedup_stats",
    "compute_file_sha256",
    "compute_length_stats",
    "detect_placeholders",
    "export_sft_jsonl",
    "format_sft_record",
    "get_pii_sanitizer",
    "ingest_raw_dataset",
    "load_raw_dataframe",
    "normalize_dataframe",
    "normalize_text",
    "run_full_pipeline",
    "run_full_profile",
    "sanitize_text",
    "split_grouped_dataset",
    "validate_dataframe",
    "validate_single_record",
]
