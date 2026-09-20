"""Structured JSON logger for SupportIQ."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON lines for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if provided
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_record["extra"] = record.extra

        return json.dumps(log_record)


def get_logger(name: str = "supportiq", level: str = "INFO") -> logging.Logger:
    """Get or configure a structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding multiple handlers if logger already initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger
