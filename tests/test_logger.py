"""Tests for structured JSON logging."""

import json
import logging

from supportiq.core.logger import JSONFormatter, get_logger


def test_json_formatter():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message with %s",
        args=("formatting",),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["name"] == "test_logger"
    assert parsed["message"] == "Test message with formatting"
    assert "timestamp" in parsed
    assert parsed["line"] == 10


def test_json_formatter_with_extra():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.WARNING,
        pathname="test.py",
        lineno=25,
        msg="Warning occurred",
        args=(),
        exc_info=None,
    )
    record.extra = {"model_name": "qwen", "step": 100}
    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "WARNING"
    assert parsed["extra"]["model_name"] == "qwen"
    assert parsed["extra"]["step"] == 100


def test_get_logger_singleton():
    logger1 = get_logger("supportiq_test")
    logger2 = get_logger("supportiq_test")
    assert logger1 is logger2
    assert len(logger1.handlers) == 1
