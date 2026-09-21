# Concept: Pytest in SupportIQ

### 1. What Problem It Solves Here
Automates testing for data pipelines, PII scrubbing, schema validation, and API contracts. Ensures changes in code or data never silently corrupt model inputs or cause runtime crashes in production.

### 2. Core Ideas
- **Test Discovery:** Automatically finds files matching `test_*.py` and functions named `test_*()`.
- **Assertions:** Uses simple, native Python `assert` statements with detailed introspection on failures.
- **Fixtures (`tmp_path`, etc.):** Injects isolated temporary directories and mock objects without polluting the file system.
- **Parametrization (`@pytest.mark.parametrize`):** Tests a function against multiple input/output pairs in a single concise definition.

### 3. Minimal Example
```python
def test_schema_quarantine(tmp_path):
    invalid_record = {"instruction": "", "category": "UNKNOWN"}
    is_valid, reason = validate_record(invalid_record)
    assert not is_valid
    assert "empty instruction" in reason
```

### 4. How to Debug When It Breaks
- Run with verbose and full diffs: `uv run pytest tests/ -vv -s`
- Stop at the first failure: `uv run pytest -x`
- Run only a specific test: `uv run pytest tests/data/test_validate.py -k test_invalid`
