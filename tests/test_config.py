"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from supportiq.core.config import AppSettings, load_config, load_yaml


class SampleConfig(BaseModel):
    name: str
    threshold: float
    items: list[str]


def test_app_settings_defaults():
    settings = AppSettings()
    assert settings.app_env in ["development", "production", "test"]
    assert settings.random_seed == 42
    assert settings.s3_bucket_name == "supportiq-artifacts-dev"


def test_load_yaml(tmp_path: Path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("name: test_run\nthreshold: 0.85\nitems:\n  - a\n  - b\n")

    data = load_yaml(yaml_file)
    assert data["name"] == "test_run"
    assert data["threshold"] == 0.85
    assert data["items"] == ["a", "b"]


def test_load_config_valid(tmp_path: Path):
    yaml_file = tmp_path / "valid.yaml"
    yaml_file.write_text("name: model_a\nthreshold: 0.95\nitems: ['x', 'y']\n")

    cfg = load_config(SampleConfig, yaml_file)
    assert cfg.name == "model_a"
    assert cfg.threshold == 0.95
    assert cfg.items == ["x", "y"]


def test_load_config_invalid(tmp_path: Path):
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text("name: model_a\nthreshold: not_a_float\nitems: ['x']\n")

    with pytest.raises(ValidationError):
        load_config(SampleConfig, yaml_file)


def test_load_yaml_missing_file():
    with pytest.raises(FileNotFoundError):
        load_yaml("non_existent_file.yaml")
