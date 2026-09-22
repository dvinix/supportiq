"""Tests for training configuration and QLoRA hyperparameter schema."""

from pathlib import Path

from supportiq.training.config import (
    TrainingConfig,
    load_training_config,
)


def test_default_training_config() -> None:
    config = TrainingConfig()
    assert config.model.name == "Qwen/Qwen3-4B-Base"
    assert config.lora.r == 16
    assert config.lora.lora_alpha == 32
    assert config.quantization.load_in_4bit is True
    assert config.training.seed == 42
    assert config.training.max_seq_length == 512


def test_load_training_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
model:
  name: "Qwen/Qwen2.5-0.5B"
lora:
  r: 8
  lora_alpha: 16
training:
  learning_rate: 1.0e-4
  num_train_epochs: 1
"""
    test_yaml = tmp_path / "test_training.yaml"
    test_yaml.write_text(yaml_content, encoding="utf-8")

    cfg = load_training_config(test_yaml)
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B"
    assert cfg.lora.r == 8
    assert cfg.lora.lora_alpha == 16
    assert cfg.training.learning_rate == 1.0e-4
    assert cfg.training.num_train_epochs == 1
    # Check default preserved
    assert cfg.quantization.bnb_4bit_quant_type == "nf4"


def test_load_training_config_fallback_missing(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.yaml"
    cfg = load_training_config(missing_file)
    assert isinstance(cfg, TrainingConfig)
    assert cfg.model.name == "Qwen/Qwen3-4B-Base"
