"""Training configuration schemas and loader for QLoRA SFT fine-tuning."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model identification and precision settings."""

    name: str = "Qwen/Qwen3-4B-Base"
    dev_model_name: str = "Qwen/Qwen2.5-0.5B"
    revision: str = "main"
    torch_dtype: str = "float16"


class LoraParameters(BaseModel):
    """PEFT LoRA adapter hyperparameter settings."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class QuantizationParameters(BaseModel):
    """BitsAndBytes 4-bit NF4 quantization settings."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"


class Hyperparameters(BaseModel):
    """SFTTrainer optimization and checkpoint settings."""

    seed: int = 42
    learning_rate: float = 2.0e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    output_dir: str = "checkpoints"
    max_seq_length: int = 512


class TrainingConfig(BaseModel):
    """Top-level fine-tuning pipeline configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraParameters = Field(default_factory=LoraParameters)
    quantization: QuantizationParameters = Field(default_factory=QuantizationParameters)
    training: Hyperparameters = Field(default_factory=Hyperparameters)


def load_training_config(config_path: Path | str = "configs/training.yaml") -> TrainingConfig:
    """Load and validate training configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        # Fallback to defaults if file missing
        return TrainingConfig()

    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    return TrainingConfig(**data)
