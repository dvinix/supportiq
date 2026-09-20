"""Configuration management for SupportIQ.

Supports loading YAML files and validating against Pydantic models with
environment variable overrides.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

T = TypeVar("T", bound=BaseModel)


class AppSettings(BaseSettings):
    """Global application settings and environment overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="development", description="Application environment")
    log_level: str = Field(default="INFO", description="Logging level")
    random_seed: int = Field(default=42, description="Global random seed")
    aws_region: str = Field(default="us-east-1", description="AWS Region")
    s3_bucket_name: str = Field(
        default="supportiq-artifacts-dev", description="Artifacts S3 Bucket"
    )
    hf_token: str | None = Field(default=None, description="Hugging Face API Token")
    mlflow_tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data or {}


def load_config[T: BaseModel](model_cls: type[T], file_path: str | Path) -> T:
    """Load a YAML file and validate it with a Pydantic model."""
    data = load_yaml(file_path)
    return model_cls.model_validate(data)
