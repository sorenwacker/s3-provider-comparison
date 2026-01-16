"""Configuration management for storage providers."""

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, SecretStr


class ProviderType(str, Enum):
    """Supported storage provider types."""

    S3 = "s3"
    AZURE = "azure"


class ProviderConfig(BaseModel):
    """Configuration for a single storage provider."""

    provider_type: ProviderType = ProviderType.S3
    endpoint_url: Optional[str] = None
    iam_endpoint: Optional[str] = None
    region: Optional[str] = None
    access_key: SecretStr
    secret_key: SecretStr
    bucket: str


class Config(BaseModel):
    """Root configuration containing all providers."""

    providers: dict[str, ProviderConfig] = {}


def get_config_path() -> Path:
    """Get the default config file path."""
    config_dir = Path.home() / ".config" / "s3bench"
    return config_dir / "config.yaml"


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file."""
    config_path = path or get_config_path()

    if not config_path.exists():
        return Config()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return Config.model_validate(data)


def save_config(config: Config, path: Optional[Path] = None) -> None:
    """Save configuration to YAML file."""
    config_path = path or get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = {"providers": {}}
    for name, provider in config.providers.items():
        provider_data = provider.model_dump()
        provider_data["access_key"] = provider.access_key.get_secret_value()
        provider_data["secret_key"] = provider.secret_key.get_secret_value()
        provider_data["provider_type"] = provider.provider_type.value
        # Remove None values and default s3 provider_type
        provider_data = {
            k: v for k, v in provider_data.items()
            if v is not None and not (k == "provider_type" and v == "s3")
        }
        data["providers"][name] = provider_data

    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def add_provider(
    name: str,
    bucket: str,
    access_key: str,
    secret_key: str,
    endpoint_url: Optional[str] = None,
    iam_endpoint: Optional[str] = None,
    region: Optional[str] = None,
    provider_type: ProviderType = ProviderType.S3,
    config_path: Optional[Path] = None,
) -> None:
    """Add a new provider to the configuration."""
    config = load_config(config_path)
    config.providers[name] = ProviderConfig(
        provider_type=provider_type,
        endpoint_url=endpoint_url,
        iam_endpoint=iam_endpoint,
        region=region,
        access_key=SecretStr(access_key),
        secret_key=SecretStr(secret_key),
        bucket=bucket,
    )
    save_config(config, config_path)


def remove_provider(name: str, config_path: Optional[Path] = None) -> bool:
    """Remove a provider from the configuration. Returns True if removed."""
    config = load_config(config_path)
    if name in config.providers:
        del config.providers[name]
        save_config(config, config_path)
        return True
    return False


def get_provider(name: str, config_path: Optional[Path] = None) -> Optional[ProviderConfig]:
    """Get a specific provider configuration."""
    config = load_config(config_path)
    return config.providers.get(name)
