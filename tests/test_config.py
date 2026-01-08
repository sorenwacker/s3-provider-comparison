"""Tests for the config module."""

import pytest
from pydantic import SecretStr

from s3bench.config import (
    Config,
    ProviderConfig,
    add_provider,
    get_provider,
    load_config,
    remove_provider,
    save_config,
)


class TestProviderConfig:
    """Tests for ProviderConfig model."""

    def test_create_with_endpoint(self):
        """Test creating a provider config with custom endpoint."""
        config = ProviderConfig(
            endpoint_url="https://custom.endpoint.com",
            access_key=SecretStr("access"),
            secret_key=SecretStr("secret"),
            bucket="my-bucket",
        )
        assert config.endpoint_url == "https://custom.endpoint.com"
        assert config.bucket == "my-bucket"
        assert config.access_key.get_secret_value() == "access"

    def test_create_with_region(self):
        """Test creating a provider config with AWS region."""
        config = ProviderConfig(
            region="eu-west-1",
            access_key=SecretStr("access"),
            secret_key=SecretStr("secret"),
            bucket="my-bucket",
        )
        assert config.region == "eu-west-1"
        assert config.endpoint_url is None


class TestConfig:
    """Tests for Config model."""

    def test_empty_config(self):
        """Test creating an empty config."""
        config = Config()
        assert config.providers == {}

    def test_config_with_providers(self):
        """Test creating a config with providers."""
        provider = ProviderConfig(
            endpoint_url="https://example.com",
            access_key=SecretStr("key"),
            secret_key=SecretStr("secret"),
            bucket="bucket",
        )
        config = Config(providers={"test": provider})
        assert "test" in config.providers


class TestConfigIO:
    """Tests for config load/save operations."""

    def test_load_missing_config(self, temp_config_path):
        """Test loading a non-existent config file."""
        config = load_config(temp_config_path)
        assert config.providers == {}

    def test_save_and_load_config(self, temp_config_path):
        """Test saving and loading a config."""
        provider = ProviderConfig(
            endpoint_url="https://example.com",
            access_key=SecretStr("mykey"),
            secret_key=SecretStr("mysecret"),
            bucket="mybucket",
        )
        config = Config(providers={"test-provider": provider})

        save_config(config, temp_config_path)
        loaded = load_config(temp_config_path)

        assert "test-provider" in loaded.providers
        assert loaded.providers["test-provider"].bucket == "mybucket"
        assert loaded.providers["test-provider"].access_key.get_secret_value() == "mykey"

    def test_save_creates_parent_dirs(self, temp_config_dir):
        """Test that save_config creates parent directories."""
        nested_path = temp_config_dir / "nested" / "dir" / "config.yaml"
        config = Config()
        save_config(config, nested_path)
        assert nested_path.exists()


class TestProviderManagement:
    """Tests for provider management functions."""

    def test_add_provider(self, temp_config_path):
        """Test adding a provider."""
        add_provider(
            name="new-provider",
            bucket="new-bucket",
            access_key="key",
            secret_key="secret",
            endpoint_url="https://new.endpoint.com",
            config_path=temp_config_path,
        )

        config = load_config(temp_config_path)
        assert "new-provider" in config.providers
        assert config.providers["new-provider"].bucket == "new-bucket"

    def test_remove_existing_provider(self, temp_config_path):
        """Test removing an existing provider."""
        add_provider(
            name="to-remove",
            bucket="bucket",
            access_key="key",
            secret_key="secret",
            config_path=temp_config_path,
        )

        result = remove_provider("to-remove", temp_config_path)
        assert result is True

        config = load_config(temp_config_path)
        assert "to-remove" not in config.providers

    def test_remove_nonexistent_provider(self, temp_config_path):
        """Test removing a provider that doesn't exist."""
        result = remove_provider("nonexistent", temp_config_path)
        assert result is False

    def test_get_existing_provider(self, temp_config_path):
        """Test getting an existing provider."""
        add_provider(
            name="existing",
            bucket="bucket",
            access_key="key",
            secret_key="secret",
            config_path=temp_config_path,
        )

        provider = get_provider("existing", temp_config_path)
        assert provider is not None
        assert provider.bucket == "bucket"

    def test_get_nonexistent_provider(self, temp_config_path):
        """Test getting a provider that doesn't exist."""
        provider = get_provider("nonexistent", temp_config_path)
        assert provider is None
