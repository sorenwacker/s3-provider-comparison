"""Pytest fixtures for s3bench tests."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
import pytest
from moto import mock_aws
from pydantic import SecretStr

from s3bench.config import ProviderConfig


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_path(temp_config_dir):
    """Create a temporary config file path."""
    return temp_config_dir / "config.yaml"


@pytest.fixture
def sample_provider_config():
    """Create a sample provider configuration."""
    return ProviderConfig(
        endpoint_url="http://localhost:5000",
        region="us-east-1",
        access_key=SecretStr("test-access-key"),
        secret_key=SecretStr("test-secret-key"),
        bucket="test-bucket",
    )


@pytest.fixture
def aws_credentials():
    """Set up mock AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def mock_s3(aws_credentials):
    """Create a mocked S3 service."""
    with mock_aws():
        conn = boto3.client("s3", region_name="us-east-1")
        conn.create_bucket(Bucket="test-bucket")
        yield conn


@pytest.fixture
def mock_provider_config():
    """Provider config for mocked S3."""
    return ProviderConfig(
        region="us-east-1",
        access_key=SecretStr("testing"),
        secret_key=SecretStr("testing"),
        bucket="test-bucket",
    )
