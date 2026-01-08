"""Tests for the providers module."""

import pytest
from moto import mock_aws

from s3bench.providers import S3Provider, TimedResult


class TestTimedResult:
    """Tests for TimedResult dataclass."""

    def test_throughput_calculation(self):
        """Test throughput calculation in MB/s."""
        result = TimedResult(duration_seconds=2.0, bytes_transferred=2 * 1024 * 1024)
        assert result.throughput_mbps == 1.0

    def test_throughput_zero_duration(self):
        """Test throughput with zero duration."""
        result = TimedResult(duration_seconds=0.0, bytes_transferred=1024)
        assert result.throughput_mbps == 0.0

    def test_throughput_small_file(self):
        """Test throughput with small file."""
        result = TimedResult(duration_seconds=0.1, bytes_transferred=1024)
        expected = (1024 / (1024 * 1024)) / 0.1
        assert abs(result.throughput_mbps - expected) < 0.0001


class TestS3Provider:
    """Tests for S3Provider class."""

    @mock_aws
    def test_client_lazy_loading(self, mock_provider_config):
        """Test that client is lazily loaded."""
        provider = S3Provider("test", mock_provider_config)
        assert provider._client is None
        _ = provider.client
        assert provider._client is not None

    @mock_aws
    def test_test_connection_success(self, mock_s3, mock_provider_config):
        """Test successful connection test."""
        provider = S3Provider("test", mock_provider_config)
        assert provider.test_connection() is True

    @mock_aws
    def test_test_connection_failure(self, aws_credentials, mock_provider_config):
        """Test failed connection test (bucket doesn't exist)."""
        mock_provider_config.bucket = "nonexistent-bucket"
        provider = S3Provider("test", mock_provider_config)
        assert provider.test_connection() is False

    @mock_aws
    def test_upload_returns_timed_result(self, mock_s3, mock_provider_config):
        """Test upload operation returns timing info."""
        provider = S3Provider("test", mock_provider_config)
        data = b"test data content"

        result = provider.upload("test-key", data)

        assert isinstance(result, TimedResult)
        assert result.bytes_transferred == len(data)
        assert result.duration_seconds > 0

    @mock_aws
    def test_download_returns_timed_result(self, mock_s3, mock_provider_config):
        """Test download operation returns timing info."""
        provider = S3Provider("test", mock_provider_config)
        data = b"test data for download"
        provider.upload("download-test", data)

        result = provider.download("download-test")

        assert isinstance(result, TimedResult)
        assert result.bytes_transferred == len(data)
        assert result.duration_seconds > 0

    @mock_aws
    def test_delete_removes_object(self, mock_s3, mock_provider_config):
        """Test delete operation removes the object."""
        provider = S3Provider("test", mock_provider_config)
        provider.upload("to-delete", b"data")

        provider.delete("to-delete")

        # Verify object is gone by trying to download
        with pytest.raises(Exception):
            provider.download("to-delete")

    @mock_aws
    def test_get_ttfb(self, mock_s3, mock_provider_config):
        """Test TTFB measurement."""
        provider = S3Provider("test", mock_provider_config)
        provider.upload("ttfb-test", b"some data for ttfb test")

        ttfb = provider.get_ttfb("ttfb-test")

        assert ttfb > 0
        assert isinstance(ttfb, float)

    @mock_aws
    def test_bucket_property(self, mock_provider_config):
        """Test bucket property returns correct value."""
        provider = S3Provider("test", mock_provider_config)
        assert provider.bucket == "test-bucket"
