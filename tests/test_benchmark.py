"""Tests for the benchmark module."""

from datetime import datetime

import pytest
from moto import mock_aws

from s3bench.benchmark import (
    DEFAULT_ITERATIONS,
    LARGE_SIZES,
    MEDIUM_SIZES,
    SMALL_SIZES,
    BenchmarkResult,
    FileSize,
    ProviderResult,
    SizeResult,
    generate_test_data,
    get_sizes_for_category,
    run_benchmark,
)
from s3bench.providers import S3Provider


class TestFileSize:
    """Tests for FileSize enum."""

    def test_small_sizes_values(self):
        """Test small file size values."""
        assert FileSize.SMALL_1KB.value == 1024
        assert FileSize.SMALL_10KB.value == 10 * 1024
        assert FileSize.SMALL_100KB.value == 100 * 1024
        assert FileSize.SMALL_1MB.value == 1024 * 1024

    def test_medium_sizes_values(self):
        """Test medium file size values."""
        assert FileSize.MEDIUM_10MB.value == 10 * 1024 * 1024
        assert FileSize.MEDIUM_50MB.value == 50 * 1024 * 1024
        assert FileSize.MEDIUM_100MB.value == 100 * 1024 * 1024

    def test_large_sizes_values(self):
        """Test large file size values."""
        assert FileSize.LARGE_200MB.value == 200 * 1024 * 1024
        assert FileSize.LARGE_500MB.value == 500 * 1024 * 1024


class TestGetSizesForCategory:
    """Tests for get_sizes_for_category function."""

    def test_small_category(self):
        """Test getting small category sizes."""
        sizes = get_sizes_for_category("small")
        assert sizes == SMALL_SIZES
        assert len(sizes) == 4

    def test_medium_category(self):
        """Test getting medium category sizes."""
        sizes = get_sizes_for_category("medium")
        assert sizes == MEDIUM_SIZES
        assert len(sizes) == 3

    def test_large_category(self):
        """Test getting large category sizes."""
        sizes = get_sizes_for_category("large")
        assert sizes == LARGE_SIZES
        assert len(sizes) == 2

    def test_invalid_category(self):
        """Test getting invalid category returns empty list."""
        sizes = get_sizes_for_category("invalid")
        assert sizes == []


class TestGenerateTestData:
    """Tests for generate_test_data function."""

    def test_generates_correct_size(self):
        """Test data generation produces correct size."""
        data = generate_test_data(1024)
        assert len(data) == 1024

    def test_generates_bytes(self):
        """Test data generation returns bytes."""
        data = generate_test_data(100)
        assert isinstance(data, bytes)

    def test_generates_random_data(self):
        """Test data is random (different each time)."""
        data1 = generate_test_data(100)
        data2 = generate_test_data(100)
        assert data1 != data2


class TestSizeResult:
    """Tests for SizeResult dataclass."""

    def test_size_label_bytes(self):
        """Test size label for bytes."""
        result = SizeResult(size_bytes=500)
        assert result.size_label == "500B"

    def test_size_label_kb(self):
        """Test size label for kilobytes."""
        result = SizeResult(size_bytes=10 * 1024)
        assert result.size_label == "10KB"

    def test_size_label_mb(self):
        """Test size label for megabytes."""
        result = SizeResult(size_bytes=50 * 1024 * 1024)
        assert result.size_label == "50MB"

    def test_size_label_gb(self):
        """Test size label for gigabytes."""
        result = SizeResult(size_bytes=1024 * 1024 * 1024)
        assert result.size_label == "1GB"

    def test_avg_upload_throughput_empty(self):
        """Test average upload throughput with no data."""
        result = SizeResult(size_bytes=1024)
        assert result.avg_upload_throughput == 0.0

    def test_avg_upload_throughput(self):
        """Test average upload throughput calculation."""
        result = SizeResult(size_bytes=1024, upload_throughputs=[10.0, 20.0, 30.0])
        assert result.avg_upload_throughput == 20.0

    def test_avg_download_throughput(self):
        """Test average download throughput calculation."""
        result = SizeResult(size_bytes=1024, download_throughputs=[5.0, 15.0])
        assert result.avg_download_throughput == 10.0

    def test_avg_latency(self):
        """Test average latency calculation."""
        result = SizeResult(size_bytes=1024, latencies=[0.1, 0.2, 0.3])
        assert abs(result.avg_latency - 0.2) < 0.0001


class TestProviderResult:
    """Tests for ProviderResult dataclass."""

    def test_add_result_creates_size_result(self):
        """Test adding result creates SizeResult if needed."""
        result = ProviderResult(provider_name="test")
        result.add_result(
            size_bytes=1024,
            upload_throughput=10.0,
            download_throughput=20.0,
            latency=0.1,
        )

        assert 1024 in result.size_results
        assert result.size_results[1024].upload_throughputs == [10.0]

    def test_add_result_appends_to_existing(self):
        """Test adding result appends to existing SizeResult."""
        result = ProviderResult(provider_name="test")
        result.add_result(1024, 10.0, 20.0, 0.1)
        result.add_result(1024, 15.0, 25.0, 0.15)

        assert len(result.size_results[1024].upload_throughputs) == 2
        assert result.size_results[1024].upload_throughputs == [10.0, 15.0]


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_get_provider_result_creates_new(self):
        """Test getting provider result creates new if needed."""
        result = BenchmarkResult(timestamp=datetime.now())
        provider_result = result.get_provider_result("new-provider")

        assert provider_result.provider_name == "new-provider"
        assert "new-provider" in result.provider_results

    def test_get_provider_result_returns_existing(self):
        """Test getting provider result returns existing."""
        result = BenchmarkResult(timestamp=datetime.now())
        first = result.get_provider_result("test")
        second = result.get_provider_result("test")

        assert first is second


class TestRunBenchmark:
    """Integration tests for run_benchmark function."""

    @mock_aws
    def test_run_benchmark_small_only(self, mock_s3, mock_provider_config):
        """Test running benchmark with small files only."""
        provider = S3Provider("test", mock_provider_config)

        result = run_benchmark(
            providers=[provider],
            categories=["small"],
            iterations={"small": 1, "medium": 1, "large": 1},
        )

        assert "test" in result.provider_results
        provider_result = result.provider_results["test"]
        # Should have results for all 4 small sizes
        assert len(provider_result.size_results) == 4

    @mock_aws
    def test_run_benchmark_progress_callback(self, mock_s3, mock_provider_config):
        """Test progress callback is called."""
        provider = S3Provider("test", mock_provider_config)
        progress_calls = []

        def callback(msg, current, total):
            progress_calls.append((msg, current, total))

        run_benchmark(
            providers=[provider],
            categories=["small"],
            iterations={"small": 1, "medium": 1, "large": 1},
            progress_callback=callback,
        )

        assert len(progress_calls) > 0
        # Last call should have current == total
        assert progress_calls[-1][1] == progress_calls[-1][2]

    @mock_aws
    def test_run_benchmark_multiple_providers(self, mock_s3, mock_provider_config):
        """Test running benchmark with multiple providers."""
        provider1 = S3Provider("provider1", mock_provider_config)
        provider2 = S3Provider("provider2", mock_provider_config)

        result = run_benchmark(
            providers=[provider1, provider2],
            categories=["small"],
            iterations={"small": 1, "medium": 1, "large": 1},
        )

        assert "provider1" in result.provider_results
        assert "provider2" in result.provider_results
