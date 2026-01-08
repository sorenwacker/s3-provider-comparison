"""Benchmark execution and result aggregation."""

import os
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from s3bench.providers import S3Provider


class FileSize(Enum):
    """Predefined file sizes for benchmarking."""

    # Small files
    SMALL_1KB = 1 * 1024
    SMALL_10KB = 10 * 1024
    SMALL_100KB = 100 * 1024
    SMALL_1MB = 1 * 1024 * 1024

    # Medium files
    MEDIUM_10MB = 10 * 1024 * 1024
    MEDIUM_50MB = 50 * 1024 * 1024
    MEDIUM_100MB = 100 * 1024 * 1024

    # Large files
    LARGE_200MB = 200 * 1024 * 1024
    LARGE_500MB = 500 * 1024 * 1024


# Size category definitions
SMALL_SIZES = [FileSize.SMALL_1KB, FileSize.SMALL_10KB, FileSize.SMALL_100KB, FileSize.SMALL_1MB]
MEDIUM_SIZES = [FileSize.MEDIUM_10MB, FileSize.MEDIUM_50MB, FileSize.MEDIUM_100MB]
LARGE_SIZES = [FileSize.LARGE_200MB, FileSize.LARGE_500MB]

# Default iterations per category
DEFAULT_ITERATIONS = {
    "small": 10,
    "medium": 5,
    "large": 2,
}


def get_sizes_for_category(category: str) -> list[FileSize]:
    """Get file sizes for a category."""
    categories = {
        "small": SMALL_SIZES,
        "medium": MEDIUM_SIZES,
        "large": LARGE_SIZES,
    }
    return categories.get(category, [])


def generate_test_data(size: int) -> bytes:
    """Generate random test data of the specified size."""
    return os.urandom(size)


@dataclass
class SizeResult:
    """Results for a single file size."""

    size_bytes: int
    upload_throughputs: list[float] = field(default_factory=list)
    download_throughputs: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    @property
    def size_label(self) -> str:
        """Human-readable size label."""
        if self.size_bytes >= 1024 * 1024 * 1024:
            return f"{self.size_bytes // (1024 * 1024 * 1024)}GB"
        elif self.size_bytes >= 1024 * 1024:
            return f"{self.size_bytes // (1024 * 1024)}MB"
        elif self.size_bytes >= 1024:
            return f"{self.size_bytes // 1024}KB"
        return f"{self.size_bytes}B"

    @property
    def avg_upload_throughput(self) -> float:
        """Average upload throughput in MB/s."""
        return statistics.mean(self.upload_throughputs) if self.upload_throughputs else 0.0

    @property
    def avg_download_throughput(self) -> float:
        """Average download throughput in MB/s."""
        return statistics.mean(self.download_throughputs) if self.download_throughputs else 0.0

    @property
    def avg_latency(self) -> float:
        """Average latency in seconds."""
        return statistics.mean(self.latencies) if self.latencies else 0.0


@dataclass
class ProviderResult:
    """Benchmark results for a single provider."""

    provider_name: str
    size_results: dict[int, SizeResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def add_result(
        self,
        size_bytes: int,
        upload_throughput: float,
        download_throughput: float,
        latency: float,
    ) -> None:
        """Add a benchmark result for a file size."""
        if size_bytes not in self.size_results:
            self.size_results[size_bytes] = SizeResult(size_bytes=size_bytes)
        result = self.size_results[size_bytes]
        result.upload_throughputs.append(upload_throughput)
        result.download_throughputs.append(download_throughput)
        result.latencies.append(latency)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    timestamp: datetime
    provider_results: dict[str, ProviderResult] = field(default_factory=dict)

    def get_provider_result(self, name: str) -> ProviderResult:
        """Get or create a provider result."""
        if name not in self.provider_results:
            self.provider_results[name] = ProviderResult(provider_name=name)
        return self.provider_results[name]


def run_benchmark(
    providers: list[S3Provider],
    categories: list[str],
    iterations: Optional[dict[str, int]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> BenchmarkResult:
    """Run benchmarks against the specified providers.

    Args:
        providers: List of S3Provider instances to benchmark
        categories: List of size categories ("small", "medium", "large")
        iterations: Override for iterations per category
        progress_callback: Optional callback(message, current, total) for progress updates

    Returns:
        BenchmarkResult with all collected metrics
    """
    iterations = iterations or DEFAULT_ITERATIONS
    result = BenchmarkResult(timestamp=datetime.now())

    # Collect all sizes to benchmark
    sizes_to_run: list[tuple[FileSize, int]] = []
    for category in categories:
        sizes = get_sizes_for_category(category)
        iters = iterations.get(category, 1)
        for size in sizes:
            sizes_to_run.append((size, iters))

    total_ops = len(providers) * sum(iters for _, iters in sizes_to_run)
    current_op = 0

    for provider in providers:
        provider_result = result.get_provider_result(provider.name)

        for file_size, num_iterations in sizes_to_run:
            size_bytes = file_size.value

            for i in range(num_iterations):
                current_op += 1
                if progress_callback:
                    progress_callback(
                        f"{provider.name}: {file_size.name} ({i + 1}/{num_iterations})",
                        current_op,
                        total_ops,
                    )

                try:
                    # Generate test data
                    data = generate_test_data(size_bytes)
                    key = f"benchmark/{uuid.uuid4()}"

                    # Upload
                    upload_result = provider.upload(key, data)

                    # Measure TTFB
                    ttfb = provider.get_ttfb(key)

                    # Download
                    download_result = provider.download(key)

                    # Cleanup
                    provider.delete(key)

                    # Record results
                    provider_result.add_result(
                        size_bytes=size_bytes,
                        upload_throughput=upload_result.throughput_mbps,
                        download_throughput=download_result.throughput_mbps,
                        latency=ttfb,
                    )

                except Exception as e:
                    provider_result.errors.append(f"{file_size.name}: {str(e)}")

    return result
