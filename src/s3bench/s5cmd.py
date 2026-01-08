"""s5cmd-based benchmark support.

s5cmd is a high-performance S3 CLI tool that uses parallel connections
for significantly faster transfers compared to AWS CLI or rclone.
https://github.com/peak/s5cmd
"""

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from s3bench.config import ProviderConfig, ProviderType


@dataclass
class S5cmdResult:
    """Result of an s5cmd operation."""
    duration_seconds: float
    bytes_transferred: int

    @property
    def throughput_mbps(self) -> float:
        """Calculate throughput in MB/s."""
        if self.duration_seconds == 0:
            return 0.0
        return (self.bytes_transferred / (1024 * 1024)) / self.duration_seconds


def check_s5cmd_installed() -> bool:
    """Check if s5cmd is installed and accessible."""
    return shutil.which("s5cmd") is not None


class S5cmdProvider:
    """s5cmd-based provider for benchmarking."""

    def __init__(self, name: str, config: ProviderConfig):
        self.name = name
        self.config = config
        self._temp_dir: Optional[Path] = None
        self._env: Optional[dict] = None

    def _ensure_temp_dir(self) -> Path:
        """Ensure temp directory exists."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="s3bench_s5cmd_"))
        return self._temp_dir

    def _get_env(self) -> dict:
        """Get environment variables for s5cmd."""
        if self._env is None:
            self._env = os.environ.copy()
            self._env["AWS_ACCESS_KEY_ID"] = self.config.access_key.get_secret_value()
            self._env["AWS_SECRET_ACCESS_KEY"] = self.config.secret_key.get_secret_value()
            if self.config.region:
                self._env["AWS_REGION"] = self.config.region
            # Clear any S3 endpoint overrides from environment
            # s5cmd will use --endpoint-url flag or default AWS endpoint
            self._env.pop("S3_ENDPOINT_URL", None)
            self._env.pop("AWS_ENDPOINT_URL", None)
        return self._env

    @property
    def bucket(self) -> str:
        """Get the bucket name."""
        return self.config.bucket

    @property
    def s3_path(self) -> str:
        """Get the S3 path prefix."""
        return f"s3://{self.bucket}"

    def _run_s5cmd(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run an s5cmd command."""
        cmd = ["s5cmd"]

        # Add endpoint if configured (non-AWS S3-compatible)
        if self.config.endpoint_url:
            cmd.extend(["--endpoint-url", self.config.endpoint_url])

        cmd.extend(args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=self._get_env(),
        )

        if check and result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            raise RuntimeError(f"s5cmd failed: {error_msg}")

        return result

    def test_connection(self) -> bool:
        """Test if the connection works."""
        try:
            result = self._run_s5cmd("ls", self.s3_path, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def upload(self, key: str, data: bytes) -> S5cmdResult:
        """Upload data using s5cmd."""
        temp_dir = self._ensure_temp_dir()

        # Write data to temp file
        temp_file = temp_dir / "upload_data"
        temp_file.write_bytes(data)

        remote_dest = f"{self.s3_path}/{key}"

        start = time.perf_counter()
        self._run_s5cmd("cp", str(temp_file), remote_dest)
        duration = time.perf_counter() - start

        temp_file.unlink()
        return S5cmdResult(duration_seconds=duration, bytes_transferred=len(data))

    def download(self, key: str) -> S5cmdResult:
        """Download data using s5cmd."""
        temp_dir = self._ensure_temp_dir()

        remote_src = f"{self.s3_path}/{key}"
        temp_file = temp_dir / "download_data"

        start = time.perf_counter()
        self._run_s5cmd("cp", remote_src, str(temp_file))
        duration = time.perf_counter() - start

        size = temp_file.stat().st_size
        temp_file.unlink()
        return S5cmdResult(duration_seconds=duration, bytes_transferred=size)

    def delete(self, key: str) -> None:
        """Delete an object using s5cmd."""
        remote_path = f"{self.s3_path}/{key}"
        self._run_s5cmd("rm", remote_path, check=False)

    def get_ttfb(self, key: str) -> float:
        """Get time to first byte - approximated via download time."""
        result = self.download(key)
        return result.duration_seconds

    def list_benchmark_objects(self) -> list[str]:
        """List all objects in the benchmark/ prefix."""
        remote_path = f"{self.s3_path}/benchmark/*"
        result = self._run_s5cmd("ls", remote_path, check=False)

        if result.returncode != 0:
            return []

        keys = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                # s5cmd ls output format: DATE TIME SIZE s3://bucket/key
                parts = line.split()
                if len(parts) >= 4:
                    s3_url = parts[-1]
                    # Extract key from s3://bucket/key
                    if s3_url.startswith(self.s3_path + "/"):
                        key = s3_url[len(self.s3_path) + 1:]
                        keys.append(key)

        return keys

    def cleanup_benchmark_objects(self) -> int:
        """Delete all objects in the benchmark/ prefix."""
        keys = self.list_benchmark_objects()
        for key in keys:
            self.delete(key)
        return len(keys)

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


def create_s5cmd_provider(name: str, config: ProviderConfig) -> S5cmdProvider:
    """Create an s5cmd provider."""
    if config.provider_type == ProviderType.AZURE:
        raise RuntimeError("s5cmd does not support Azure Blob Storage, use rclone instead")
    if not check_s5cmd_installed():
        raise RuntimeError(
            "s5cmd is not installed. Install it from https://github.com/peak/s5cmd\n"
            "  brew install peak/tap/s5cmd  # macOS\n"
            "  go install github.com/peak/s5cmd/v2@latest  # Go"
        )
    return S5cmdProvider(name, config)
