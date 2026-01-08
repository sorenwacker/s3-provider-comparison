"""Rclone-based benchmark support."""

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
class RcloneResult:
    """Result of an rclone operation."""
    duration_seconds: float
    bytes_transferred: int

    @property
    def throughput_mbps(self) -> float:
        """Calculate throughput in MB/s."""
        if self.duration_seconds == 0:
            return 0.0
        return (self.bytes_transferred / (1024 * 1024)) / self.duration_seconds


def check_rclone_installed() -> bool:
    """Check if rclone is installed and accessible."""
    return shutil.which("rclone") is not None


def generate_rclone_config(name: str, config: ProviderConfig) -> str:
    """Generate rclone config section for a provider."""
    if config.provider_type == ProviderType.AZURE:
        # Azure Blob Storage config
        return f"""[{name}]
type = azureblob
account = {config.access_key.get_secret_value()}
key = {config.secret_key.get_secret_value()}
"""
    else:
        # S3-compatible config
        lines = [
            f"[{name}]",
            "type = s3",
        ]

        if config.endpoint_url:
            # Custom endpoint - use "Other" provider
            lines.append("provider = Other")
            endpoint = config.endpoint_url.replace("https://", "").replace("http://", "")
            lines.append(f"endpoint = {endpoint}")
        else:
            # Native AWS S3
            lines.append("provider = AWS")

        lines.extend([
            f"access_key_id = {config.access_key.get_secret_value()}",
            f"secret_access_key = {config.secret_key.get_secret_value()}",
            "no_check_bucket = true",  # Don't try to create bucket
        ])

        if config.region:
            lines.append(f"region = {config.region}")

        lines.append("")  # Empty line at end
        return "\n".join(lines)


class RcloneProvider:
    """Rclone-based provider for benchmarking."""

    def __init__(self, name: str, config: ProviderConfig):
        self.name = name
        self.config = config
        self._config_file: Optional[Path] = None
        self._temp_dir: Optional[Path] = None

    def _ensure_config(self) -> Path:
        """Ensure rclone config file exists."""
        if self._config_file is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="s3bench_rclone_"))
            self._config_file = self._temp_dir / "rclone.conf"
            config_content = generate_rclone_config(self.name, self.config)
            self._config_file.write_text(config_content)
        return self._config_file

    @property
    def bucket(self) -> str:
        """Get the bucket/container name."""
        return self.config.bucket

    @property
    def remote_path(self) -> str:
        """Get the rclone remote path."""
        return f"{self.name}:{self.bucket}"

    def _run_rclone(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run an rclone command."""
        config_file = self._ensure_config()
        cmd = ["rclone", "--config", str(config_file)] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if check and result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            raise RuntimeError(f"rclone failed: {error_msg}")
        return result

    def test_connection(self) -> bool:
        """Test if the connection works."""
        try:
            result = self._run_rclone("lsd", self.remote_path, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def upload(self, key: str, data: bytes) -> RcloneResult:
        """Upload data using rclone."""
        self._ensure_config()  # Ensure temp_dir exists

        # Write data to temp file
        temp_file = self._temp_dir / "upload_data"
        temp_file.write_bytes(data)

        remote_dest = f"{self.remote_path}/{key}"

        start = time.perf_counter()
        self._run_rclone("copyto", str(temp_file), remote_dest)
        duration = time.perf_counter() - start

        temp_file.unlink()
        return RcloneResult(duration_seconds=duration, bytes_transferred=len(data))

    def download(self, key: str) -> RcloneResult:
        """Download data using rclone."""
        self._ensure_config()  # Ensure temp_dir exists

        remote_src = f"{self.remote_path}/{key}"
        temp_file = self._temp_dir / "download_data"

        start = time.perf_counter()
        self._run_rclone("copyto", remote_src, str(temp_file))
        duration = time.perf_counter() - start

        size = temp_file.stat().st_size
        temp_file.unlink()
        return RcloneResult(duration_seconds=duration, bytes_transferred=size)

    def delete(self, key: str) -> None:
        """Delete an object using rclone."""
        remote_path = f"{self.remote_path}/{key}"
        self._run_rclone("deletefile", remote_path, check=False)

    def get_ttfb(self, key: str) -> float:
        """Get time to first byte - not directly supported by rclone, use download time."""
        # For rclone, we can't easily measure TTFB, so we return download time for small file
        result = self.download(key)
        return result.duration_seconds

    def list_benchmark_objects(self) -> list[str]:
        """List all objects in the benchmark/ prefix."""
        remote_path = f"{self.remote_path}/benchmark"
        result = self._run_rclone("lsf", remote_path, check=False)
        if result.returncode != 0:
            return []
        return [f"benchmark/{line.strip()}" for line in result.stdout.strip().split("\n") if line.strip()]

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
            self._config_file = None

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


def create_rclone_provider(name: str, config: ProviderConfig) -> RcloneProvider:
    """Create an rclone provider."""
    if not check_rclone_installed():
        raise RuntimeError("rclone is not installed. Install it from https://rclone.org/install/")
    return RcloneProvider(name, config)
