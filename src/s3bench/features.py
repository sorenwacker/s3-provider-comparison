"""Feature detection for S3-compatible storage providers."""

import io
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any
import urllib.request
import urllib.error


class FeatureStatus(str, Enum):
    """Status of a feature test."""
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


@dataclass
class FeatureResult:
    """Result of testing a single feature."""
    feature_name: str
    status: FeatureStatus
    message: Optional[str] = None

    @property
    def status_display(self) -> str:
        """Human-readable status for display."""
        return {
            FeatureStatus.SUPPORTED: "Yes",
            FeatureStatus.NOT_SUPPORTED: "No",
            FeatureStatus.NOT_APPLICABLE: "N/A",
            FeatureStatus.ERROR: "Error",
        }[self.status]


@dataclass
class ProviderFeatureResults:
    """Results of feature tests for a provider."""
    provider_name: str
    results: dict[str, FeatureResult] = field(default_factory=dict)

    def add_result(self, result: FeatureResult) -> None:
        """Add a feature result."""
        self.results[result.feature_name] = result


# Feature test key prefix to avoid conflicts
TEST_PREFIX = "feature-test-"


def _generate_test_key() -> str:
    """Generate a unique test key."""
    return f"{TEST_PREFIX}{uuid.uuid4().hex[:8]}"


def test_presigned_get(provider: Any) -> FeatureResult:
    """Test presigned GET URL generation and usage."""
    feature_name = "Presigned GET"
    test_key = _generate_test_key()
    test_data = b"presigned-get-test-data"

    try:
        # Upload test object
        provider.upload(test_key, test_data)

        # Generate presigned URL
        url = provider.generate_presigned_get_url(test_key, expiry_seconds=300)

        # Try to fetch without credentials
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as response:
            downloaded = response.read()

        # Cleanup
        provider.delete(test_key)

        if downloaded == test_data:
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED)
        else:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, "Data mismatch")

    except NotImplementedError:
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE, "Not implemented for this provider type")
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, str(e))
    except Exception as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_presigned_put(provider: Any) -> FeatureResult:
    """Test presigned PUT URL generation and usage."""
    feature_name = "Presigned PUT"
    test_key = _generate_test_key()
    test_data = b"presigned-put-test-data"

    try:
        # Generate presigned URL for upload
        url = provider.generate_presigned_put_url(test_key, expiry_seconds=300)

        # Try to upload without credentials
        req = urllib.request.Request(url, data=test_data, method="PUT")
        req.add_header("Content-Type", "application/octet-stream")
        with urllib.request.urlopen(req, timeout=30) as response:
            pass  # Just need successful upload

        # Verify by downloading
        result = provider.download(test_key)

        # Cleanup
        provider.delete(test_key)

        if result.bytes_transferred == len(test_data):
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED)
        else:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, "Upload verification failed")

    except NotImplementedError:
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE, "Not implemented for this provider type")
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, str(e))
    except Exception as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_multipart_upload(provider: Any) -> FeatureResult:
    """Test multipart upload capability."""
    feature_name = "Multipart Upload"
    test_key = _generate_test_key()
    # Create data larger than typical part size minimum (5MB for S3)
    test_data = b"x" * (6 * 1024 * 1024)  # 6MB

    try:
        provider.multipart_upload(test_key, test_data, part_size=5 * 1024 * 1024)

        # Verify
        result = provider.download(test_key)
        provider.delete(test_key)

        if result.bytes_transferred == len(test_data):
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED)
        else:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, "Size mismatch")

    except NotImplementedError:
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE)
    except Exception as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        error_str = str(e).lower()
        if "not implemented" in error_str or "not supported" in error_str:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, str(e))
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_byte_range(provider: Any) -> FeatureResult:
    """Test byte-range GET requests."""
    feature_name = "Byte-range GET"
    test_key = _generate_test_key()
    test_data = b"0123456789ABCDEF"  # 16 bytes

    try:
        provider.upload(test_key, test_data)

        # Request bytes 4-7 (should be "4567")
        data = provider.get_byte_range(test_key, start=4, end=7)
        provider.delete(test_key)

        if data == b"4567":
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED)
        else:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, f"Expected b'4567', got {data!r}")

    except NotImplementedError:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE)
    except Exception as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        error_str = str(e).lower()
        if "not implemented" in error_str or "range" in error_str:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, str(e))
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_copy_object(provider: Any) -> FeatureResult:
    """Test server-side object copy."""
    feature_name = "Server-side Copy"
    src_key = _generate_test_key()
    dst_key = _generate_test_key()
    test_data = b"copy-test-data"

    try:
        provider.upload(src_key, test_data)
        provider.copy_object(src_key, dst_key)

        # Verify copy
        result = provider.download(dst_key)

        # Cleanup
        provider.delete(src_key)
        provider.delete(dst_key)

        if result.bytes_transferred == len(test_data):
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED)
        else:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, "Copy verification failed")

    except NotImplementedError:
        try:
            provider.delete(src_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE)
    except Exception as e:
        try:
            provider.delete(src_key)
            provider.delete(dst_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_head_object(provider: Any) -> FeatureResult:
    """Test HEAD request (metadata without body)."""
    feature_name = "HEAD Request"
    test_key = _generate_test_key()
    test_data = b"head-test-data"

    try:
        provider.upload(test_key, test_data)
        metadata = provider.head_object(test_key)
        provider.delete(test_key)

        if "content_length" in metadata and metadata["content_length"] == len(test_data):
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED)
        else:
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, "Basic HEAD works")

    except NotImplementedError:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE)
    except Exception as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_custom_metadata(provider: Any) -> FeatureResult:
    """Test custom metadata on objects."""
    feature_name = "Custom Metadata"
    test_key = _generate_test_key()
    test_data = b"metadata-test"
    # Use alphanumeric keys only (Azure doesn't allow hyphens)
    test_metadata = {"customkey": "customvalue", "anotherkey": "anothervalue"}

    try:
        provider.put_with_metadata(test_key, test_data, test_metadata)
        retrieved = provider.get_metadata(test_key)
        provider.delete(test_key)

        # Check if at least one custom key is present
        if any(k in retrieved for k in test_metadata.keys()):
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED)
        elif any(k.lower().replace("-", "") in str(retrieved).lower() for k in test_metadata.keys()):
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, "Metadata keys normalized")
        else:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, "Metadata not preserved")

    except NotImplementedError:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE)
    except Exception as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_object_tagging(provider: Any) -> FeatureResult:
    """Test object tagging."""
    feature_name = "Object Tagging"
    test_key = _generate_test_key()
    test_data = b"tagging-test"
    test_tags = {"Environment": "test", "Project": "s3bench"}

    try:
        provider.upload(test_key, test_data)
        provider.put_tags(test_key, test_tags)
        retrieved_tags = provider.get_tags(test_key)
        provider.delete(test_key)

        if retrieved_tags == test_tags:
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED)
        elif any(k in retrieved_tags for k in test_tags.keys()):
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, "Partial tag support")
        else:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, "Tags not preserved")

    except NotImplementedError:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE)
    except Exception as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        error_str = str(e).lower()
        if any(x in error_str for x in ["not implemented", "tagging", "featurenotenabled", "not enabled"]):
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, "Feature not enabled")
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_conditional_get(provider: Any) -> FeatureResult:
    """Test conditional GET with ETag."""
    feature_name = "Conditional GET"
    test_key = _generate_test_key()
    test_data = b"conditional-test"

    try:
        provider.upload(test_key, test_data)

        # Get ETag
        metadata = provider.head_object(test_key)
        etag = metadata.get("etag")

        if not etag:
            provider.delete(test_key)
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, "No ETag returned")

        # Test If-None-Match (should return 304 or similar)
        result = provider.conditional_get(test_key, if_none_match=etag)
        provider.delete(test_key)

        if result.get("not_modified", False):
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED)
        else:
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, "ETag available")

    except NotImplementedError:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE)
    except Exception as e:
        try:
            provider.delete(test_key)
        except Exception:
            pass
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_versioning(provider: Any) -> FeatureResult:
    """Test bucket versioning support."""
    feature_name = "Versioning"

    try:
        status = provider.get_versioning()

        if status == "Enabled":
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, "Enabled")
        elif status == "Suspended":
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, "Suspended")
        elif status is None or status == "":
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, "Disabled")
        else:
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, status)

    except NotImplementedError:
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE)
    except Exception as e:
        error_str = str(e).lower()
        if "not implemented" in error_str or "versioning" in error_str:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, str(e))
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


def test_lifecycle(provider: Any) -> FeatureResult:
    """Test lifecycle configuration support."""
    feature_name = "Lifecycle Rules"

    try:
        rules = provider.get_lifecycle()

        if rules is not None:
            count = len(rules) if isinstance(rules, list) else 1
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, f"{count} rule(s)")
        else:
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, "No rules configured")

    except NotImplementedError:
        return FeatureResult(feature_name, FeatureStatus.NOT_APPLICABLE)
    except Exception as e:
        error_str = str(e).lower()
        if "not implemented" in error_str or "lifecycle" in error_str or "no.*configuration" in error_str:
            return FeatureResult(feature_name, FeatureStatus.NOT_SUPPORTED, str(e))
        # NoSuchLifecycleConfiguration means the API is supported but not configured
        if "nosuchlifecycleconfiguration" in error_str.replace(" ", ""):
            return FeatureResult(feature_name, FeatureStatus.SUPPORTED, "Not configured")
        return FeatureResult(feature_name, FeatureStatus.ERROR, str(e))


# All available feature tests
FEATURE_TESTS: dict[str, Callable] = {
    "presigned_get": test_presigned_get,
    "presigned_put": test_presigned_put,
    "multipart": test_multipart_upload,
    "byte_range": test_byte_range,
    "copy": test_copy_object,
    "head": test_head_object,
    "metadata": test_custom_metadata,
    "tagging": test_object_tagging,
    "conditional": test_conditional_get,
    "versioning": test_versioning,
    "lifecycle": test_lifecycle,
}


def run_feature_tests(
    provider: Any,
    features: Optional[list[str]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> ProviderFeatureResults:
    """Run feature tests on a provider.

    Args:
        provider: The provider to test
        features: List of feature keys to test (default: all)
        progress_callback: Optional callback(feature_name, current, total)

    Returns:
        ProviderFeatureResults with all test results
    """
    results = ProviderFeatureResults(provider_name=provider.name)

    tests_to_run = features or list(FEATURE_TESTS.keys())
    total = len(tests_to_run)

    for i, feature_key in enumerate(tests_to_run):
        if feature_key not in FEATURE_TESTS:
            continue

        if progress_callback:
            progress_callback(feature_key, i, total)

        test_func = FEATURE_TESTS[feature_key]
        result = test_func(provider)
        results.add_result(result)

    if progress_callback:
        progress_callback("Done", total, total)

    return results
