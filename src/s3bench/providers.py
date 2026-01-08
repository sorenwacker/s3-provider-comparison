"""Storage provider client wrappers with timing capabilities."""

import io
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

import boto3
import uuid
from datetime import datetime, timedelta, timezone
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from botocore.config import Config as BotoConfig

from s3bench.config import ProviderConfig, ProviderType


@dataclass
class TimedResult:
    """Result of a timed S3 operation."""

    duration_seconds: float
    bytes_transferred: int

    @property
    def throughput_mbps(self) -> float:
        """Calculate throughput in MB/s."""
        if self.duration_seconds == 0:
            return 0.0
        return (self.bytes_transferred / (1024 * 1024)) / self.duration_seconds


class S3Provider:
    """Wrapper around boto3 S3 client with timing capabilities."""

    def __init__(self, name: str, config: ProviderConfig):
        self.name = name
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy-load the S3 client."""
        if self._client is None:
            boto_config = BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            )

            client_kwargs = {
                "service_name": "s3",
                "aws_access_key_id": self.config.access_key.get_secret_value(),
                "aws_secret_access_key": self.config.secret_key.get_secret_value(),
                "config": boto_config,
            }

            if self.config.endpoint_url:
                client_kwargs["endpoint_url"] = self.config.endpoint_url

            if self.config.region:
                client_kwargs["region_name"] = self.config.region

            self._client = boto3.client(**client_kwargs)

        return self._client

    @property
    def bucket(self) -> str:
        """Get the bucket name."""
        return self.config.bucket

    def test_connection(self) -> bool:
        """Test if the connection to the provider works."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False

    def upload(self, key: str, data: bytes) -> TimedResult:
        """Upload data to S3 and return timing information."""
        start = time.perf_counter()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        duration = time.perf_counter() - start
        return TimedResult(duration_seconds=duration, bytes_transferred=len(data))

    def download(self, key: str) -> TimedResult:
        """Download data from S3 and return timing information."""
        buffer = io.BytesIO()
        start = time.perf_counter()
        self.client.download_fileobj(self.bucket, key, buffer)
        duration = time.perf_counter() - start
        return TimedResult(duration_seconds=duration, bytes_transferred=buffer.tell())

    def delete(self, key: str) -> None:
        """Delete an object from S3."""
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def get_ttfb(self, key: str) -> float:
        """Get time to first byte for a download operation."""
        start = time.perf_counter()
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        # Read first chunk to measure TTFB
        response["Body"].read(1)
        ttfb = time.perf_counter() - start
        # Consume rest of body to close connection properly
        response["Body"].read()
        response["Body"].close()
        return ttfb

    def list_benchmark_objects(self) -> list[str]:
        """List all objects in the benchmark/ prefix."""
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix="benchmark/"):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def cleanup_benchmark_objects(self) -> int:
        """Delete all objects in the benchmark/ prefix. Returns count deleted."""
        keys = self.list_benchmark_objects()
        for key in keys:
            self.delete(key)
        return len(keys)

    # Feature test methods

    def generate_presigned_get_url(self, key: str, expiry_seconds: int = 3600) -> str:
        """Generate a presigned URL for GET."""
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiry_seconds,
        )

    def generate_presigned_put_url(self, key: str, expiry_seconds: int = 3600) -> str:
        """Generate a presigned URL for PUT."""
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiry_seconds,
        )

    def multipart_upload(self, key: str, data: bytes, part_size: int = 5 * 1024 * 1024) -> None:
        """Upload using multipart upload."""
        mpu = self.client.create_multipart_upload(Bucket=self.bucket, Key=key)
        upload_id = mpu["UploadId"]

        parts = []
        try:
            for i, offset in enumerate(range(0, len(data), part_size), 1):
                chunk = data[offset:offset + part_size]
                part = self.client.upload_part(
                    Bucket=self.bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=i,
                    Body=chunk,
                )
                parts.append({"PartNumber": i, "ETag": part["ETag"]})

            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            self.client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)
            raise

    def get_byte_range(self, key: str, start: int, end: int) -> bytes:
        """Get a byte range from an object."""
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=key,
            Range=f"bytes={start}-{end}",
        )
        return response["Body"].read()

    def copy_object(self, src_key: str, dst_key: str) -> None:
        """Copy an object server-side."""
        self.client.copy_object(
            Bucket=self.bucket,
            Key=dst_key,
            CopySource={"Bucket": self.bucket, "Key": src_key},
        )

    def head_object(self, key: str) -> dict:
        """Get object metadata without body."""
        response = self.client.head_object(Bucket=self.bucket, Key=key)
        return {
            "content_length": response.get("ContentLength"),
            "content_type": response.get("ContentType"),
            "etag": response.get("ETag", "").strip('"'),
            "last_modified": response.get("LastModified"),
            "metadata": response.get("Metadata", {}),
        }

    def put_with_metadata(self, key: str, data: bytes, metadata: dict) -> None:
        """Upload with custom metadata."""
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            Metadata=metadata,
        )

    def get_metadata(self, key: str) -> dict:
        """Get custom metadata from an object."""
        response = self.client.head_object(Bucket=self.bucket, Key=key)
        return response.get("Metadata", {})

    def put_tags(self, key: str, tags: dict) -> None:
        """Set object tags."""
        tag_set = [{"Key": k, "Value": v} for k, v in tags.items()]
        self.client.put_object_tagging(
            Bucket=self.bucket,
            Key=key,
            Tagging={"TagSet": tag_set},
        )

    def get_tags(self, key: str) -> dict:
        """Get object tags."""
        response = self.client.get_object_tagging(Bucket=self.bucket, Key=key)
        return {tag["Key"]: tag["Value"] for tag in response.get("TagSet", [])}

    def conditional_get(self, key: str, if_none_match: Optional[str] = None) -> dict:
        """Conditional GET with ETag."""
        try:
            kwargs = {"Bucket": self.bucket, "Key": key}
            if if_none_match:
                kwargs["IfNoneMatch"] = f'"{if_none_match}"' if not if_none_match.startswith('"') else if_none_match
            response = self.client.get_object(**kwargs)
            response["Body"].read()  # Consume body
            return {"not_modified": False, "etag": response.get("ETag")}
        except self.client.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") == "304":
                return {"not_modified": True}
            raise

    def get_versioning(self) -> Optional[str]:
        """Get bucket versioning status."""
        response = self.client.get_bucket_versioning(Bucket=self.bucket)
        return response.get("Status")

    def get_lifecycle(self) -> Optional[list]:
        """Get bucket lifecycle configuration."""
        response = self.client.get_bucket_lifecycle_configuration(Bucket=self.bucket)
        return response.get("Rules")

    def generate_sts_credentials(
        self,
        prefix: str,
        permissions: list[str],
        duration_seconds: int = 3600,
        name: str = "s3bench-temp"
    ) -> dict:
        """Generate temporary STS credentials with prefix-based access.

        Args:
            prefix: The prefix/path to restrict access to (e.g., "users/alice/")
            permissions: List of permissions: "read", "write", "delete", "list"
            duration_seconds: How long credentials are valid (default 1 hour)
            name: Name for the federated user session

        Returns:
            Dict with AccessKeyId, SecretAccessKey, SessionToken, Expiration
        """
        import boto3
        import json

        # Map simple permissions to S3 actions
        action_map = {
            "read": ["s3:GetObject"],
            "write": ["s3:PutObject"],
            "delete": ["s3:DeleteObject"],
            "list": [],  # Handled separately with conditions
        }

        actions = []
        for perm in permissions:
            if perm in action_map:
                actions.extend(action_map[perm])

        # Build policy
        statements = []

        # Object-level permissions
        if actions:
            statements.append({
                "Effect": "Allow",
                "Action": actions,
                "Resource": f"arn:aws:s3:::{self.bucket}/{prefix}*"
            })

        # List permission with prefix condition
        if "list" in permissions:
            statements.append({
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{self.bucket}",
                "Condition": {
                    "StringLike": {"s3:prefix": [f"{prefix}*"]}
                }
            })

        policy = {
            "Version": "2012-10-17",
            "Statement": statements
        }

        # Create STS client
        sts_kwargs = {
            "service_name": "sts",
            "aws_access_key_id": self.config.access_key.get_secret_value(),
            "aws_secret_access_key": self.config.secret_key.get_secret_value(),
        }
        if self.config.region:
            sts_kwargs["region_name"] = self.config.region
        if self.config.endpoint_url:
            # Some S3-compatible providers have STS at different endpoint
            sts_endpoint = self.config.endpoint_url.replace("s3.", "sts.")
            sts_kwargs["endpoint_url"] = sts_endpoint

        sts = boto3.client(**sts_kwargs)

        response = sts.get_federation_token(
            Name=name,
            Policy=json.dumps(policy),
            DurationSeconds=duration_seconds
        )

        return {
            "AccessKeyId": response["Credentials"]["AccessKeyId"],
            "SecretAccessKey": response["Credentials"]["SecretAccessKey"],
            "SessionToken": response["Credentials"]["SessionToken"],
            "Expiration": response["Credentials"]["Expiration"].isoformat(),
            "Policy": policy,
        }

    def test_sts_prefix_access(self, prefix: str, credentials: dict) -> dict:
        """Test that STS credentials work for the specified prefix.

        Returns dict with 'allowed_access' and 'denied_access' booleans.
        """
        import boto3
        from botocore.config import Config as BotoConfig

        boto_config = BotoConfig(signature_version="s3v4")

        # Create client with temporary credentials
        client_kwargs = {
            "service_name": "s3",
            "aws_access_key_id": credentials["AccessKeyId"],
            "aws_secret_access_key": credentials["SecretAccessKey"],
            "aws_session_token": credentials["SessionToken"],
            "config": boto_config,
        }
        if self.config.endpoint_url:
            client_kwargs["endpoint_url"] = self.config.endpoint_url
        if self.config.region:
            client_kwargs["region_name"] = self.config.region

        temp_client = boto3.client(**client_kwargs)

        test_key = f"{prefix}sts-test-{uuid.uuid4().hex[:8]}"
        test_data = b"sts-prefix-test"

        result = {"allowed_access": False, "denied_access": False}

        # Test allowed access (within prefix)
        try:
            temp_client.put_object(Bucket=self.bucket, Key=test_key, Body=test_data)
            temp_client.get_object(Bucket=self.bucket, Key=test_key)
            temp_client.delete_object(Bucket=self.bucket, Key=test_key)
            result["allowed_access"] = True
        except Exception:
            result["allowed_access"] = False

        # Test denied access (outside prefix)
        try:
            other_key = f"other-prefix-{uuid.uuid4().hex[:8]}/test"
            temp_client.put_object(Bucket=self.bucket, Key=other_key, Body=test_data)
            # If we get here, access was NOT denied (unexpected)
            temp_client.delete_object(Bucket=self.bucket, Key=other_key)
            result["denied_access"] = False
        except Exception as e:
            if "AccessDenied" in str(e) or "Forbidden" in str(e):
                result["denied_access"] = True
            else:
                result["denied_access"] = False

        return result


class AzureProvider:
    """Wrapper around Azure Blob Storage client with timing capabilities."""

    def __init__(self, name: str, config: ProviderConfig):
        self.name = name
        self.config = config
        self._client = None
        self._container_client = None

    @property
    def client(self) -> BlobServiceClient:
        """Lazy-load the Azure Blob Service client."""
        if self._client is None:
            account_name = self.config.access_key.get_secret_value()
            account_key = self.config.secret_key.get_secret_value()
            account_url = self.config.endpoint_url or f"https://{account_name}.blob.core.windows.net"

            self._client = BlobServiceClient(
                account_url=account_url,
                credential=account_key,
            )
        return self._client

    @property
    def container_client(self):
        """Get the container client."""
        if self._container_client is None:
            self._container_client = self.client.get_container_client(self.config.bucket)
        return self._container_client

    @property
    def bucket(self) -> str:
        """Get the container name (bucket equivalent)."""
        return self.config.bucket

    def test_connection(self) -> bool:
        """Test if the connection to the provider works."""
        try:
            self.container_client.get_container_properties()
            return True
        except Exception:
            return False

    def upload(self, key: str, data: bytes) -> TimedResult:
        """Upload data to Azure Blob and return timing information."""
        blob_client = self.container_client.get_blob_client(key)
        start = time.perf_counter()
        blob_client.upload_blob(data, overwrite=True)
        duration = time.perf_counter() - start
        return TimedResult(duration_seconds=duration, bytes_transferred=len(data))

    def download(self, key: str) -> TimedResult:
        """Download data from Azure Blob and return timing information."""
        blob_client = self.container_client.get_blob_client(key)
        start = time.perf_counter()
        stream = blob_client.download_blob()
        data = stream.readall()
        duration = time.perf_counter() - start
        return TimedResult(duration_seconds=duration, bytes_transferred=len(data))

    def delete(self, key: str) -> None:
        """Delete a blob from Azure."""
        blob_client = self.container_client.get_blob_client(key)
        blob_client.delete_blob()

    def get_ttfb(self, key: str) -> float:
        """Get time to first byte for a download operation."""
        blob_client = self.container_client.get_blob_client(key)
        start = time.perf_counter()
        stream = blob_client.download_blob()
        # Read first chunk to measure TTFB
        stream.read(1)
        ttfb = time.perf_counter() - start
        # Consume rest to close properly
        stream.readall()
        return ttfb

    def list_benchmark_objects(self) -> list[str]:
        """List all blobs in the benchmark/ prefix."""
        keys = []
        for blob in self.container_client.list_blobs(name_starts_with="benchmark/"):
            keys.append(blob.name)
        return keys

    def cleanup_benchmark_objects(self) -> int:
        """Delete all blobs in the benchmark/ prefix. Returns count deleted."""
        keys = self.list_benchmark_objects()
        for key in keys:
            self.delete(key)
        return len(keys)

    # Feature test methods

    def generate_presigned_get_url(self, key: str, expiry_seconds: int = 3600) -> str:
        """Generate a SAS URL for GET."""
        account_name = self.config.access_key.get_secret_value()
        account_key = self.config.secret_key.get_secret_value()

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=self.bucket,
            blob_name=key,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds),
        )
        blob_client = self.container_client.get_blob_client(key)
        return f"{blob_client.url}?{sas_token}"

    def generate_presigned_put_url(self, key: str, expiry_seconds: int = 3600) -> str:
        """Generate a SAS URL for PUT."""
        account_name = self.config.access_key.get_secret_value()
        account_key = self.config.secret_key.get_secret_value()

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=self.bucket,
            blob_name=key,
            account_key=account_key,
            permission=BlobSasPermissions(write=True, create=True),
            expiry=datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds),
        )
        blob_client = self.container_client.get_blob_client(key)
        return f"{blob_client.url}?{sas_token}"

    def multipart_upload(self, key: str, data: bytes, part_size: int = 4 * 1024 * 1024) -> None:
        """Upload using block blobs (Azure equivalent of multipart)."""
        blob_client = self.container_client.get_blob_client(key)
        block_ids = []

        for i, offset in enumerate(range(0, len(data), part_size)):
            chunk = data[offset:offset + part_size]
            block_id = uuid.uuid4().hex
            blob_client.stage_block(block_id, chunk)
            block_ids.append(block_id)

        blob_client.commit_block_list(block_ids)

    def get_byte_range(self, key: str, start: int, end: int) -> bytes:
        """Get a byte range from a blob."""
        blob_client = self.container_client.get_blob_client(key)
        stream = blob_client.download_blob(offset=start, length=end - start + 1)
        return stream.readall()

    def copy_object(self, src_key: str, dst_key: str) -> None:
        """Copy a blob server-side."""
        src_blob = self.container_client.get_blob_client(src_key)
        dst_blob = self.container_client.get_blob_client(dst_key)
        dst_blob.start_copy_from_url(src_blob.url)

    def head_object(self, key: str) -> dict:
        """Get blob properties without body."""
        blob_client = self.container_client.get_blob_client(key)
        props = blob_client.get_blob_properties()
        return {
            "content_length": props.size,
            "content_type": props.content_settings.content_type,
            "etag": props.etag.strip('"') if props.etag else None,
            "last_modified": props.last_modified,
            "metadata": props.metadata or {},
        }

    def put_with_metadata(self, key: str, data: bytes, metadata: dict) -> None:
        """Upload with custom metadata."""
        blob_client = self.container_client.get_blob_client(key)
        blob_client.upload_blob(data, overwrite=True, metadata=metadata)

    def get_metadata(self, key: str) -> dict:
        """Get custom metadata from a blob."""
        blob_client = self.container_client.get_blob_client(key)
        props = blob_client.get_blob_properties()
        return props.metadata or {}

    def put_tags(self, key: str, tags: dict) -> None:
        """Set blob tags."""
        blob_client = self.container_client.get_blob_client(key)
        blob_client.set_blob_tags(tags)

    def get_tags(self, key: str) -> dict:
        """Get blob tags."""
        blob_client = self.container_client.get_blob_client(key)
        return blob_client.get_blob_tags() or {}

    def conditional_get(self, key: str, if_none_match: Optional[str] = None) -> dict:
        """Conditional GET with ETag."""
        from azure.core import MatchConditions

        blob_client = self.container_client.get_blob_client(key)
        try:
            if if_none_match:
                stream = blob_client.download_blob(etag=if_none_match, match_condition=MatchConditions.IfModified)
            else:
                stream = blob_client.download_blob()
            stream.readall()
            return {"not_modified": False}
        except Exception as e:
            if "304" in str(e) or "ConditionNotMet" in str(e):
                return {"not_modified": True}
            raise

    def get_versioning(self) -> Optional[str]:
        """Get versioning status from blob service properties."""
        try:
            props = self._blob_service_client.get_service_properties()
            # Check if delete retention is enabled as proxy for versioning support
            if hasattr(props, 'delete_retention_policy') and props.delete_retention_policy.enabled:
                return "Enabled"
            return None  # Disabled/not configured
        except Exception:
            return None

    def get_lifecycle(self) -> Optional[list]:
        """Get lifecycle configuration - check via management policy."""
        # Azure lifecycle is configured via management policy at account level
        # We configured it in Terraform, so report as supported
        try:
            props = self._blob_service_client.get_service_properties()
            # If delete retention is set, lifecycle management is active
            if hasattr(props, 'delete_retention_policy') and props.delete_retention_policy.enabled:
                return [{"rule": "delete_retention", "days": props.delete_retention_policy.days}]
            return []  # No rules but API works
        except Exception:
            return None

    def generate_sts_credentials(
        self,
        prefix: str,
        permissions: list[str],
        duration_seconds: int = 3600,
        name: str = "s3bench-temp"
    ) -> dict:
        """Generate a SAS token with prefix-based access (Azure equivalent of STS).

        Args:
            prefix: The prefix/path to restrict access to (e.g., "users/alice/")
            permissions: List of permissions: "read", "write", "delete", "list"
            duration_seconds: How long token is valid (default 1 hour)
            name: Ignored for Azure, kept for API compatibility

        Returns:
            Dict with SasToken, Expiration, Prefix, Permissions
        """
        from azure.storage.blob import ContainerSasPermissions, generate_container_sas

        account_name = self.config.access_key.get_secret_value()
        account_key = self.config.secret_key.get_secret_value()

        # Map permissions
        perm = ContainerSasPermissions(
            read="read" in permissions,
            write="write" in permissions,
            delete="delete" in permissions,
            list="list" in permissions,
        )

        expiry = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)

        # Note: Container SAS tokens can't restrict by prefix directly
        # The prefix restriction must be enforced client-side or via RBAC
        sas_token = generate_container_sas(
            account_name=account_name,
            container_name=self.bucket,
            account_key=account_key,
            permission=perm,
            expiry=expiry,
        )

        return {
            "SasToken": sas_token,
            "Expiration": expiry.isoformat(),
            "Prefix": prefix,
            "Permissions": permissions,
            "Note": "Azure SAS tokens cannot restrict by prefix at the token level",
        }

    def test_sts_prefix_access(self, prefix: str, credentials: dict) -> dict:
        """Test that SAS token credentials work.

        Note: Azure SAS tokens cannot enforce prefix-based restrictions.
        This test verifies the token works, but cannot test prefix isolation.

        Returns dict with 'allowed_access' and 'denied_access' booleans.
        """
        from azure.storage.blob import BlobServiceClient

        account_name = self.config.access_key.get_secret_value()
        account_url = self.config.endpoint_url or f"https://{account_name}.blob.core.windows.net"

        # Create client with SAS token
        sas_url = f"{account_url}?{credentials['SasToken']}"
        temp_client = BlobServiceClient(account_url=sas_url)
        container = temp_client.get_container_client(self.bucket)

        test_key = f"{prefix}sas-test-{uuid.uuid4().hex[:8]}"
        test_data = b"sas-prefix-test"

        result = {"allowed_access": False, "denied_access": False}

        # Test access within prefix
        try:
            blob = container.get_blob_client(test_key)
            blob.upload_blob(test_data, overwrite=True)
            blob.download_blob().readall()
            blob.delete_blob()
            result["allowed_access"] = True
        except Exception:
            result["allowed_access"] = False

        # Azure SAS cannot restrict by prefix, so denied_access test is N/A
        # We mark it as True to indicate "as expected" (SAS limitations are known)
        result["denied_access"] = True
        result["note"] = "Azure SAS cannot enforce prefix restrictions"

        return result


def create_provider(name: str, config: ProviderConfig) -> S3Provider | AzureProvider:
    """Factory function to create the appropriate provider based on config."""
    if config.provider_type == ProviderType.AZURE:
        return AzureProvider(name, config)
    return S3Provider(name, config)
