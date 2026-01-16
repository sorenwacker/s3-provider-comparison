"""
Wasabi Credential Vending Service Example

This demonstrates how to bridge Keycloak authentication to Wasabi S3 access
by using Wasabi's IAM API to manage users and credentials.

Flow:
1. User authenticates with Keycloak (gets JWT)
2. User calls this service with their JWT
3. Service validates JWT and extracts user/group info
4. Service creates/retrieves Wasabi IAM user with appropriate policy
5. Service returns temporary or permanent S3 credentials
"""

import json
import hashlib
from typing import Optional
import boto3
from botocore.config import Config as BotoConfig


class WasabiCredentialVendor:
    """Vends Wasabi S3 credentials based on Keycloak identity."""

    def __init__(
        self,
        wasabi_access_key: str,
        wasabi_secret_key: str,
        bucket_name: str,
    ):
        self.bucket_name = bucket_name

        # Wasabi IAM client (separate endpoint from S3)
        self.iam = boto3.client(
            "iam",
            endpoint_url="https://iam.wasabisys.com",
            aws_access_key_id=wasabi_access_key,
            aws_secret_access_key=wasabi_secret_key,
            region_name="us-east-1",
            config=BotoConfig(signature_version="v4"),
        )

        # Wasabi S3 client (for presigned URLs)
        self.s3 = boto3.client(
            "s3",
            endpoint_url="https://s3.wasabisys.com",
            aws_access_key_id=wasabi_access_key,
            aws_secret_access_key=wasabi_secret_key,
            region_name="us-east-1",
        )

    def _user_name_from_keycloak(self, keycloak_sub: str) -> str:
        """Convert Keycloak subject ID to Wasabi IAM username."""
        # Wasabi usernames have restrictions, so hash the sub
        short_hash = hashlib.sha256(keycloak_sub.encode()).hexdigest()[:12]
        return f"kc-{short_hash}"

    def _create_prefix_policy(self, user_id: str, prefixes: list[str]) -> str:
        """Create IAM policy document for prefix-based access."""
        # Build resource list for allowed prefixes
        resources = []
        for prefix in prefixes:
            resources.append(f"arn:aws:s3:::{self.bucket_name}/{prefix}/*")

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowListBucket",
                    "Effect": "Allow",
                    "Action": ["s3:ListBucket"],
                    "Resource": f"arn:aws:s3:::{self.bucket_name}",
                    "Condition": {
                        "StringLike": {
                            "s3:prefix": [f"{p}/*" for p in prefixes]
                        }
                    },
                },
                {
                    "Sid": "AllowObjectAccess",
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                    ],
                    "Resource": resources,
                },
            ],
        }
        return json.dumps(policy)

    def ensure_user_exists(
        self,
        keycloak_sub: str,
        keycloak_groups: list[str],
    ) -> dict:
        """
        Ensure Wasabi IAM user exists with correct policy.

        Args:
            keycloak_sub: Keycloak subject ID (unique user identifier)
            keycloak_groups: List of Keycloak groups the user belongs to

        Returns:
            dict with username and whether user was created
        """
        username = self._user_name_from_keycloak(keycloak_sub)

        # Check if user exists
        try:
            self.iam.get_user(UserName=username)
            user_exists = True
        except self.iam.exceptions.NoSuchEntityException:
            user_exists = False

        if not user_exists:
            # Create the user
            self.iam.create_user(UserName=username)

            # Map Keycloak groups to S3 prefixes
            # Example: group "project-alpha" -> prefix "projects/alpha/"
            prefixes = []
            for group in keycloak_groups:
                if group.startswith("project-"):
                    project_name = group.replace("project-", "")
                    prefixes.append(f"projects/{project_name}")
                elif group == "admin":
                    prefixes.append("")  # Root access

            # Default: user gets their own prefix
            if not prefixes:
                prefixes = [f"users/{username}"]

            # Create and attach inline policy
            policy_doc = self._create_prefix_policy(username, prefixes)
            self.iam.put_user_policy(
                UserName=username,
                PolicyName=f"{username}-access",
                PolicyDocument=policy_doc,
            )

        return {"username": username, "created": not user_exists}

    def get_or_create_access_key(self, keycloak_sub: str) -> dict:
        """
        Get existing or create new access key for user.

        Note: Wasabi allows max 2 access keys per user.
        In production, you'd want to rotate/manage these.
        """
        username = self._user_name_from_keycloak(keycloak_sub)

        # List existing keys
        response = self.iam.list_access_keys(UserName=username)
        existing_keys = response.get("AccessKeyMetadata", [])

        if existing_keys:
            # Return info about existing key (can't retrieve secret again)
            return {
                "access_key_id": existing_keys[0]["AccessKeyId"],
                "note": "Existing key - secret not retrievable. Delete and recreate if needed.",
            }

        # Create new access key
        response = self.iam.create_access_key(UserName=username)
        key_data = response["AccessKey"]

        return {
            "access_key_id": key_data["AccessKeyId"],
            "secret_access_key": key_data["SecretAccessKey"],
            "endpoint": "https://s3.wasabisys.com",
            "bucket": self.bucket_name,
        }

    def generate_presigned_url(
        self,
        keycloak_sub: str,
        keycloak_groups: list[str],
        object_key: str,
        operation: str = "get_object",
        expiry_seconds: int = 3600,
    ) -> Optional[str]:
        """
        Generate presigned URL if user has access to the object.

        This is an alternative to vending credentials - instead,
        generate short-lived presigned URLs for specific operations.
        """
        # Validate user has access to this prefix
        username = self._user_name_from_keycloak(keycloak_sub)
        allowed_prefixes = []

        for group in keycloak_groups:
            if group.startswith("project-"):
                project_name = group.replace("project-", "")
                allowed_prefixes.append(f"projects/{project_name}/")
            elif group == "admin":
                allowed_prefixes.append("")  # All access

        # Add user's personal prefix
        allowed_prefixes.append(f"users/{username}/")

        # Check if object_key is under an allowed prefix
        has_access = any(
            object_key.startswith(prefix) or prefix == ""
            for prefix in allowed_prefixes
        )

        if not has_access:
            return None

        # Generate presigned URL
        client_method = operation  # get_object or put_object
        url = self.s3.generate_presigned_url(
            ClientMethod=client_method,
            Params={"Bucket": self.bucket_name, "Key": object_key},
            ExpiresIn=expiry_seconds,
        )
        return url

    def delete_user(self, keycloak_sub: str) -> bool:
        """Delete Wasabi IAM user and all associated resources."""
        username = self._user_name_from_keycloak(keycloak_sub)

        try:
            # Delete access keys first
            response = self.iam.list_access_keys(UserName=username)
            for key in response.get("AccessKeyMetadata", []):
                self.iam.delete_access_key(
                    UserName=username,
                    AccessKeyId=key["AccessKeyId"],
                )

            # Delete inline policies
            response = self.iam.list_user_policies(UserName=username)
            for policy_name in response.get("PolicyNames", []):
                self.iam.delete_user_policy(
                    UserName=username,
                    PolicyName=policy_name,
                )

            # Delete user
            self.iam.delete_user(UserName=username)
            return True

        except self.iam.exceptions.NoSuchEntityException:
            return False


# Example FastAPI integration
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

app = FastAPI()
security = HTTPBearer()
vendor = WasabiCredentialVendor(
    wasabi_access_key="YOUR_ADMIN_KEY",
    wasabi_secret_key="YOUR_ADMIN_SECRET",
    bucket_name="your-bucket",
)

KEYCLOAK_PUBLIC_KEY = "..."  # Get from Keycloak

def get_keycloak_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            KEYCLOAK_PUBLIC_KEY,
            algorithms=["RS256"],
            audience="your-client-id",
        )
        return {
            "sub": payload["sub"],
            "groups": payload.get("groups", []),
        }
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/credentials")
def get_credentials(user: dict = Depends(get_keycloak_user)):
    # Ensure user exists in Wasabi
    vendor.ensure_user_exists(user["sub"], user["groups"])
    # Return credentials
    return vendor.get_or_create_access_key(user["sub"])

@app.post("/presign")
def get_presigned_url(
    key: str,
    operation: str = "get_object",
    user: dict = Depends(get_keycloak_user),
):
    url = vendor.generate_presigned_url(
        user["sub"],
        user["groups"],
        key,
        operation,
    )
    if not url:
        raise HTTPException(status_code=403, detail="Access denied")
    return {"url": url}
"""


if __name__ == "__main__":
    # Demo usage
    import os

    vendor = WasabiCredentialVendor(
        wasabi_access_key=os.environ.get("WASABI_ACCESS_KEY", ""),
        wasabi_secret_key=os.environ.get("WASABI_SECRET_KEY", ""),
        bucket_name=os.environ.get("WASABI_BUCKET", "test-bucket"),
    )

    # Simulate a Keycloak user
    keycloak_sub = "user-uuid-12345"
    keycloak_groups = ["project-alpha", "project-beta"]

    # Create user with appropriate policies
    result = vendor.ensure_user_exists(keycloak_sub, keycloak_groups)
    print(f"User: {result}")

    # Get credentials
    creds = vendor.get_or_create_access_key(keycloak_sub)
    print(f"Credentials: {creds}")

    # Generate presigned URL
    url = vendor.generate_presigned_url(
        keycloak_sub,
        keycloak_groups,
        "projects/alpha/data.csv",
        "get_object",
    )
    print(f"Presigned URL: {url}")
