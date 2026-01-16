# S3 Access Mechanisms for User File Access

This document describes different mechanisms for granting users access to specific files in S3-compatible object storage.

## Overview

| Mechanism | Scope | Duration | Provider Support | Use Case |
|-----------|-------|----------|------------------|----------|
| Presigned URLs | Single object | Minutes to hours | Universal | Public sharing, direct browser downloads |
| Bucket Policy | Bucket/prefix | Permanent | Most providers | Service accounts, cross-account access |
| ACL | Object/bucket | Permanent | Most providers | Legacy per-object permissions |
| STS Credentials | Prefix-scoped | Minutes to hours | AWS, some S3-compatible | Multi-tenant apps, credential vending |
| IAM API | User-scoped | Until revoked | AWS, Wasabi | Programmatic user management |

## 1. Presigned URLs

Presigned URLs are time-limited URLs that grant temporary access to a specific object without requiring credentials from the requester.

### How it works

1. Server generates a URL containing a cryptographic signature
2. Signature encodes: object key, expiration time, permissions
3. Anyone with the URL can access the object until expiration
4. No authentication required from the client

### Presigned GET

```python
url = s3_client.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-bucket', 'Key': 'path/to/file.pdf'},
    ExpiresIn=3600  # 1 hour
)
```

### Presigned PUT

```python
url = s3_client.generate_presigned_url(
    'put_object',
    Params={'Bucket': 'my-bucket', 'Key': 'uploads/user-file.pdf'},
    ExpiresIn=3600
)
```

### Characteristics

- **Pros**: No credential distribution, works with any HTTP client, universal provider support
- **Cons**: URL can be shared/leaked, no user identity tracking, single object per URL
- **Typical expiry**: 15 minutes to 7 days (provider-dependent maximum)

## 2. Bucket Policies

JSON-based policies attached to buckets that define access rules based on principals, actions, resources, and conditions.

### How it works

1. Policy is attached to the bucket (not individual objects)
2. Evaluated on every request to the bucket
3. Can grant/deny based on: IAM user, IP address, prefix, request headers
4. Supports complex conditions (time-based, referer, VPC endpoint)

### Example: Prefix-based access

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::123456789:user/app-user"},
    "Action": ["s3:GetObject", "s3:PutObject"],
    "Resource": "arn:aws:s3:::my-bucket/users/user-123/*"
  }]
}
```

### Characteristics

- **Pros**: Fine-grained control, prefix scoping, condition-based rules
- **Cons**: Requires IAM principal, policy size limits, no per-request expiry
- **Use case**: Service-to-service access, multi-tenant prefix isolation

## 3. Access Control Lists (ACL)

Legacy mechanism for granting permissions to predefined groups or specific accounts.

### How it works

1. Each object/bucket has an ACL
2. ACL contains grants to: owner, authenticated users, all users, specific accounts
3. Predefined "canned ACLs" for common patterns

### Canned ACLs

| ACL | Effect |
|-----|--------|
| `private` | Owner-only access (default) |
| `public-read` | Anyone can read |
| `public-read-write` | Anyone can read/write |
| `authenticated-read` | Any authenticated AWS user can read |

### Characteristics

- **Pros**: Simple, per-object granularity
- **Cons**: Limited expressiveness, no prefix support, deprecated by AWS
- **Status**: AWS recommends bucket policies instead; ACLs disabled by default on new buckets

## 4. STS Temporary Credentials

Security Token Service (STS) issues short-lived credentials scoped to specific permissions.

### How it works

1. Application calls STS with a policy defining allowed actions
2. STS returns temporary access key, secret key, and session token
3. Client uses these credentials for S3 API calls
4. Credentials expire automatically

### GetFederationToken (prefix-scoped)

```python
sts_client = boto3.client('sts')
response = sts_client.get_federation_token(
    Name='user-123',
    Policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject"],
            "Resource": "arn:aws:s3:::my-bucket/users/user-123/*"
        }]
    }),
    DurationSeconds=3600
)
credentials = response['Credentials']
# Returns: AccessKeyId, SecretAccessKey, SessionToken, Expiration
```

### AssumeRole

```python
response = sts_client.assume_role(
    RoleArn='arn:aws:iam::123456789:role/S3AccessRole',
    RoleSessionName='user-session',
    DurationSeconds=3600
)
```

### Characteristics

- **Pros**: Prefix isolation, automatic expiry, full S3 API access, audit trail
- **Cons**: Limited provider support (AWS, some S3-compatible), requires STS endpoint
- **Provider support**: AWS (full), Wasabi (no), Exoscale (no), MinIO (yes)

## 5. IAM API Credential Vending

For providers without STS, a credential vending service can create/manage IAM users programmatically.

### How it works

1. User authenticates with identity provider (e.g., Keycloak, OAuth)
2. Credential vending service validates identity
3. Service creates/updates IAM user via provider's IAM API
4. Service attaches policy restricting access to user's prefix
5. Service returns IAM credentials to user

### Architecture

```
User -> Identity Provider (JWT) -> Credential Vending API -> Provider IAM API
                                          |
                                          v
                                   IAM User + Policy
                                          |
                                          v
                                   Access Key + Secret
```

### Provider-specific IAM APIs

| Provider | IAM API Type | Endpoint |
|----------|--------------|----------|
| AWS | AWS-compatible | `iam.amazonaws.com` |
| Wasabi | AWS-compatible | `iam.wasabisys.com` |
| Exoscale | Proprietary REST | `api-{zone}.exoscale.com/v2` |
| MinIO | AWS-compatible | Same as S3 endpoint |

### Characteristics

- **Pros**: Works without STS, full credential lifecycle control
- **Cons**: Requires admin credentials, credentials are long-lived, more complex
- **Use case**: Multi-tenant applications on providers without STS

## Comparison for Multi-Tenant Applications

| Requirement | Presigned URLs | Bucket Policy | STS | IAM Vending |
|-------------|----------------|---------------|-----|-------------|
| Per-user prefix isolation | No | Yes | Yes | Yes |
| No credential distribution | Yes | No | No | No |
| Automatic expiry | Yes | No | Yes | No |
| Full S3 API access | No | Yes | Yes | Yes |
| Works on all providers | Yes | Mostly | No | Varies |
| Audit trail per user | No | Yes | Yes | Yes |

## Feature Matrix Interpretation

The `s3bench features` command tests for AWS-compatible API support:

- **Yes** = AWS-compatible API supported (works with boto3/AWS SDK)
- **No** = AWS-compatible API not supported (vendor-specific alternative may exist)

### Vendor-Specific Alternatives

| Provider | Feature | AWS-compatible | Vendor Alternative |
|----------|---------|----------------|-------------------|
| AWS | IAM API | Yes | - |
| AWS | STS | Yes | - |
| Wasabi | IAM API | Yes | - |
| Wasabi | STS | No | IAM API credential vending |
| Azure | IAM API | No | RBAC / Entra ID |
| Azure | STS | No | SAS tokens |
| Azure | Bucket Policy | No | RBAC / SAS |
| Azure | ACL | No | RBAC |
| Exoscale | IAM API | No | Exoscale API v2 |
| Exoscale | STS | No | Exoscale API v2 |
| Impossible Cloud | IAM API | Yes | - |
| Intercolo | IAM API | No | Unknown |
| Intercolo | STS | No | Unknown |
| MinIO | IAM API | Yes | - |
| MinIO | STS | Yes | - |

When a feature shows "No", check if the provider offers a proprietary API that achieves the same goal. The implementation will differ, but the capability may still exist.

## Recommendations

1. **Simple file sharing**: Use presigned URLs
2. **Service-to-service**: Use bucket policies with IAM principals
3. **Multi-tenant with AWS**: Use STS with prefix-scoped policies
4. **Multi-tenant with Wasabi**: Use IAM API credential vending
5. **Multi-tenant with Exoscale**: Use native Exoscale API or presigned URLs
6. **Multi-tenant with Azure**: Use SAS tokens with Entra ID integration
