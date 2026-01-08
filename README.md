# S3 Provider Benchmark Tool

CLI tool to benchmark S3-compatible storage providers.

## Installation

```bash
uv sync
```

## Usage

### Add a provider

```bash
s3bench provider add mycloud \
  --bucket my-bucket \
  --endpoint https://s3.example.com \
  --access-key ACCESS_KEY \
  --secret-key SECRET_KEY
```

### List providers

```bash
s3bench provider list
```

### Test connection

```bash
s3bench provider test mycloud
```

### Run benchmarks

```bash
# Run against specific providers
s3bench run --provider mycloud --provider aws

# Run against all providers
s3bench run --all

# Customize sizes and iterations
s3bench run --all --sizes small,medium --small-iter 5

# Use SDK only
s3bench run --all --method sdk

# Use rclone only
s3bench run --all --method rclone

# Compare SDK vs rclone (default)
s3bench run --all --method both
```

### Test feature support

Test which S3 API features each provider supports:

```bash
# Test all features on all providers
s3bench features --all

# Test specific providers
s3bench features --provider aws --provider azure

# Test specific features only
s3bench features --all --features presigned_get,multipart,tagging

# Output as JSON
s3bench features --all --json
```

Available features to test:
- `presigned_get` - Presigned GET URLs
- `presigned_put` - Presigned PUT URLs
- `multipart` - Multipart uploads
- `byte_range` - Byte-range GET requests
- `copy` - Server-side object copy
- `head` - HEAD request (metadata without body)
- `metadata` - Custom object metadata
- `tagging` - Object tagging
- `conditional` - Conditional GET (ETag)
- `versioning` - Bucket versioning status
- `lifecycle` - Lifecycle rules

## Configuration

Providers are stored in `~/.config/s3bench/config.yaml`.

## Infrastructure

Terraform configurations are provided to create S3-compatible storage for benchmarking.

### AWS S3

Creates an S3 bucket with IAM user and minimal permissions.

```bash
cd terraform/aws
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars as needed

terraform init
terraform apply

# Add the provider to s3bench
eval "$(terraform output -raw s3bench_add_command)"
```

### Azure Blob Storage

Creates an Azure Storage Account with S3-compatible API access.

```bash
cd terraform/azure
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars as needed

terraform init
terraform apply

# Add the provider to s3bench
eval "$(terraform output -raw s3bench_add_command)"
```

### Cleanup

To destroy the infrastructure after benchmarking:

```bash
# AWS
cd terraform/aws
terraform destroy

# Azure
cd terraform/azure
terraform destroy
```

Both configurations include lifecycle policies that automatically delete benchmark data after 7 days.
