terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.region
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

locals {
  bucket_name = "${var.bucket_name_prefix}-${random_id.bucket_suffix.hex}"
}

resource "aws_s3_bucket" "benchmark" {
  bucket = local.bucket_name

  tags = var.tags
}

resource "aws_s3_bucket_versioning" "benchmark" {
  bucket = aws_s3_bucket.benchmark.id

  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "benchmark" {
  bucket = aws_s3_bucket.benchmark.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "benchmark" {
  bucket = aws_s3_bucket.benchmark.id

  rule {
    id     = "cleanup-benchmark-data"
    status = "Enabled"

    filter {}

    expiration {
      days = 7
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

resource "aws_s3_bucket_public_access_block" "benchmark" {
  bucket = aws_s3_bucket.benchmark.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_user" "s3bench" {
  name = "${var.bucket_name_prefix}-s3bench-user"

  tags = var.tags
}

resource "aws_iam_access_key" "s3bench" {
  user = aws_iam_user.s3bench.name
}

resource "aws_iam_user_policy" "s3bench" {
  name = "${var.bucket_name_prefix}-s3bench-policy"
  user = aws_iam_user.s3bench.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.benchmark.arn,
          "${aws_s3_bucket.benchmark.arn}/*"
        ]
      }
    ]
  })
}
