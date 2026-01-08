output "endpoint_url" {
  description = "S3 endpoint URL (empty for AWS, uses default)"
  value       = ""
}

output "region" {
  description = "AWS region"
  value       = var.region
}

output "bucket" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.benchmark.id
}

output "access_key" {
  description = "IAM access key ID for s3bench"
  value       = aws_iam_access_key.s3bench.id
}

output "secret_key" {
  description = "IAM secret access key for s3bench"
  value       = aws_iam_access_key.s3bench.secret
  sensitive   = true
}

output "s3bench_add_command" {
  description = "Command to add this provider to s3bench"
  sensitive   = true
  value       = "s3bench provider add aws --region ${var.region} --access-key ${aws_iam_access_key.s3bench.id} --secret-key '${aws_iam_access_key.s3bench.secret}' --bucket ${aws_s3_bucket.benchmark.id}"
}
