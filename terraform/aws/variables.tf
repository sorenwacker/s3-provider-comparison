variable "bucket_name_prefix" {
  description = "Prefix for the S3 bucket name (a random suffix will be appended)"
  type        = string
  default     = "s3bench"
}

variable "region" {
  description = "AWS region for the S3 bucket"
  type        = string
  default     = "eu-west-1"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "s3bench"
    ManagedBy = "terraform"
  }
}
