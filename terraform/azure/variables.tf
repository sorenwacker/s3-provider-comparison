variable "storage_account_prefix" {
  description = "Prefix for the storage account name (a random suffix will be appended)"
  type        = string
  default     = "s3bench"

  validation {
    condition     = can(regex("^[a-z0-9]{3,15}$", var.storage_account_prefix))
    error_message = "Storage account prefix must be 3-15 lowercase alphanumeric characters."
  }
}

variable "container_name" {
  description = "Name of the blob container (equivalent to S3 bucket)"
  type        = string
  default     = "benchmark"
}

variable "location" {
  description = "Azure region for the storage account"
  type        = string
  default     = "westeurope"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "s3bench"
    ManagedBy = "terraform"
  }
}
