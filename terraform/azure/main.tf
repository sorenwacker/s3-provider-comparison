terraform {
  required_version = ">= 1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "random_id" "storage_suffix" {
  byte_length = 4
}

locals {
  # Storage account names must be 3-24 chars, lowercase alphanumeric only
  storage_account_name = "${lower(var.storage_account_prefix)}${random_id.storage_suffix.hex}"
}

resource "azurerm_resource_group" "s3bench" {
  name     = "${var.storage_account_prefix}-rg"
  location = var.location

  tags = var.tags
}

resource "azurerm_storage_account" "s3bench" {
  name                = local.storage_account_name
  resource_group_name = azurerm_resource_group.s3bench.name
  location            = azurerm_resource_group.s3bench.location

  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  access_tier              = "Hot"

  # Required for S3 API compatibility
  shared_access_key_enabled = true

  # Enable hierarchical namespace for better S3 compatibility
  is_hns_enabled = true

  blob_properties {
    delete_retention_policy {
      days = 7
    }
  }

  tags = var.tags
}

resource "azurerm_storage_container" "benchmark" {
  name                  = var.container_name
  storage_account_name  = azurerm_storage_account.s3bench.name
  container_access_type = "private"
}

# Lifecycle management policy to auto-delete old benchmark data
resource "azurerm_storage_management_policy" "cleanup" {
  storage_account_id = azurerm_storage_account.s3bench.id

  rule {
    name    = "cleanup-benchmark-data"
    enabled = true

    filters {
      blob_types = ["blockBlob"]
    }

    actions {
      base_blob {
        delete_after_days_since_modification_greater_than = 7
      }
    }
  }
}
