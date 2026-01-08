output "endpoint_url" {
  description = "Azure Blob endpoint URL"
  value       = "https://${azurerm_storage_account.s3bench.name}.blob.core.windows.net"
}

output "region" {
  description = "Azure region"
  value       = var.location
}

output "bucket" {
  description = "Container name"
  value       = azurerm_storage_container.benchmark.name
}

output "access_key" {
  description = "Storage account name"
  value       = azurerm_storage_account.s3bench.name
}

output "secret_key" {
  description = "Storage account primary access key"
  value       = azurerm_storage_account.s3bench.primary_access_key
  sensitive   = true
}

output "s3bench_add_command" {
  description = "Command to add this provider to s3bench"
  sensitive   = true
  value       = "s3bench provider add azure --type azure --endpoint https://${azurerm_storage_account.s3bench.name}.blob.core.windows.net --region ${var.location} --access-key ${azurerm_storage_account.s3bench.name} --secret-key '${azurerm_storage_account.s3bench.primary_access_key}' --bucket ${azurerm_storage_container.benchmark.name}"
}
