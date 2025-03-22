variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "raw_data_bucket_name" {
  description = "Name of the S3 bucket for raw data"
  type        = string
}

variable "processed_data_bucket_name" {
  description = "Name of the S3 bucket for processed data"
  type        = string
}

variable "model_artifacts_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  type        = string
}

variable "logs_bucket_name" {
  description = "Name of the S3 bucket for logs"
  type        = string
}

variable "enable_versioning" {
  description = "Enable versioning for S3 buckets"
  type        = bool
  default     = true
}

variable "enable_lifecycle_rules" {
  description = "Enable lifecycle rules for S3 buckets"
  type        = bool
  default     = true
}

variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default     = {}
}
