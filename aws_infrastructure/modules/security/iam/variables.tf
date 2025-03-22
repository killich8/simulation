variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "create_carla_simulation_role" {
  description = "Whether to create IAM role for CARLA simulation"
  type        = bool
  default     = true
}

variable "create_data_processing_role" {
  description = "Whether to create IAM role for data processing"
  type        = bool
  default     = true
}

variable "create_model_training_role" {
  description = "Whether to create IAM role for model training"
  type        = bool
  default     = true
}

variable "create_monitoring_role" {
  description = "Whether to create IAM role for monitoring"
  type        = bool
  default     = true
}

variable "s3_raw_data_bucket_arn" {
  description = "ARN of the S3 bucket for raw data"
  type        = string
}

variable "s3_processed_data_bucket_arn" {
  description = "ARN of the S3 bucket for processed data"
  type        = string
}

variable "s3_model_artifacts_bucket_arn" {
  description = "ARN of the S3 bucket for model artifacts"
  type        = string
}

variable "s3_logs_bucket_arn" {
  description = "ARN of the S3 bucket for logs"
  type        = string
}

variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default     = {}
}
