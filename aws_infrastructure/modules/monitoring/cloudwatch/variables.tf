variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "create_carla_simulation_alarms" {
  description = "Whether to create CloudWatch alarms for CARLA simulation instances"
  type        = bool
  default     = true
}

variable "create_eks_cluster_alarms" {
  description = "Whether to create CloudWatch alarms for EKS cluster"
  type        = bool
  default     = true
}

variable "create_s3_bucket_alarms" {
  description = "Whether to create CloudWatch alarms for S3 buckets"
  type        = bool
  default     = true
}

variable "carla_instance_ids" {
  description = "List of CARLA simulation instance IDs"
  type        = list(string)
  default     = []
}

variable "eks_cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = ""
}

variable "s3_bucket_names" {
  description = "List of S3 bucket names to monitor"
  type        = list(string)
  default     = []
}

variable "alarm_actions" {
  description = "List of ARNs to notify when an alarm transitions to ALARM state"
  type        = list(string)
  default     = []
}

variable "ok_actions" {
  description = "List of ARNs to notify when an alarm transitions to OK state"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default     = {}
}
