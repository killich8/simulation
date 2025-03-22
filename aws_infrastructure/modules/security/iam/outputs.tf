output "carla_simulation_role_arn" {
  description = "ARN of the CARLA simulation IAM role"
  value       = var.create_carla_simulation_role ? aws_iam_role.carla_simulation[0].arn : null
}

output "carla_simulation_role_name" {
  description = "Name of the CARLA simulation IAM role"
  value       = var.create_carla_simulation_role ? aws_iam_role.carla_simulation[0].name : null
}

output "carla_simulation_instance_profile_arn" {
  description = "ARN of the CARLA simulation instance profile"
  value       = var.create_carla_simulation_role ? aws_iam_instance_profile.carla_simulation[0].arn : null
}

output "carla_simulation_instance_profile_name" {
  description = "Name of the CARLA simulation instance profile"
  value       = var.create_carla_simulation_role ? aws_iam_instance_profile.carla_simulation[0].name : null
}

output "data_processing_role_arn" {
  description = "ARN of the data processing IAM role"
  value       = var.create_data_processing_role ? aws_iam_role.data_processing[0].arn : null
}

output "data_processing_role_name" {
  description = "Name of the data processing IAM role"
  value       = var.create_data_processing_role ? aws_iam_role.data_processing[0].name : null
}

output "model_training_role_arn" {
  description = "ARN of the model training IAM role"
  value       = var.create_model_training_role ? aws_iam_role.model_training[0].arn : null
}

output "model_training_role_name" {
  description = "Name of the model training IAM role"
  value       = var.create_model_training_role ? aws_iam_role.model_training[0].name : null
}

output "monitoring_role_arn" {
  description = "ARN of the monitoring IAM role"
  value       = var.create_monitoring_role ? aws_iam_role.monitoring[0].arn : null
}

output "monitoring_role_name" {
  description = "Name of the monitoring IAM role"
  value       = var.create_monitoring_role ? aws_iam_role.monitoring[0].name : null
}

output "monitoring_instance_profile_arn" {
  description = "ARN of the monitoring instance profile"
  value       = var.create_monitoring_role ? aws_iam_instance_profile.monitoring[0].arn : null
}

output "monitoring_instance_profile_name" {
  description = "Name of the monitoring instance profile"
  value       = var.create_monitoring_role ? aws_iam_instance_profile.monitoring[0].name : null
}
