output "dashboard_arn" {
  description = "ARN of the CloudWatch dashboard"
  value       = aws_cloudwatch_dashboard.main.dashboard_arn
}

output "carla_cpu_alarm_arns" {
  description = "List of ARNs of CARLA CPU alarms"
  value       = var.create_carla_simulation_alarms ? aws_cloudwatch_metric_alarm.carla_cpu_high[*].arn : []
}

output "carla_memory_alarm_arns" {
  description = "List of ARNs of CARLA memory alarms"
  value       = var.create_carla_simulation_alarms ? aws_cloudwatch_metric_alarm.carla_memory_high[*].arn : []
}

output "carla_disk_alarm_arns" {
  description = "List of ARNs of CARLA disk alarms"
  value       = var.create_carla_simulation_alarms ? aws_cloudwatch_metric_alarm.carla_disk_high[*].arn : []
}

output "eks_node_count_alarm_arn" {
  description = "ARN of EKS node count alarm"
  value       = var.create_eks_cluster_alarms && var.eks_cluster_name != "" ? aws_cloudwatch_metric_alarm.eks_node_count[0].arn : null
}

output "eks_pod_cpu_alarm_arn" {
  description = "ARN of EKS pod CPU alarm"
  value       = var.create_eks_cluster_alarms && var.eks_cluster_name != "" ? aws_cloudwatch_metric_alarm.eks_pod_cpu_high[0].arn : null
}

output "eks_pod_memory_alarm_arn" {
  description = "ARN of EKS pod memory alarm"
  value       = var.create_eks_cluster_alarms && var.eks_cluster_name != "" ? aws_cloudwatch_metric_alarm.eks_pod_memory_high[0].arn : null
}

output "carla_simulation_log_group_arn" {
  description = "ARN of CARLA simulation log group"
  value       = aws_cloudwatch_log_group.carla_simulation.arn
}

output "eks_cluster_log_group_arn" {
  description = "ARN of EKS cluster log group"
  value       = var.eks_cluster_name != "" ? aws_cloudwatch_log_group.eks_cluster[0].arn : null
}

output "data_processing_log_group_arn" {
  description = "ARN of data processing log group"
  value       = aws_cloudwatch_log_group.data_processing.arn
}

output "model_training_log_group_arn" {
  description = "ARN of model training log group"
  value       = aws_cloudwatch_log_group.model_training.arn
}

output "carla_error_metric_filter_id" {
  description = "ID of CARLA error metric filter"
  value       = aws_cloudwatch_log_metric_filter.carla_error.id
}

output "data_processing_error_metric_filter_id" {
  description = "ID of data processing error metric filter"
  value       = aws_cloudwatch_log_metric_filter.data_processing_error.id
}

output "model_training_error_metric_filter_id" {
  description = "ID of model training error metric filter"
  value       = aws_cloudwatch_log_metric_filter.model_training_error.id
}

output "carla_error_alarm_arn" {
  description = "ARN of CARLA error alarm"
  value       = aws_cloudwatch_metric_alarm.carla_error_high.arn
}

output "data_processing_error_alarm_arn" {
  description = "ARN of data processing error alarm"
  value       = aws_cloudwatch_metric_alarm.data_processing_error_high.arn
}

output "model_training_error_alarm_arn" {
  description = "ARN of model training error alarm"
  value       = aws_cloudwatch_metric_alarm.model_training_error_high.arn
}
