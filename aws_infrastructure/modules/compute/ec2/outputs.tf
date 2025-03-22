output "instance_ids" {
  description = "List of IDs of instances"
  value       = aws_instance.this[*].id
}

output "instance_arns" {
  description = "List of ARNs of instances"
  value       = aws_instance.this[*].arn
}

output "private_ips" {
  description = "List of private IP addresses assigned to the instances"
  value       = aws_instance.this[*].private_ip
}

output "public_ips" {
  description = "List of public IP addresses assigned to the instances, if applicable"
  value       = aws_instance.this[*].public_ip
}

output "instance_state" {
  description = "List of instance states"
  value       = aws_instance.this[*].instance_state
}

output "primary_network_interface_ids" {
  description = "List of IDs of the primary network interface"
  value       = aws_instance.this[*].primary_network_interface_id
}

output "security_groups" {
  description = "List of associated security groups of instances"
  value       = aws_instance.this[*].security_groups
}

output "vpc_security_group_ids" {
  description = "List of associated security groups of instances, if running in non-default VPC"
  value       = aws_instance.this[*].vpc_security_group_ids
}

output "subnet_ids" {
  description = "List of IDs of VPC subnets of instances"
  value       = aws_instance.this[*].subnet_id
}

output "root_block_device_volume_ids" {
  description = "List of volume IDs of root block devices of instances"
  value       = [for instance in aws_instance.this : instance.root_block_device[0].volume_id]
}

output "ebs_block_device_volume_ids" {
  description = "List of volume IDs of EBS block devices of instances"
  value       = [for instance in aws_instance.this : [for bd in instance.ebs_block_device : bd.volume_id]]
}

output "cloudwatch_agent_config_parameter_name" {
  description = "Name of the SSM parameter for CloudWatch agent configuration"
  value       = aws_ssm_parameter.cloudwatch_agent_config.name
}

output "cloudwatch_agent_config_parameter_arn" {
  description = "ARN of the SSM parameter for CloudWatch agent configuration"
  value       = aws_ssm_parameter.cloudwatch_agent_config.arn
}
