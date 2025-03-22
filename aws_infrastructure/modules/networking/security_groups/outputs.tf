output "carla_simulation_sg_id" {
  description = "ID of the security group for CARLA simulation instances"
  value       = aws_security_group.carla_simulation.id
}

output "eks_cluster_sg_id" {
  description = "ID of the security group for EKS cluster"
  value       = aws_security_group.eks_cluster.id
}

output "eks_workers_sg_id" {
  description = "ID of the security group for EKS worker nodes"
  value       = aws_security_group.eks_workers.id
}

output "bastion_sg_id" {
  description = "ID of the security group for bastion host"
  value       = aws_security_group.bastion.id
}

output "alb_sg_id" {
  description = "ID of the security group for Application Load Balancer"
  value       = aws_security_group.alb.id
}
