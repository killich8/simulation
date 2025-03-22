output "cluster_id" {
  description = "The name of the EKS cluster"
  value       = aws_eks_cluster.this.id
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the EKS cluster"
  value       = aws_eks_cluster.this.arn
}

output "cluster_endpoint" {
  description = "The endpoint for the Kubernetes API server"
  value       = aws_eks_cluster.this.endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.this.certificate_authority[0].data
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_security_group.cluster.id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = aws_iam_role.cluster.arn
}

output "cluster_iam_role_name" {
  description = "IAM role name of the EKS cluster"
  value       = aws_iam_role.cluster.name
}

output "node_groups" {
  description = "Map of EKS node groups"
  value       = aws_eks_node_group.this
}

output "node_iam_role_arn" {
  description = "IAM role ARN for EKS node groups"
  value       = aws_iam_role.node.arn
}

output "node_iam_role_name" {
  description = "IAM role name for EKS node groups"
  value       = aws_iam_role.node.name
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider"
  value       = aws_iam_openid_connect_provider.this.arn
}

output "load_balancer_controller_role_arn" {
  description = "ARN of IAM role for AWS Load Balancer Controller"
  value       = aws_iam_role.load_balancer_controller.arn
}

output "load_balancer_controller_policy_arn" {
  description = "ARN of IAM policy for AWS Load Balancer Controller"
  value       = aws_iam_policy.load_balancer_controller.arn
}
