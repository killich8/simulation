variable "aws_region" {
  description = "The AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "road-object-detection"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones to use"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for the public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for the private subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

variable "enable_nat_gateway" {
  description = "Should be true if you want to provision NAT Gateways for each of your private networks"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Should be true if you want to provision a single shared NAT Gateway across all of your private networks"
  type        = bool
  default     = true
}

variable "enable_vpn_gateway" {
  description = "Should be true if you want to create a VPN Gateway"
  type        = bool
  default     = false
}

variable "enable_s3_versioning" {
  description = "Enable versioning for S3 buckets"
  type        = bool
  default     = true
}

variable "enable_s3_lifecycle_rules" {
  description = "Enable lifecycle rules for S3 buckets"
  type        = bool
  default     = true
}

variable "carla_instance_count" {
  description = "Number of EC2 instances for CARLA simulation"
  type        = number
  default     = 2
}

variable "carla_ami_id" {
  description = "AMI ID for CARLA simulation instances"
  type        = string
  default     = "ami-0c55b159cbfafe1f0" # This is a placeholder, use a real Deep Learning AMI
}

variable "carla_instance_type" {
  description = "Instance type for CARLA simulation"
  type        = string
  default     = "g4dn.xlarge"
}

variable "key_name" {
  description = "Key pair name for EC2 instances"
  type        = string
  default     = ""
}

variable "eks_cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.24"
}

variable "eks_node_desired_capacity" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

variable "eks_node_max_capacity" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 5
}

variable "eks_node_min_capacity" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}

variable "eks_node_instance_types" {
  description = "Instance types for EKS worker nodes"
  type        = list(string)
  default     = ["t3.large"]
}

variable "eks_node_disk_size" {
  description = "Disk size for EKS worker nodes in GB"
  type        = number
  default     = 50
}

variable "eks_gpu_node_desired_capacity" {
  description = "Desired number of GPU worker nodes"
  type        = number
  default     = 1
}

variable "eks_gpu_node_max_capacity" {
  description = "Maximum number of GPU worker nodes"
  type        = number
  default     = 3
}

variable "eks_gpu_node_min_capacity" {
  description = "Minimum number of GPU worker nodes"
  type        = number
  default     = 0
}

variable "eks_gpu_node_instance_types" {
  description = "Instance types for EKS GPU worker nodes"
  type        = list(string)
  default     = ["g4dn.xlarge"]
}

variable "eks_gpu_node_disk_size" {
  description = "Disk size for EKS GPU worker nodes in GB"
  type        = number
  default     = 100
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
  default     = {
    Terraform   = "true"
    Application = "road-object-detection"
  }
}
