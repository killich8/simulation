provider "aws" {
  region = var.aws_region
}

terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
  backend "s3" {
    # These values must be provided via backend configuration
    # bucket         = "terraform-state-bucket"
    # key            = "road-object-detection/terraform.tfstate"
    # region         = "us-east-1"
    # dynamodb_table = "terraform-locks"
    # encrypt        = true
  }
}

# VPC Module
module "vpc" {
  source = "../modules/networking/vpc"

  vpc_name             = "${var.project_name}-${var.environment}-vpc"
  vpc_cidr             = var.vpc_cidr
  availability_zones   = var.availability_zones
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  enable_nat_gateway   = var.enable_nat_gateway
  single_nat_gateway   = var.single_nat_gateway
  enable_vpn_gateway   = var.enable_vpn_gateway
  
  tags = merge(
    var.tags,
    {
      Environment = var.environment
      Project     = var.project_name
    }
  )
}

# Security Groups Module
module "security_groups" {
  source = "../modules/networking/security_groups"

  vpc_id      = module.vpc.vpc_id
  environment = var.environment
  project_name = var.project_name
  
  tags = merge(
    var.tags,
    {
      Environment = var.environment
      Project     = var.project_name
    }
  )
}

# S3 Storage Module
module "s3_storage" {
  source = "../modules/storage/s3"

  project_name = var.project_name
  environment  = var.environment
  
  raw_data_bucket_name      = "${var.project_name}-${var.environment}-raw-data"
  processed_data_bucket_name = "${var.project_name}-${var.environment}-processed-data"
  model_artifacts_bucket_name = "${var.project_name}-${var.environment}-model-artifacts"
  logs_bucket_name          = "${var.project_name}-${var.environment}-logs"
  
  enable_versioning         = var.enable_s3_versioning
  enable_lifecycle_rules    = var.enable_s3_lifecycle_rules
  
  tags = merge(
    var.tags,
    {
      Environment = var.environment
      Project     = var.project_name
    }
  )
}

# EC2 Compute Module for CARLA Simulation
module "carla_simulation_ec2" {
  source = "../modules/compute/ec2"

  name                        = "${var.project_name}-${var.environment}-carla-simulation"
  instance_count              = var.carla_instance_count
  ami                         = var.carla_ami_id
  instance_type               = var.carla_instance_type
  key_name                    = var.key_name
  monitoring                  = true
  vpc_security_group_ids      = [module.security_groups.carla_simulation_sg_id]
  subnet_ids                  = module.vpc.private_subnet_ids
  associate_public_ip_address = false
  
  root_block_device = [
    {
      volume_type = "gp3"
      volume_size = 100
      encrypted   = true
    }
  ]
  
  ebs_block_device = [
    {
      device_name = "/dev/sdf"
      volume_type = "gp3"
      volume_size = 200
      encrypted   = true
    }
  ]
  
  user_data = file("${path.module}/scripts/carla_setup.sh")
  
  tags = merge(
    var.tags,
    {
      Environment = var.environment
      Project     = var.project_name
      Role        = "CARLA Simulation"
    }
  )
}

# EKS Cluster Module
module "eks_cluster" {
  source = "../modules/compute/eks"

  cluster_name    = "${var.project_name}-${var.environment}-eks"
  cluster_version = var.eks_cluster_version
  
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids
  
  node_groups = {
    main = {
      desired_capacity = var.eks_node_desired_capacity
      max_capacity     = var.eks_node_max_capacity
      min_capacity     = var.eks_node_min_capacity
      instance_types   = var.eks_node_instance_types
      disk_size        = var.eks_node_disk_size
    }
    gpu = {
      desired_capacity = var.eks_gpu_node_desired_capacity
      max_capacity     = var.eks_gpu_node_max_capacity
      min_capacity     = var.eks_gpu_node_min_capacity
      instance_types   = var.eks_gpu_node_instance_types
      disk_size        = var.eks_gpu_node_disk_size
      labels = {
        "accelerator" = "gpu"
      }
      taints = {
        "nvidia.com/gpu" = {
          effect = "NoSchedule"
        }
      }
    }
  }
  
  tags = merge(
    var.tags,
    {
      Environment = var.environment
      Project     = var.project_name
    }
  )
}

# IAM Module
module "iam" {
  source = "../modules/security/iam"

  project_name = var.project_name
  environment  = var.environment
  
  create_carla_simulation_role = true
  create_data_processing_role  = true
  create_model_training_role   = true
  create_monitoring_role       = true
  
  s3_raw_data_bucket_arn       = module.s3_storage.raw_data_bucket_arn
  s3_processed_data_bucket_arn = module.s3_storage.processed_data_bucket_arn
  s3_model_artifacts_bucket_arn = module.s3_storage.model_artifacts_bucket_arn
  s3_logs_bucket_arn           = module.s3_storage.logs_bucket_arn
  
  tags = merge(
    var.tags,
    {
      Environment = var.environment
      Project     = var.project_name
    }
  )
}

# CloudWatch Monitoring Module
module "monitoring" {
  source = "../modules/monitoring/cloudwatch"

  project_name = var.project_name
  environment  = var.environment
  
  create_carla_simulation_alarms = true
  create_eks_cluster_alarms      = true
  create_s3_bucket_alarms        = true
  
  carla_instance_ids             = module.carla_simulation_ec2.instance_ids
  eks_cluster_name               = module.eks_cluster.cluster_name
  s3_bucket_names                = [
    module.s3_storage.raw_data_bucket_id,
    module.s3_storage.processed_data_bucket_id,
    module.s3_storage.model_artifacts_bucket_id,
    module.s3_storage.logs_bucket_id
  ]
  
  alarm_actions                  = var.alarm_actions
  ok_actions                     = var.ok_actions
  
  tags = merge(
    var.tags,
    {
      Environment = var.environment
      Project     = var.project_name
    }
  )
}

# Outputs
output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.vpc.vpc_id
}

output "public_subnet_ids" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnet_ids
}

output "carla_simulation_sg_id" {
  description = "ID of the security group for CARLA simulation instances"
  value       = module.security_groups.carla_simulation_sg_id
}

output "raw_data_bucket_id" {
  description = "The name of the bucket for raw data"
  value       = module.s3_storage.raw_data_bucket_id
}

output "processed_data_bucket_id" {
  description = "The name of the bucket for processed data"
  value       = module.s3_storage.processed_data_bucket_id
}

output "model_artifacts_bucket_id" {
  description = "The name of the bucket for model artifacts"
  value       = module.s3_storage.model_artifacts_bucket_id
}

output "logs_bucket_id" {
  description = "The name of the bucket for logs"
  value       = module.s3_storage.logs_bucket_id
}

output "carla_simulation_instance_ids" {
  description = "List of IDs of CARLA simulation instances"
  value       = module.carla_simulation_ec2.instance_ids
}

output "eks_cluster_id" {
  description = "The name of the EKS cluster"
  value       = module.eks_cluster.cluster_id
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks_cluster.cluster_endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks_cluster.cluster_security_group_id
}

output "eks_node_groups" {
  description = "Map of EKS node groups"
  value       = module.eks_cluster.node_groups
}

output "carla_simulation_role_arn" {
  description = "ARN of the CARLA simulation IAM role"
  value       = module.iam.carla_simulation_role_arn
}

output "data_processing_role_arn" {
  description = "ARN of the data processing IAM role"
  value       = module.iam.data_processing_role_arn
}

output "model_training_role_arn" {
  description = "ARN of the model training IAM role"
  value       = module.iam.model_training_role_arn
}

output "monitoring_role_arn" {
  description = "ARN of the monitoring IAM role"
  value       = module.iam.monitoring_role_arn
}
