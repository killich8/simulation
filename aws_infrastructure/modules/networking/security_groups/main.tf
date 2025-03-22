resource "aws_security_group" "carla_simulation" {
  name        = "${var.project_name}-${var.environment}-carla-simulation-sg"
  description = "Security group for CARLA simulation instances"
  vpc_id      = var.vpc_id
  
  # SSH access from VPN/Bastion
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
    description = "SSH access from internal network"
  }
  
  # CARLA server ports
  ingress {
    from_port   = 2000
    to_port     = 2002
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
    description = "CARLA server ports"
  }
  
  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-carla-simulation-sg"
    },
    var.tags
  )
}

resource "aws_security_group" "eks_cluster" {
  name        = "${var.project_name}-${var.environment}-eks-cluster-sg"
  description = "Security group for EKS cluster"
  vpc_id      = var.vpc_id
  
  # Allow all internal traffic within the security group
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
    description = "Allow all internal traffic"
  }
  
  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-eks-cluster-sg"
    },
    var.tags
  )
}

resource "aws_security_group" "eks_workers" {
  name        = "${var.project_name}-${var.environment}-eks-workers-sg"
  description = "Security group for EKS worker nodes"
  vpc_id      = var.vpc_id
  
  # Allow all internal traffic within the security group
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
    description = "Allow all internal traffic"
  }
  
  # Allow all traffic from the EKS cluster security group
  ingress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    security_groups = [aws_security_group.eks_cluster.id]
    description     = "Allow all traffic from EKS cluster security group"
  }
  
  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-eks-workers-sg"
    },
    var.tags
  )
}

resource "aws_security_group" "bastion" {
  name        = "${var.project_name}-${var.environment}-bastion-sg"
  description = "Security group for bastion host"
  vpc_id      = var.vpc_id
  
  # SSH access from allowed IPs
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # This should be restricted to specific IPs in production
    description = "SSH access from allowed IPs"
  }
  
  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-bastion-sg"
    },
    var.tags
  )
}

resource "aws_security_group" "alb" {
  name        = "${var.project_name}-${var.environment}-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = var.vpc_id
  
  # HTTP access from anywhere
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP access from anywhere"
  }
  
  # HTTPS access from anywhere
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS access from anywhere"
  }
  
  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-alb-sg"
    },
    var.tags
  )
}
