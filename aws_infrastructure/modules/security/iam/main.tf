# CARLA Simulation Role
resource "aws_iam_role" "carla_simulation" {
  count = var.create_carla_simulation_role ? 1 : 0
  
  name = "${var.project_name}-${var.environment}-carla-simulation-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-carla-simulation-role"
    },
    var.tags
  )
}

resource "aws_iam_policy" "carla_simulation" {
  count = var.create_carla_simulation_role ? 1 : 0
  
  name        = "${var.project_name}-${var.environment}-carla-simulation-policy"
  description = "Policy for CARLA simulation instances"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Effect   = "Allow"
        Resource = [
          var.s3_raw_data_bucket_arn,
          "${var.s3_raw_data_bucket_arn}/*",
          var.s3_logs_bucket_arn,
          "${var.s3_logs_bucket_arn}/*"
        ]
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeTags"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-carla-simulation-policy"
    },
    var.tags
  )
}

resource "aws_iam_role_policy_attachment" "carla_simulation" {
  count = var.create_carla_simulation_role ? 1 : 0
  
  role       = aws_iam_role.carla_simulation[0].name
  policy_arn = aws_iam_policy.carla_simulation[0].arn
}

resource "aws_iam_role_policy_attachment" "carla_simulation_ssm" {
  count = var.create_carla_simulation_role ? 1 : 0
  
  role       = aws_iam_role.carla_simulation[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "carla_simulation_cloudwatch" {
  count = var.create_carla_simulation_role ? 1 : 0
  
  role       = aws_iam_role.carla_simulation[0].name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "carla_simulation" {
  count = var.create_carla_simulation_role ? 1 : 0
  
  name = "${var.project_name}-${var.environment}-carla-simulation-profile"
  role = aws_iam_role.carla_simulation[0].name
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-carla-simulation-profile"
    },
    var.tags
  )
}

# Data Processing Role
resource "aws_iam_role" "data_processing" {
  count = var.create_data_processing_role ? 1 : 0
  
  name = "${var.project_name}-${var.environment}-data-processing-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "ec2.amazonaws.com",
            "ecs-tasks.amazonaws.com"
          ]
        }
      }
    ]
  })
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-data-processing-role"
    },
    var.tags
  )
}

resource "aws_iam_policy" "data_processing" {
  count = var.create_data_processing_role ? 1 : 0
  
  name        = "${var.project_name}-${var.environment}-data-processing-policy"
  description = "Policy for data processing tasks"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Effect   = "Allow"
        Resource = [
          var.s3_raw_data_bucket_arn,
          "${var.s3_raw_data_bucket_arn}/*",
          var.s3_processed_data_bucket_arn,
          "${var.s3_processed_data_bucket_arn}/*",
          var.s3_logs_bucket_arn,
          "${var.s3_logs_bucket_arn}/*"
        ]
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-data-processing-policy"
    },
    var.tags
  )
}

resource "aws_iam_role_policy_attachment" "data_processing" {
  count = var.create_data_processing_role ? 1 : 0
  
  role       = aws_iam_role.data_processing[0].name
  policy_arn = aws_iam_policy.data_processing[0].arn
}

# Model Training Role
resource "aws_iam_role" "model_training" {
  count = var.create_model_training_role ? 1 : 0
  
  name = "${var.project_name}-${var.environment}-model-training-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "ec2.amazonaws.com",
            "ecs-tasks.amazonaws.com",
            "sagemaker.amazonaws.com"
          ]
        }
      }
    ]
  })
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-model-training-role"
    },
    var.tags
  )
}

resource "aws_iam_policy" "model_training" {
  count = var.create_model_training_role ? 1 : 0
  
  name        = "${var.project_name}-${var.environment}-model-training-policy"
  description = "Policy for model training tasks"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Effect   = "Allow"
        Resource = [
          var.s3_processed_data_bucket_arn,
          "${var.s3_processed_data_bucket_arn}/*",
          var.s3_model_artifacts_bucket_arn,
          "${var.s3_model_artifacts_bucket_arn}/*",
          var.s3_logs_bucket_arn,
          "${var.s3_logs_bucket_arn}/*"
        ]
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-model-training-policy"
    },
    var.tags
  )
}

resource "aws_iam_role_policy_attachment" "model_training" {
  count = var.create_model_training_role ? 1 : 0
  
  role       = aws_iam_role.model_training[0].name
  policy_arn = aws_iam_policy.model_training[0].arn
}

# Monitoring Role
resource "aws_iam_role" "monitoring" {
  count = var.create_monitoring_role ? 1 : 0
  
  name = "${var.project_name}-${var.environment}-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "ec2.amazonaws.com",
            "ecs-tasks.amazonaws.com"
          ]
        }
      }
    ]
  })
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-monitoring-role"
    },
    var.tags
  )
}

resource "aws_iam_policy" "monitoring" {
  count = var.create_monitoring_role ? 1 : 0
  
  name        = "${var.project_name}-${var.environment}-monitoring-policy"
  description = "Policy for monitoring tasks"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics",
          "cloudwatch:DescribeAlarms"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents",
          "logs:FilterLogEvents"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Effect   = "Allow"
        Resource = [
          var.s3_logs_bucket_arn,
          "${var.s3_logs_bucket_arn}/*"
        ]
      }
    ]
  })
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-monitoring-policy"
    },
    var.tags
  )
}

resource "aws_iam_role_policy_attachment" "monitoring" {
  count = var.create_monitoring_role ? 1 : 0
  
  role       = aws_iam_role.monitoring[0].name
  policy_arn = aws_iam_policy.monitoring[0].arn
}

resource "aws_iam_role_policy_attachment" "monitoring_cloudwatch" {
  count = var.create_monitoring_role ? 1 : 0
  
  role       = aws_iam_role.monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "monitoring" {
  count = var.create_monitoring_role ? 1 : 0
  
  name = "${var.project_name}-${var.environment}-monitoring-profile"
  role = aws_iam_role.monitoring[0].name
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-monitoring-profile"
    },
    var.tags
  )
}
