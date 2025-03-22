# CloudWatch Dashboard for Road Object Detection Project
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-${var.environment}-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 1
        properties = {
          markdown = "# ${var.project_name} - ${var.environment} Environment Monitoring Dashboard"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 1
        width  = 12
        height = 6
        properties = {
          view    = "timeSeries"
          stacked = false
          metrics = [
            [ "AWS/EC2", "CPUUtilization", "InstanceId", "*" ]
          ]
          region = "us-east-1"
          title  = "EC2 CPU Utilization"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 1
        width  = 12
        height = 6
        properties = {
          view    = "timeSeries"
          stacked = false
          metrics = [
            [ "AWS/EC2", "MemoryUtilization", "InstanceId", "*" ]
          ]
          region = "us-east-1"
          title  = "EC2 Memory Utilization"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 7
        width  = 12
        height = 6
        properties = {
          view    = "timeSeries"
          stacked = false
          metrics = [
            [ "AWS/EKS", "cluster_failed_node_count", "ClusterName", var.eks_cluster_name ]
          ]
          region = "us-east-1"
          title  = "EKS Failed Node Count"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 7
        width  = 12
        height = 6
        properties = {
          view    = "timeSeries"
          stacked = false
          metrics = [
            [ "AWS/EKS", "pod_cpu_utilization", "ClusterName", var.eks_cluster_name ]
          ]
          region = "us-east-1"
          title  = "EKS Pod CPU Utilization"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 13
        width  = 12
        height = 6
        properties = {
          view    = "timeSeries"
          stacked = false
          metrics = [
            [ "AWS/S3", "BucketSizeBytes", "BucketName", var.s3_bucket_names[0], "StorageType", "StandardStorage" ],
            [ "AWS/S3", "BucketSizeBytes", "BucketName", var.s3_bucket_names[1], "StorageType", "StandardStorage" ],
            [ "AWS/S3", "BucketSizeBytes", "BucketName", var.s3_bucket_names[2], "StorageType", "StandardStorage" ]
          ]
          region = "us-east-1"
          title  = "S3 Bucket Size"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 13
        width  = 12
        height = 6
        properties = {
          view    = "timeSeries"
          stacked = false
          metrics = [
            [ "AWS/S3", "NumberOfObjects", "BucketName", var.s3_bucket_names[0], "StorageType", "AllStorageTypes" ],
            [ "AWS/S3", "NumberOfObjects", "BucketName", var.s3_bucket_names[1], "StorageType", "AllStorageTypes" ],
            [ "AWS/S3", "NumberOfObjects", "BucketName", var.s3_bucket_names[2], "StorageType", "AllStorageTypes" ]
          ]
          region = "us-east-1"
          title  = "S3 Number of Objects"
        }
      }
    ]
  })
}

# CloudWatch Alarms for CARLA Simulation Instances
resource "aws_cloudwatch_metric_alarm" "carla_cpu_high" {
  count = var.create_carla_simulation_alarms ? length(var.carla_instance_ids) : 0
  
  alarm_name          = "${var.project_name}-${var.environment}-carla-cpu-high-${count.index}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors EC2 CPU utilization for CARLA simulation instances"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  
  dimensions = {
    InstanceId = var.carla_instance_ids[count.index]
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-carla-cpu-high-${count.index}"
    },
    var.tags
  )
}

resource "aws_cloudwatch_metric_alarm" "carla_memory_high" {
  count = var.create_carla_simulation_alarms ? length(var.carla_instance_ids) : 0
  
  alarm_name          = "${var.project_name}-${var.environment}-carla-memory-high-${count.index}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "mem_used_percent"
  namespace           = "CWAgent"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors memory utilization for CARLA simulation instances"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  
  dimensions = {
    InstanceId = var.carla_instance_ids[count.index]
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-carla-memory-high-${count.index}"
    },
    var.tags
  )
}

resource "aws_cloudwatch_metric_alarm" "carla_disk_high" {
  count = var.create_carla_simulation_alarms ? length(var.carla_instance_ids) : 0
  
  alarm_name          = "${var.project_name}-${var.environment}-carla-disk-high-${count.index}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "disk_used_percent"
  namespace           = "CWAgent"
  period              = 300
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "This metric monitors disk utilization for CARLA simulation instances"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  
  dimensions = {
    InstanceId = var.carla_instance_ids[count.index],
    path       = "/"
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-carla-disk-high-${count.index}"
    },
    var.tags
  )
}

# CloudWatch Alarms for EKS Cluster
resource "aws_cloudwatch_metric_alarm" "eks_node_count" {
  count = var.create_eks_cluster_alarms && var.eks_cluster_name != "" ? 1 : 0
  
  alarm_name          = "${var.project_name}-${var.environment}-eks-node-count"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "cluster_failed_node_count"
  namespace           = "AWS/EKS"
  period              = 300
  statistic           = "Maximum"
  threshold           = 0
  alarm_description   = "This metric monitors EKS cluster failed node count"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  
  dimensions = {
    ClusterName = var.eks_cluster_name
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-eks-node-count"
    },
    var.tags
  )
}

resource "aws_cloudwatch_metric_alarm" "eks_pod_cpu_high" {
  count = var.create_eks_cluster_alarms && var.eks_cluster_name != "" ? 1 : 0
  
  alarm_name          = "${var.project_name}-${var.environment}-eks-pod-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "pod_cpu_utilization"
  namespace           = "AWS/EKS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors EKS pod CPU utilization"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  
  dimensions = {
    ClusterName = var.eks_cluster_name
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-eks-pod-cpu-high"
    },
    var.tags
  )
}

resource "aws_cloudwatch_metric_alarm" "eks_pod_memory_high" {
  count = var.create_eks_cluster_alarms && var.eks_cluster_name != "" ? 1 : 0
  
  alarm_name          = "${var.project_name}-${var.environment}-eks-pod-memory-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "pod_memory_utilization"
  namespace           = "AWS/EKS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors EKS pod memory utilization"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  
  dimensions = {
    ClusterName = var.eks_cluster_name
  }
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-eks-pod-memory-high"
    },
    var.tags
  )
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "carla_simulation" {
  name              = "/aws/ec2/${var.project_name}-${var.environment}-carla-simulation"
  retention_in_days = 30
  
  tags = merge(
    {
      Name = "/aws/ec2/${var.project_name}-${var.environment}-carla-simulation"
    },
    var.tags
  )
}

resource "aws_cloudwatch_log_group" "eks_cluster" {
  count = var.eks_cluster_name != "" ? 1 : 0
  
  name              = "/aws/eks/${var.eks_cluster_name}/cluster"
  retention_in_days = 30
  
  tags = merge(
    {
      Name = "/aws/eks/${var.eks_cluster_name}/cluster"
    },
    var.tags
  )
}

resource "aws_cloudwatch_log_group" "data_processing" {
  name              = "/aws/ecs/${var.project_name}-${var.environment}-data-processing"
  retention_in_days = 30
  
  tags = merge(
    {
      Name = "/aws/ecs/${var.project_name}-${var.environment}-data-processing"
    },
    var.tags
  )
}

resource "aws_cloudwatch_log_group" "model_training" {
  name              = "/aws/ecs/${var.project_name}-${var.environment}-model-training"
  retention_in_days = 30
  
  tags = merge(
    {
      Name = "/aws/ecs/${var.project_name}-${var.environment}-model-training"
    },
    var.tags
  )
}

# CloudWatch Metric Filters
resource "aws_cloudwatch_log_metric_filter" "carla_error" {
  name           = "${var.project_name}-${var.environment}-carla-error"
  pattern        = "ERROR"
  log_group_name = aws_cloudwatch_log_group.carla_simulation.name
  
  metric_transformation {
    name      = "CarlaErrorCount"
    namespace = "${var.project_name}/${var.environment}"
    value     = "1"
  }
}

resource "aws_cloudwatch_log_metric_filter" "data_processing_error" {
  name           = "${var.project_name}-${var.environment}-data-processing-error"
  pattern        = "ERROR"
  log_group_name = aws_cloudwatch_log_group.data_processing.name
  
  metric_transformation {
    name      = "DataProcessingErrorCount"
    namespace = "${var.project_name}/${var.environment}"
    value     = "1"
  }
}

resource "aws_cloudwatch_log_metric_filter" "model_training_error" {
  name           = "${var.project_name}-${var.environment}-model-training-error"
  pattern        = "ERROR"
  log_group_name = aws_cloudwatch_log_group.model_training.name
  
  metric_transformation {
    name      = "ModelTrainingErrorCount"
    namespace = "${var.project_name}/${var.environment}"
    value     = "1"
  }
}

# CloudWatch Alarms for Error Metrics
resource "aws_cloudwatch_metric_alarm" "carla_error_high" {
  alarm_name          = "${var.project_name}-${var.environment}-carla-error-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "CarlaErrorCount"
  namespace           = "${var.project_name}/${var.environment}"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "This metric monitors CARLA simulation error count"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-carla-error-high"
    },
    var.tags
  )
}

resource "aws_cloudwatch_metric_alarm" "data_processing_error_high" {
  alarm_name          = "${var.project_name}-${var.environment}-data-processing-error-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "DataProcessingErrorCount"
  namespace           = "${var.project_name}/${var.environment}"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "This metric monitors data processing error count"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-data-processing-error-high"
    },
    var.tags
  )
}

resource "aws_cloudwatch_metric_alarm" "model_training_error_high" {
  alarm_name          = "${var.project_name}-${var.environment}-model-training-error-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ModelTrainingErrorCount"
  namespace           = "${var.project_name}/${var.environment}"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "This metric monitors model training error count"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  
  tags = merge(
    {
      Name = "${var.project_name}-${var.environment}-model-training-error-high"
    },
    var.tags
  )
}
