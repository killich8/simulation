resource "aws_instance" "this" {
  count = var.instance_count

  ami                    = var.ami
  instance_type          = var.instance_type
  key_name               = var.key_name
  monitoring             = var.monitoring
  vpc_security_group_ids = var.vpc_security_group_ids
  subnet_id              = element(var.subnet_ids, count.index)
  associate_public_ip_address = var.associate_public_ip_address
  user_data              = var.user_data

  dynamic "root_block_device" {
    for_each = var.root_block_device
    content {
      delete_on_termination = lookup(root_block_device.value, "delete_on_termination", true)
      encrypted             = lookup(root_block_device.value, "encrypted", true)
      iops                  = lookup(root_block_device.value, "iops", null)
      throughput            = lookup(root_block_device.value, "throughput", null)
      volume_size           = lookup(root_block_device.value, "volume_size", null)
      volume_type           = lookup(root_block_device.value, "volume_type", "gp3")
    }
  }

  dynamic "ebs_block_device" {
    for_each = var.ebs_block_device
    content {
      delete_on_termination = lookup(ebs_block_device.value, "delete_on_termination", true)
      device_name           = ebs_block_device.value.device_name
      encrypted             = lookup(ebs_block_device.value, "encrypted", true)
      iops                  = lookup(ebs_block_device.value, "iops", null)
      throughput            = lookup(ebs_block_device.value, "throughput", null)
      volume_size           = lookup(ebs_block_device.value, "volume_size", null)
      volume_type           = lookup(ebs_block_device.value, "volume_type", "gp3")
    }
  }

  tags = merge(
    {
      Name = "${var.name}-${count.index + 1}"
    },
    var.tags
  )

  lifecycle {
    create_before_destroy = true
  }
}

# Create CloudWatch agent configuration
resource "aws_ssm_parameter" "cloudwatch_agent_config" {
  name  = "/cloudwatch-agent/config/${var.name}"
  type  = "String"
  value = jsonencode({
    agent = {
      metrics_collection_interval = 60
      run_as_user                 = "root"
    }
    metrics = {
      append_dimensions = {
        InstanceId = "$${aws:InstanceId}"
      }
      metrics_collected = {
        cpu = {
          measurement = [
            "cpu_usage_idle",
            "cpu_usage_iowait",
            "cpu_usage_user",
            "cpu_usage_system"
          ]
          metrics_collection_interval = 60
          totalcpu                    = true
        }
        disk = {
          measurement = [
            "used_percent",
            "inodes_free"
          ]
          metrics_collection_interval = 60
          resources                   = ["*"]
        }
        diskio = {
          measurement = [
            "io_time",
            "write_bytes",
            "read_bytes",
            "writes",
            "reads"
          ]
          metrics_collection_interval = 60
          resources                   = ["*"]
        }
        mem = {
          measurement = [
            "mem_used_percent"
          ]
          metrics_collection_interval = 60
        }
        netstat = {
          measurement = [
            "tcp_established",
            "tcp_time_wait"
          ]
          metrics_collection_interval = 60
        }
        swap = {
          measurement = [
            "swap_used_percent"
          ]
          metrics_collection_interval = 60
        }
      }
    }
    logs = {
      logs_collected = {
        files = {
          collect_list = [
            {
              file_path        = "/var/log/syslog"
              log_group_name   = "/ec2/${var.name}/syslog"
              log_stream_name  = "{instance_id}"
              retention_in_days = 30
            },
            {
              file_path        = "/var/log/carla/carla.log"
              log_group_name   = "/ec2/${var.name}/carla"
              log_stream_name  = "{instance_id}"
              retention_in_days = 30
            }
          ]
        }
      }
    }
  })

  tags = merge(
    {
      Name = "/cloudwatch-agent/config/${var.name}"
    },
    var.tags
  )
}
