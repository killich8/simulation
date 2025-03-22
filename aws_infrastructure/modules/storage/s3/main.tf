resource "aws_s3_bucket" "raw_data" {
  bucket = var.raw_data_bucket_name
  
  tags = merge(
    {
      Name        = var.raw_data_bucket_name
      Environment = var.environment
      Project     = var.project_name
      DataType    = "RawData"
    },
    var.tags
  )
}

resource "aws_s3_bucket_versioning" "raw_data" {
  count = var.enable_versioning ? 1 : 0
  
  bucket = aws_s3_bucket.raw_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "raw_data" {
  count = var.enable_lifecycle_rules ? 1 : 0
  
  bucket = aws_s3_bucket.raw_data.id
  
  rule {
    id     = "transition-to-intelligent-tiering"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "processed_data" {
  bucket = var.processed_data_bucket_name
  
  tags = merge(
    {
      Name        = var.processed_data_bucket_name
      Environment = var.environment
      Project     = var.project_name
      DataType    = "ProcessedData"
    },
    var.tags
  )
}

resource "aws_s3_bucket_versioning" "processed_data" {
  count = var.enable_versioning ? 1 : 0
  
  bucket = aws_s3_bucket.processed_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "processed_data" {
  count = var.enable_lifecycle_rules ? 1 : 0
  
  bucket = aws_s3_bucket.processed_data.id
  
  rule {
    id     = "transition-to-intelligent-tiering"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "processed_data" {
  bucket = aws_s3_bucket.processed_data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "model_artifacts" {
  bucket = var.model_artifacts_bucket_name
  
  tags = merge(
    {
      Name        = var.model_artifacts_bucket_name
      Environment = var.environment
      Project     = var.project_name
      DataType    = "ModelArtifacts"
    },
    var.tags
  )
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  count = var.enable_versioning ? 1 : 0
  
  bucket = aws_s3_bucket.model_artifacts.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "model_artifacts" {
  count = var.enable_lifecycle_rules ? 1 : 0
  
  bucket = aws_s3_bucket.model_artifacts.id
  
  rule {
    id     = "transition-to-intelligent-tiering"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "logs" {
  bucket = var.logs_bucket_name
  
  tags = merge(
    {
      Name        = var.logs_bucket_name
      Environment = var.environment
      Project     = var.project_name
      DataType    = "Logs"
    },
    var.tags
  )
}

resource "aws_s3_bucket_versioning" "logs" {
  count = var.enable_versioning ? 1 : 0
  
  bucket = aws_s3_bucket.logs.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  count = var.enable_lifecycle_rules ? 1 : 0
  
  bucket = aws_s3_bucket.logs.id
  
  rule {
    id     = "expire-old-logs"
    status = "Enabled"
    
    expiration {
      days = 90
    }
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Create bucket policies for cross-account access if needed
# resource "aws_s3_bucket_policy" "example" {
#   bucket = aws_s3_bucket.example.id
#   policy = data.aws_iam_policy_document.example.json
# }

# data "aws_iam_policy_document" "example" {
#   statement {
#     principals {
#       type        = "AWS"
#       identifiers = ["arn:aws:iam::ACCOUNT-ID:root"]
#     }
#     actions   = ["s3:GetObject"]
#     resources = ["${aws_s3_bucket.example.arn}/*"]
#   }
# }
