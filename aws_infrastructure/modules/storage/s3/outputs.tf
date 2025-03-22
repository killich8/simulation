output "raw_data_bucket_id" {
  description = "The name of the bucket for raw data"
  value       = aws_s3_bucket.raw_data.id
}

output "raw_data_bucket_arn" {
  description = "The ARN of the bucket for raw data"
  value       = aws_s3_bucket.raw_data.arn
}

output "processed_data_bucket_id" {
  description = "The name of the bucket for processed data"
  value       = aws_s3_bucket.processed_data.id
}

output "processed_data_bucket_arn" {
  description = "The ARN of the bucket for processed data"
  value       = aws_s3_bucket.processed_data.arn
}

output "model_artifacts_bucket_id" {
  description = "The name of the bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.id
}

output "model_artifacts_bucket_arn" {
  description = "The ARN of the bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.arn
}

output "logs_bucket_id" {
  description = "The name of the bucket for logs"
  value       = aws_s3_bucket.logs.id
}

output "logs_bucket_arn" {
  description = "The ARN of the bucket for logs"
  value       = aws_s3_bucket.logs.arn
}
