#!/bin/bash
# Docker entrypoint script for model training service

set -e

# Check if AWS credentials are provided
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "AWS credentials found, configuring AWS CLI"
  mkdir -p ~/.aws
  cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF

  if [ -n "$AWS_REGION" ]; then
    cat > ~/.aws/config << EOF
[default]
region = $AWS_REGION
EOF
  fi
fi

# Create directories if they don't exist
mkdir -p $DATASET_DIR $MODEL_DIR $RESULTS_DIR

# Check if we need to download dataset from S3
if [ -n "$S3_DATASET_BUCKET" ] && [ -n "$S3_DATASET_KEY" ]; then
  echo "Downloading dataset from S3: s3://$S3_DATASET_BUCKET/$S3_DATASET_KEY"
  aws s3 cp s3://$S3_DATASET_BUCKET/$S3_DATASET_KEY $DATASET_DIR --recursive
fi

# Check if we need to download pre-trained weights from S3
if [ -n "$S3_WEIGHTS_BUCKET" ] && [ -n "$S3_WEIGHTS_KEY" ]; then
  echo "Downloading pre-trained weights from S3: s3://$S3_WEIGHTS_BUCKET/$S3_WEIGHTS_KEY"
  aws s3 cp s3://$S3_WEIGHTS_BUCKET/$S3_WEIGHTS_KEY $MODEL_DIR
  # Update weights path in command if provided
  if [[ "$@" == *"--weights"* ]]; then
    WEIGHTS_FILE=$(basename $S3_WEIGHTS_KEY)
    set -- "${@/--weights yolov5s.pt/--weights $MODEL_DIR/$WEIGHTS_FILE}"
  fi
fi

# Execute the command passed to the container
echo "Executing: $@"
exec "$@"
