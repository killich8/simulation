#!/bin/bash
# Docker entrypoint script for model inference service

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
mkdir -p $MODEL_DIR $IMAGES_DIR $RESULTS_DIR

# Check if we need to download model from S3
if [ -n "$S3_MODEL_BUCKET" ] && [ -n "$S3_MODEL_KEY" ]; then
  echo "Downloading model from S3: s3://$S3_MODEL_BUCKET/$S3_MODEL_KEY"
  aws s3 cp s3://$S3_MODEL_BUCKET/$S3_MODEL_KEY $MODEL_DIR
  # Update model path in command if provided
  if [[ "$@" == *"--model"* ]]; then
    MODEL_FILE=$(basename $S3_MODEL_KEY)
    set -- "${@/--model \/data\/models\/best.pt/--model $MODEL_DIR/$MODEL_FILE}"
  fi
fi

# Execute the command passed to the container
echo "Executing: $@"
exec "$@"
