#!/bin/bash
# Docker entrypoint script for data processing service

set -e

# Check if config file exists
if [ ! -f "$1" ]; then
  echo "Config file not found: $1"
  echo "Using default configuration"
fi

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
mkdir -p $DATA_INPUT_DIR $DATA_OUTPUT_DIR $DATA_TEMP_DIR

# Execute the command passed to the container
echo "Executing: $@"
exec "$@"
