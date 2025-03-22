# Dockerfile for Data Processing Service
# This Dockerfile sets up an environment for data processing pipeline

FROM python:3.9-slim

LABEL maintainer="Road Object Detection Project Team"
LABEL description="Data Processing Service for validation, transformation, and augmentation"
LABEL version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data processing code
COPY data_processing/ /app/data_processing/

# Create directories for data
RUN mkdir -p /data/input /data/output /data/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_INPUT_DIR=/data/input
ENV DATA_OUTPUT_DIR=/data/output
ENV DATA_TEMP_DIR=/data/temp

# Set up entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "data_processing.pipeline", "--config", "/app/data_processing/config.py"]
