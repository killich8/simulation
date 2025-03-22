# Dockerfile for Model Training Service
# This Dockerfile sets up an environment for training YOLOv5 models

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

LABEL maintainer="Road Object Detection Project Team"
LABEL description="Model Training Service for YOLOv5 object detection"
LABEL version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5

# Install YOLOv5 dependencies
RUN pip install --no-cache-dir -r /app/yolov5/requirements.txt

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model training code
COPY model_training/ /app/model_training/

# Create directories for data and models
RUN mkdir -p /data/datasets /data/models /data/results

# Set environment variables
ENV PYTHONPATH=/app
ENV DATASET_DIR=/data/datasets
ENV MODEL_DIR=/data/models
ENV RESULTS_DIR=/data/results

# Set up entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "model_training.train", "--data", "/data/datasets/data.yaml", "--cfg", "/app/model_training/models/yolov5s.yaml", "--weights", "yolov5s.pt", "--batch-size", "16"]
