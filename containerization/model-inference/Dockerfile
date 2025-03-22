# Dockerfile for Model Inference Service
# This Dockerfile sets up an environment for running YOLOv5 inference

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

LABEL maintainer="Road Object Detection Project Team"
LABEL description="Model Inference Service for YOLOv5 object detection"
LABEL version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
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

# Install additional dependencies for API
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy inference code
COPY model_inference/ /app/model_inference/

# Create directories for models and data
RUN mkdir -p /data/models /data/images /data/results

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/data/models
ENV IMAGES_DIR=/data/images
ENV RESULTS_DIR=/data/results

# Set up entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Expose port for API
EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "model_inference.api", "--model", "/data/models/best.pt", "--port", "8000"]
