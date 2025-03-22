# Dockerfile for CARLA Simulation Service
# This Dockerfile sets up an environment for running CARLA simulation with Python scripts

FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

LABEL maintainer="Road Object Detection Project Team"
LABEL description="CARLA Simulation Service for synthetic data generation"
LABEL version="1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    software-properties-common \
    git \
    cmake \
    build-essential \
    libpng-dev \
    libtiff-dev \
    libjpeg-dev \
    libgif-dev \
    libgtk2.0-dev \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-opencv \
    libopencv-dev \
    libvulkan1 \
    vulkan-utils \
    libomp-dev \
    xdg-user-dirs \
    xserver-xorg \
    x11-xserver-utils \
    libsdl2-2.0-0 \
    libsdl2-dev \
    libvulkan-dev \
    libopenexr-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Install CARLA Python API
RUN mkdir -p /opt/carla
WORKDIR /opt/carla

# Download and extract CARLA (version 0.9.13)
RUN wget -q https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz \
    && tar -xzf CARLA_0.9.13.tar.gz \
    && rm CARLA_0.9.13.tar.gz

# Add CARLA Python API to PYTHONPATH
ENV PYTHONPATH=/opt/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:/opt/carla/PythonAPI/carla

# Create directory for simulation scripts
RUN mkdir -p /app/simulation
WORKDIR /app/simulation

# Copy simulation scripts
COPY carla_simulation/*.py /app/simulation/
COPY carla_simulation/simulation_config_template.json /app/simulation/

# Create directories for output data
RUN mkdir -p /data/raw /data/processed

# Set environment variables
ENV CARLA_ROOT=/opt/carla
ENV PYTHONPATH=$PYTHONPATH:/app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Set up entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python3", "simulation.py", "--config", "simulation_config_template.json"]
