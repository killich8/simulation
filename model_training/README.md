# Road Object Detection Project with YOLOv5 and CARLA

This project aims to improve the accuracy of a road object detection model in specific scenarios by generating and integrating synthetic data using the CARLA simulator, with a focus on underrepresented classes such as cyclists.

## Overview

The project implements an end-to-end pipeline that:

1. Generates synthetic road scenarios via the CARLA simulator
2. Processes and augments the data with focus on underrepresented classes
3. Trains a YOLOv5 model with optimized hyperparameters
4. Deploys the model for inference
5. Monitors model performance in real-time

The entire pipeline is containerized, deployed on AWS, and managed through CI/CD practices.

## Key Features

- Automated generation of road scenarios with configurable parameters
- Special focus on underrepresented classes like cyclists
- Containerized architecture for portability and scalability
- AWS infrastructure managed with Terraform and Ansible
- CI/CD pipeline with Jenkins and security scanning
- Real-time monitoring with Prometheus and Grafana
- Optimized YOLOv5 model training and inference

## Model Training Integration

The model training component is designed to improve detection accuracy for underrepresented classes like cyclists by:

1. Using class weights to handle class imbalance
2. Implementing specialized augmentation techniques for cyclist instances
3. Optimizing hyperparameters for synthetic data
4. Providing detailed evaluation metrics focused on cyclist detection
5. Supporting model optimization for deployment

## Usage

The pipeline starts automatically once the user defines simulation parameters in a JSON file. The parameters include:

- Number of pedestrians, vehicles, and cyclists
- Town/environment selection
- Weather and lighting conditions
- Camera configurations
- Simulation duration

## Directory Structure

```
road_object_detection_project/
├── carla_simulation/         # CARLA simulation scripts
├── data_processing/          # Data processing pipeline
├── aws_infrastructure/       # Terraform and Ansible files
├── containerization/         # Docker and Kubernetes files
├── cicd_pipeline/            # Jenkins pipeline configuration
├── monitoring/               # Prometheus and Grafana setup
├── model_training/           # YOLOv5 integration
│   ├── config/               # Dataset and hyperparameter configs
│   ├── scripts/              # Training, evaluation, and inference scripts
│   └── utils/                # Utility functions
├── orchestration/            # End-to-end pipeline orchestration
└── docs/                     # Documentation
```

## Model Training Configuration

The model training is configured to focus on cyclist detection:

- Dataset configuration with higher weights for cyclists
- Specialized data augmentation for cyclist instances
- Evaluation metrics focused on cyclist detection performance
- Inference optimization for deployment

## Next Steps

The next phase involves implementing the end-to-end pipeline orchestration to connect all components and ensure seamless operation from data generation to model deployment.
