# Road Object Detection Improvement Project

## Project Overview

This project aims to improve the accuracy of road object detection models in specific scenarios by generating and integrating synthetic data using the CARLA simulator, with a focus on underrepresented classes such as cyclists.

## Key Components

1. **Synthetic Data Generation**: Automated generation of road scenarios via the CARLA simulator and Python scripts
2. **AWS Infrastructure**: Cloud infrastructure deployed using Terraform and Ansible
3. **Containerized Architecture**: Docker and Kubernetes for simulation and inference services
4. **CI/CD Pipeline**: Automated build, test, and deployment using Jenkins
5. **Monitoring Solution**: Prometheus and Grafana for tracking model performance
6. **DevSecOps Practices**: Security scans integrated into CI/CD pipelines
7. **Model Training**: YOLOv5 model from Ultralytics for object detection

## Workflow

The pipeline starts automatically once the user defines simulation parameters (number of pedestrians, vehicles, cyclists, town, etc.) in a JSON file. The complete workflow includes:

1. Data generation in CARLA simulator
2. Data quality testing
3. Storage in AWS S3
4. Model training using YOLOv5
5. Model evaluation
6. Deployment for inference

## Project Structure

```
road_object_detection_project/
├── carla_simulation/      # CARLA simulator scripts and configurations
├── data_processing/       # Data validation, transformation, and augmentation
├── aws_infrastructure/    # Terraform and Ansible files for AWS deployment
├── containerization/      # Docker and Kubernetes configurations
├── cicd_pipeline/         # Jenkins pipeline definitions
├── monitoring/            # Prometheus and Grafana configurations
├── model_training/        # YOLOv5 training scripts and configurations
├── orchestration/         # End-to-end pipeline orchestration
└── docs/                  # Project documentation
```

## Getting Started

Detailed instructions for setup and usage can be found in the documentation:
- [User Guide](./docs/user_guide.md)
- [Developer Guide](./docs/developer_guide.md)
- [Deployment Guide](./docs/deployment_guide.md)

## Requirements

See [requirements.txt](./requirements.txt) for Python dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
