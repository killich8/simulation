# Jenkins CI/CD Pipeline Configuration

This directory contains the CI/CD pipeline configuration for the Road Object Detection project using Jenkins.

## Overview

The CI/CD pipeline automates the building, testing, and deployment of the road object detection system. It includes:

- Static code analysis
- Unit testing
- Security scanning
- Docker image building
- Deployment to Kubernetes
- Integration and performance testing

## Pipeline Stages

The Jenkinsfile defines a complete pipeline with the following stages:

1. **Checkout**: Retrieves the source code from the repository
2. **Static Code Analysis**: Runs linting tools on Python code, Dockerfiles, Terraform files, and Kubernetes manifests
3. **Unit Tests**: Executes unit tests with code coverage reporting
4. **Security Scan**: Performs dependency checks, container security scans, and secret detection
5. **Build Docker Images**: Creates Docker images for all services (CARLA simulation, data processing, model training, inference)
6. **Push Docker Images**: Pushes images to Amazon ECR (only on main branch)
7. **Deploy to Kubernetes**: Deploys the application to the Kubernetes cluster (only on main branch)
8. **Integration Tests**: Runs integration tests against the deployed services
9. **Performance Tests**: Executes performance tests to ensure the system meets performance requirements

## Requirements

To use this pipeline, you need:

1. Jenkins server with the following plugins:
   - Kubernetes plugin
   - Docker plugin
   - AWS Credentials plugin
   - JUnit plugin
   - Cobertura plugin
   - Email Extension plugin

2. Kubernetes cluster with:
   - Jenkins agent configured
   - Proper RBAC permissions

3. AWS credentials configured in Jenkins with:
   - ECR push permissions
   - EKS access permissions

## Configuration

The pipeline uses the following environment variables:

- `AWS_REGION`: AWS region where resources are deployed
- `AWS_ACCOUNT_ID`: AWS account ID (stored as a Jenkins credential)
- `PROJECT_NAME`: Name of the project (road-object-detection)
- `GITHUB_REPO`: GitHub repository URL
- `BRANCH_NAME`: Branch to build (main)
- `K8S_NAMESPACE`: Kubernetes namespace for deployment

## Security Considerations

The pipeline includes several security measures:

1. Credentials are stored securely in Jenkins and accessed via the credentials binding plugin
2. Security scanning is performed on dependencies, containers, and source code
3. Secrets are not hardcoded in the pipeline configuration

## Extending the Pipeline

To extend the pipeline:

1. Add new stages to the Jenkinsfile
2. Configure additional tools in the Kubernetes pod template
3. Add new environment variables as needed

## Troubleshooting

Common issues:

1. **Docker build failures**: Check Docker daemon access and resource constraints
2. **AWS authentication issues**: Verify AWS credentials are correctly configured
3. **Kubernetes deployment failures**: Check EKS cluster access and RBAC permissions
