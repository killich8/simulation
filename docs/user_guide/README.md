# User Guide

## Introduction

This guide provides instructions for using the Road Object Detection Improvement Pipeline. The pipeline is designed to improve the accuracy of road object detection models by generating and integrating synthetic data from the CARLA simulator, with a focus on underrepresented classes such as cyclists.

## Prerequisites

- AWS account with appropriate permissions
- Basic understanding of machine learning concepts
- Basic understanding of Docker and Kubernetes (for advanced usage)

## Getting Started

### Configuration

The pipeline is configured using a JSON file that defines simulation parameters. Create a file named `simulation_params.json` with the following structure:

```json
{
  "simulation_parameters": {
    "town": "Town05",
    "weather": {
      "cloudiness": 0,
      "precipitation": 0,
      "precipitation_deposits": 0,
      "wind_intensity": 0,
      "sun_azimuth_angle": 70,
      "sun_altitude_angle": 70,
      "fog_density": 0,
      "fog_distance": 0,
      "wetness": 0
    },
    "actors": {
      "vehicles": {
        "count": 50,
        "safe_distance": 5.0,
        "speed_factor": 1.0
      },
      "pedestrians": {
        "count": 30,
        "safe_distance": 2.0,
        "speed_factor": 0.8
      },
      "cyclists": {
        "count": 15,
        "safe_distance": 3.0,
        "speed_factor": 1.2
      }
    },
    "data_collection": {
      "frames_per_second": 10,
      "duration_seconds": 300,
      "sensors": {
        "rgb_camera": {
          "width": 1280,
          "height": 720,
          "fov": 90,
          "position": [1.5, 0.0, 2.4],
          "rotation": [0.0, 0.0, 0.0]
        },
        "depth_camera": {
          "width": 1280,
          "height": 720,
          "fov": 90,
          "position": [1.5, 0.0, 2.4],
          "rotation": [0.0, 0.0, 0.0]
        },
        "semantic_segmentation": {
          "width": 1280,
          "height": 720,
          "fov": 90,
          "position": [1.5, 0.0, 2.4],
          "rotation": [0.0, 0.0, 0.0]
        },
        "lidar": {
          "channels": 64,
          "range": 100,
          "points_per_second": 500000,
          "rotation_frequency": 10,
          "position": [0.0, 0.0, 2.4]
        }
      }
    }
  }
}
```

Adjust the parameters according to your specific requirements:

- **town**: CARLA town map to use (e.g., Town01, Town05)
- **weather**: Weather conditions for the simulation
- **actors**: Number and behavior of vehicles, pedestrians, and cyclists
- **data_collection**: Sensor configurations and data collection parameters

### Running the Pipeline

1. Upload your configuration file to the S3 bucket:
   ```
   aws s3 cp simulation_params.json s3://road-detection-data/config/simulation_params.json
   ```

2. Trigger the pipeline using the AWS CLI:
   ```
   aws lambda invoke --function-name road-detection-pipeline-trigger --payload '{"configPath": "config/simulation_params.json"}' response.json
   ```

3. Monitor the pipeline progress through the Grafana dashboard:
   ```
   https://<your-grafana-url>/d/road-detection/pipeline-status
   ```

## Pipeline Stages

The pipeline consists of the following stages:

1. **Data Generation**: CARLA simulator generates synthetic road scenarios based on your configuration.
2. **Data Processing**: Generated data is validated, transformed, and augmented.
3. **Data Storage**: Processed data is stored in S3.
4. **Model Training**: YOLOv5 model is trained using the synthetic data.
5. **Model Evaluation**: Trained model is evaluated on validation datasets.
6. **Model Deployment**: If performance metrics meet thresholds, the model is deployed for inference.

## Monitoring

You can monitor the pipeline and model performance through the Grafana dashboard:

- **Pipeline Status**: Current stage, progress, and estimated completion time
- **Data Generation Metrics**: Number of frames generated, distribution of object classes
- **Training Metrics**: Loss curves, learning rate, and other training statistics
- **Model Performance**: Precision, recall, mAP, and other evaluation metrics
- **Resource Utilization**: CPU, GPU, memory usage of various components

## Troubleshooting

### Common Issues

1. **Pipeline Fails at Data Generation Stage**
   - Check if CARLA simulator has enough resources
   - Verify that the town map specified in configuration exists
   - Reduce the number of actors if the simulation is too complex

2. **Low Quality Synthetic Data**
   - Adjust weather parameters for better visibility
   - Modify camera positions and angles
   - Increase the variety of scenarios by changing actor parameters

3. **Model Training Issues**
   - Check if there's enough data for each class
   - Adjust training hyperparameters
   - Verify that the data is properly annotated

### Getting Help

For additional support, contact the system administrator or refer to the developer documentation.
