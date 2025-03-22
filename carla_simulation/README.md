#!/usr/bin/env python

"""
Road Object Detection - Documentation for CARLA Simulation Module
This document provides an overview of the CARLA simulation scripts and their usage
"""

# CARLA Simulation Module Documentation

## Overview

The CARLA simulation module is designed to generate synthetic data for training and improving 
road object detection models, with a special focus on underrepresented classes such as cyclists.
The module consists of three main components:

1. **Simulation Script** (`simulation.py`): Handles the generation of synthetic data using the CARLA simulator
2. **Data Collection Script** (`data_collector.py`): Processes and collects data from the simulation
3. **Data Annotation Script** (`data_annotator.py`): Creates annotations in various formats (COCO, Pascal VOC, YOLO)

## Requirements

- CARLA Simulator (version 0.9.13 or later)
- Python 3.7+
- Required Python packages (see requirements.txt)
- AWS SDK (boto3) for S3 integration (optional)

## Installation

1. Install CARLA Simulator following the official documentation
2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Simulation

The simulation script (`simulation.py`) is the main entry point for generating synthetic data:

```bash
python simulation.py --config simulation_config.json [--host 127.0.0.1] [--port 2000] [--seed 42]
```

Parameters:
- `--config`: Path to simulation configuration file (JSON)
- `--host`: IP address of the CARLA server (default: 127.0.0.1)
- `--port`: TCP port of the CARLA server (default: 2000)
- `--tm-port`: Traffic Manager port (default: 8000)
- `--seed`: Random seed for reproducibility
- `--sync`: Enable synchronous mode

### Data Collection

The data collection script (`data_collector.py`) processes and collects data from the simulation:

```bash
python data_collector.py --output-dir ./output [--s3-bucket my-bucket] [--s3-prefix carla_data]
```

Parameters:
- `--output-dir`: Output directory for collected data
- `--s3-bucket`: S3 bucket for data upload (optional)
- `--s3-prefix`: S3 prefix for data upload (default: carla_data)
- `--create-archive`: Create compressed archive of dataset
- `--archive-path`: Path for dataset archive

### Data Annotation

The data annotation script (`data_annotator.py`) creates annotations in various formats:

```bash
python data_annotator.py --input-dir ./output --output-dir ./annotations --format yolo [--visualize]
```

Parameters:
- `--input-dir`: Input directory containing collected data
- `--output-dir`: Output directory for annotations
- `--format`: Annotation format (coco, pascal_voc, yolo)
- `--visualize`: Create visualizations of annotations
- `--s3-bucket`: S3 bucket for annotation upload (optional)
- `--s3-prefix`: S3 prefix for annotation upload (default: carla_data)

## Configuration

The simulation is configured using a JSON file with the following structure:

```json
{
  "simulation_parameters": {
    "town": "Town05",
    "ego_vehicle": "vehicle.tesla.model3",
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

## Output Structure

The simulation generates the following directory structure:

```
output/
├── rgb/              # RGB camera images
├── depth/            # Depth camera images
├── semantic/         # Semantic segmentation images
├── lidar/            # LiDAR point cloud data
├── annotations/      # Raw annotations from CARLA
└── metadata/         # Metadata about each frame
```

The annotation script generates annotations in the specified format:

```
annotations/
├── coco/             # COCO format annotations
├── pascal_voc/       # Pascal VOC format annotations
├── yolo/             # YOLO format annotations
└── visualizations/   # Visualizations of annotations (if enabled)
```

## Testing

The module includes a test script (`test_simulation.py`) to verify functionality:

```bash
python -m unittest test_simulation.py
```

## Focus on Cyclists

This module places special emphasis on generating synthetic data for cyclists, which are often underrepresented in real-world datasets. The simulation parameters can be adjusted to increase the number of cyclists relative to other actors, and the annotation process ensures accurate labeling of cyclist instances.

## Integration with AWS

The module supports integration with AWS S3 for storing generated data and annotations. To enable this feature, provide the S3 bucket name and prefix when running the data collection and annotation scripts.

## Limitations

- The CARLA simulator must be running before executing the simulation script
- The quality of synthetic data depends on the realism of the CARLA simulator
- Large datasets may require significant storage space and processing time
