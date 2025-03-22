# Data Processing Pipeline

This module provides a comprehensive data processing pipeline for the road object detection project, with a special focus on underrepresented classes such as cyclists.

## Overview

The data processing pipeline consists of three main components:

1. **Data Validation**: Validates input data to ensure it meets quality standards
2. **Data Transformation**: Transforms data to the required format and size
3. **Data Augmentation**: Augments data to increase dataset size and diversity, with special emphasis on cyclists

## Directory Structure

```
data_processing/
├── pipeline.py                # Main pipeline orchestration
├── config.py                  # Configuration settings
├── validation/
│   └── data_validator.py      # Data validation module
├── transformation/
│   └── data_transformer.py    # Data transformation module
├── augmentation/
│   └── data_augmenter.py      # Data augmentation module
└── utils/
    └── data_utils.py          # Utility functions
```

## Installation

The data processing pipeline requires the following dependencies:

```bash
pip install numpy opencv-python pillow albumentations boto3 matplotlib seaborn
```

## Usage

### Basic Usage

```python
from data_processing.pipeline import DataProcessingPipeline

# Create pipeline with configuration file
pipeline = DataProcessingPipeline('config.json')

# Run the complete pipeline
result = pipeline.run()

if result:
    print("Pipeline completed successfully")
else:
    print("Pipeline failed")
```

### Configuration

The pipeline can be configured using a JSON file with the following structure:

```json
{
  "input_dir": "./input",
  "output_dir": "./output",
  "temp_dir": "./temp",
  "validation": {
    "min_image_size": [640, 480],
    "max_image_size": [4096, 4096],
    "min_objects_per_image": 1,
    "required_classes": ["car", "pedestrian", "bicycle"],
    "min_class_instances": {
      "bicycle": 100,
      "cyclist": 100
    }
  },
  "transformation": {
    "target_size": [640, 640],
    "normalize": true,
    "convert_to_format": "yolo"
  },
  "augmentation": {
    "augmentation_types": ["flip", "rotate", "brightness", "contrast", "noise"],
    "augmentation_per_image": 3,
    "cyclist_augmentation_factor": 2.0,
    "include_original": true
  }
}
```

## Components

### Data Validator

The data validator ensures that input data meets quality standards:

- Validates image sizes
- Checks for minimum number of objects per image
- Ensures required classes are present
- Verifies annotation format and correctness
- Provides detailed statistics about the dataset

### Data Transformer

The data transformer prepares data for model training:

- Resizes images to target dimensions
- Normalizes pixel values
- Converts between annotation formats (YOLO, COCO, Pascal VOC)
- Handles class mapping for consistent labeling

### Data Augmenter

The data augmenter increases dataset size and diversity:

- Applies various augmentation techniques (flip, rotate, brightness, contrast, noise)
- Provides special emphasis on underrepresented classes like cyclists
- Supports class-specific augmentation strategies
- Maintains annotation consistency during augmentation

### Utilities

The utilities module provides helper functions:

- Dataset statistics generation
- Visualization of dataset characteristics
- S3 integration for data storage and retrieval
- Format detection and conversion

## Focus on Cyclists

This pipeline places special emphasis on cyclists as an underrepresented class:

- Increased augmentation factor for images containing cyclists
- Special augmentation techniques tailored for cyclist detection
- Validation to ensure sufficient cyclist instances in the dataset
- Statistics tracking for cyclist representation

## Integration with AWS

The pipeline integrates with AWS S3 for data storage:

- Upload processed datasets to S3
- Download datasets from S3
- Store statistics and visualizations alongside data

## Command Line Interface

The pipeline can be run from the command line:

```bash
python -m data_processing.pipeline --config config.json
```

## Output

The pipeline generates the following outputs:

- Processed dataset in the specified format
- Dataset statistics and visualizations
- Logs of the processing steps and any issues encountered
