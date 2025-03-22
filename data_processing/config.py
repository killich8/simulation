#!/usr/bin/env python

"""
Road Object Detection - Data Processing Configuration
This file contains the default configuration for the data processing pipeline
"""

import os

# Default configuration
DEFAULT_CONFIG = {
    # Input and output directories
    "input_dir": "./input",
    "output_dir": "./output",
    "temp_dir": "./temp",
    
    # Validation configuration
    "validation": {
        "min_image_size": [640, 480],
        "max_image_size": [4096, 4096],
        "min_objects_per_image": 1,
        "required_classes": ["car", "pedestrian", "bicycle"],
        "min_class_instances": {
            "bicycle": 100,
            "cyclist": 100
        },
        "annotation_formats": ["yolo", "coco", "pascal_voc"]
    },
    
    # Transformation configuration
    "transformation": {
        "target_size": [640, 640],
        "normalize": True,
        "convert_to_format": "yolo",
        "include_original": False,
        "class_mapping": {}
    },
    
    # Augmentation configuration
    "augmentation": {
        "augmentation_types": ["flip", "rotate", "brightness", "contrast", "noise"],
        "augmentation_per_image": 3,
        "cyclist_augmentation_factor": 2.0,
        "include_original": True,
        "class_specific_augmentations": {
            "bicycle": {
                "augmentation_factor": 2.0
            },
            "cyclist": {
                "augmentation_factor": 2.0
            }
        }
    },
    
    # AWS configuration
    "aws": {
        "s3_bucket": "road-object-detection-data",
        "s3_prefix": "processed_data"
    }
}

def get_config(config_path=None):
    """
    Get configuration from file or use default
    
    Args:
        config_path: Path to configuration file (JSON)
    
    Returns:
        Configuration dictionary
    """
    import json
    
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Update default config with user config
            _update_dict(config, user_config)
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            print("Using default configuration")
    
    return config

def _update_dict(d, u):
    """
    Recursively update dictionary
    
    Args:
        d: Dictionary to update
        u: Dictionary with updates
    
    Returns:
        Updated dictionary
    """
    import collections.abc
    
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def save_config(config, config_path):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    
    Returns:
        Success status
    """
    import json
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {e}")
        return False
