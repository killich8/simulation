#!/usr/bin/env python3
"""
train_yolov5.py - Script for training YOLOv5 models on road object detection dataset
with focus on underrepresented classes like cyclists.

This script integrates with the CARLA synthetic data pipeline and handles:
1. Loading synthetic data from S3
2. Preparing the dataset with proper class balancing
3. Training YOLOv5 with optimized hyperparameters
4. Evaluating model performance with focus on cyclist detection
5. Saving model artifacts to S3
"""

import argparse
import os
import sys
import yaml
import json
import boto3
import torch
import numpy as np
from pathlib import Path
import shutil
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_yolov5.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("train_yolov5")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv5 on road object detection dataset')
    parser.add_argument('--simulation-id', type=str, required=True,
                        help='ID of the simulation run to use for training')
    parser.add_argument('--data-config', type=str, 
                        default='config/road_objects.yaml',
                        help='Path to dataset configuration YAML')
    parser.add_argument('--hyp-config', type=str, 
                        default='config/hyp_road_objects.yaml',
                        help='Path to hyperparameters YAML')
    parser.add_argument('--weights', type=str, 
                        default='yolov5s.pt',
                        help='Initial weights path')
    parser.add_argument('--img-size', type=int, 
                        default=640,
                        help='Image size for training')
    parser.add_argument('--batch-size', type=int, 
                        default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, 
                        default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--s3-bucket', type=str, 
                        default='road-object-detection-data',
                        help='S3 bucket for data and model artifacts')
    parser.add_argument('--device', type=str, 
                        default='0',
                        help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, 
                        default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--project', type=str, 
                        default='runs/train',
                        help='Directory to save results to')
    parser.add_argument('--name', type=str, 
                        default='exp',
                        help='Name of the experiment')
    parser.add_argument('--exist-ok', action='store_true',
                        help='Existing project/name ok, do not increment')
    parser.add_argument('--no-upload', action='store_true',
                        help='Do not upload model artifacts to S3')
    parser.add_argument('--cyclist-focus', action='store_true',
                        help='Enable special handling for cyclist class')
    
    return parser.parse_args()

def download_data_from_s3(bucket_name, simulation_id, local_dataset_path):
    """Download synthetic data from S3 bucket."""
    logger.info(f"Downloading data from S3 bucket {bucket_name} for simulation {simulation_id}")
    
    s3 = boto3.client('s3')
    
    # Create local directories
    os.makedirs(local_dataset_path, exist_ok=True)
    os.makedirs(os.path.join(local_dataset_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(local_dataset_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(local_dataset_path, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(local_dataset_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(local_dataset_path, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(local_dataset_path, 'labels', 'test'), exist_ok=True)
    
    # Download data
    prefix = f"simulations/{simulation_id}/processed/"
    
    try:
        # List objects in the bucket with the given prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' not in response:
            logger.error(f"No data found in S3 bucket {bucket_name} with prefix {prefix}")
            return False
        
        total_files = len(response['Contents'])
        logger.info(f"Found {total_files} files to download")
        
        for i, item in enumerate(response['Contents']):
            key = item['Key']
            filename = os.path.basename(key)
            
            # Determine destination path based on file type and name
            if filename.endswith('.jpg') or filename.endswith('.png'):
                if 'train' in key:
                    dest_path = os.path.join(local_dataset_path, 'images', 'train', filename)
                elif 'val' in key:
                    dest_path = os.path.join(local_dataset_path, 'images', 'val', filename)
                elif 'test' in key:
                    dest_path = os.path.join(local_dataset_path, 'images', 'test', filename)
                else:
                    # Default to train if not specified
                    dest_path = os.path.join(local_dataset_path, 'images', 'train', filename)
            elif filename.endswith('.txt') and not filename.endswith('classes.txt'):
                if 'train' in key:
                    dest_path = os.path.join(local_dataset_path, 'labels', 'train', filename)
                elif 'val' in key:
                    dest_path = os.path.join(local_dataset_path, 'labels', 'val', filename)
                elif 'test' in key:
                    dest_path = os.path.join(local_dataset_path, 'labels', 'test', filename)
                else:
                    # Default to train if not specified
                    dest_path = os.path.join(local_dataset_path, 'labels', 'train', filename)
            else:
                # Skip other files
                continue
            
            # Download file
            s3.download_file(bucket_name, key, dest_path)
            
            # Log progress every 100 files
            if (i + 1) % 100 == 0:
                logger.info(f"Downloaded {i + 1}/{total_files} files")
        
        logger.info(f"Successfully downloaded all data from S3")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading data from S3: {str(e)}")
        return False

def prepare_dataset_config(config_path, local_dataset_path, cyclist_focus=False):
    """Prepare dataset configuration for training."""
    logger.info(f"Preparing dataset configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update paths to point to local dataset
        config['path'] = local_dataset_path
        
        # If cyclist focus is enabled, adjust class weights
        if cyclist_focus and 'class_weights' in config:
            logger.info("Applying enhanced focus on cyclist class")
            cyclist_idx = None
            for idx, name in config['names'].items():
                if name.lower() == 'cyclist':
                    cyclist_idx = idx
                    break
            
            if cyclist_idx is not None:
                # Increase weight for cyclist class even more
                config['class_weights'][cyclist_idx] = 5.0
                logger.info(f"Increased weight for cyclist class (idx {cyclist_idx}) to 5.0")
        
        # Save updated config
        updated_config_path = os.path.join(os.path.dirname(config_path), 'road_objects_updated.yaml')
        with open(updated_config_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        logger.info(f"Updated dataset configuration saved to {updated_config_path}")
        return updated_config_path
    
    except Exception as e:
        logger.error(f"Error preparing dataset configuration: {str(e)}")
        return None

def train_yolov5(args, data_config_path):
    """Train YOLOv5 model with the specified configuration."""
    logger.info("Starting YOLOv5 training")
    
    try:
        # Import YOLOv5 modules (assuming YOLOv5 is installed or in PYTHONPATH)
        sys.path.append('/opt/yolov5')  # Adjust path if needed
        
        # Construct command for training
        cmd = [
            "python", "/opt/yolov5/train.py",
            "--data", data_config_path,
            "--hyp", args.hyp_config,
            "--weights", args.weights,
            "--img", str(args.img_size),
            "--batch", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--device", args.device,
            "--workers", str(args.workers),
            "--project", args.project,
            "--name", args.name,
        ]
        
        if args.exist_ok:
            cmd.append("--exist-ok")
        
        # Add cache for faster training
        cmd.extend(["--cache", "ram"])
        
        # Execute training command
        logger.info(f"Executing training command: {' '.join(cmd)}")
        
        # Use subprocess to run the command
        import subprocess
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output to log
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Training failed with return code {process.returncode}")
            return None
        
        # Return path to trained model
        exp_dir = os.path.join(args.project, args.name)
        model_path = os.path.join(exp_dir, 'weights', 'best.pt')
        
        if os.path.exists(model_path):
            logger.info(f"Training completed successfully. Model saved to {model_path}")
            return model_path
        else:
            logger.error(f"Training completed but model not found at {model_path}")
            return None
    
    except Exception as e:
        logger.error(f"Error during YOLOv5 training: {str(e)}")
        return None

def evaluate_model(model_path, data_config_path, img_size=640, batch_size=16, device='0'):
    """Evaluate trained model with focus on cyclist detection."""
    logger.info(f"Evaluating model {model_path}")
    
    try:
        # Import YOLOv5 modules
        sys.path.append('/opt/yolov5')  # Adjust path if needed
        
        # Construct command for validation
        cmd = [
            "python", "/opt/yolov5/val.py",
            "--data", data_config_path,
            "--weights", model_path,
            "--img", str(img_size),
            "--batch-size", str(batch_size),
            "--device", device,
            "--task", "test",
            "--verbose"
        ]
        
        # Execute validation command
        logger.info(f"Executing validation command: {' '.join(cmd)}")
        
        # Use subprocess to run the command
        import subprocess
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Capture output for parsing
        output = ""
        for line in process.stdout:
            logger.info(line.strip())
            output += line
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Evaluation failed with return code {process.returncode}")
            return None
        
        # Parse results to extract metrics for cyclist class
        results = {}
        
        # Extract overall mAP
        import re
        map_match = re.search(r'all\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+.\d+)', output)
        if map_match:
            results['mAP@.5'] = float(map_match.group(3))
            results['mAP@.5:.95'] = float(map_match.group(4))
        
        # Extract cyclist class metrics if available
        cyclist_match = re.search(r'cyclist\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+.\d+)', output, re.IGNORECASE)
        if cyclist_match:
            results['cyclist_precision'] = float(cyclist_match.group(2))
            results['cyclist_recall'] = float(cyclist_match.group(3))
            results['cyclist_mAP@.5'] = float(cyclist_match.group(4))
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        return None

def upload_model_to_s3(model_path, bucket_name, simulation_id):
    """Upload trained model to S3 bucket."""
    logger.info(f"Uploading model {model_path} to S3 bucket {bucket_name}")
    
    try:
        s3 = boto3.client('s3')
        
        # Define S3 key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"models/{simulation_id}/{timestamp}/best.pt"
        
        # Upload model
        s3.upload_file(model_path, bucket_name, s3_key)
        
        # Upload model metadata
        model_info = {
            'simulation_id': simulation_id,
            'timestamp': timestamp,
            'model_path': model_path,
            's3_path': f"s3://{bucket_name}/{s3_key}"
        }
        
        metadata_path = os.path.join(os.path.dirname(model_path), 'model_info.json')
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        metadata_s3_key = f"models/{simulation_id}/{timestamp}/model_info.json"
        s3.upload_file(metadata_path, bucket_name, metadata_s3_key)
        
        logger.info(f"Model successfully uploaded to s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
    
    except Exception as e:
        logger.error(f"Error uploading model to S3: {str(e)}")
        return None

def main():
    """Main function to run the training pipeline."""
    args = parse_args()
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_config):
        args.data_config = os.path.join(script_dir, args.data_config)
    if not os.path.isabs(args.hyp_config):
        args.hyp_config = os.path.join(script_dir, args.hyp_config)
    
    # Create local dataset path
    local_dataset_path = os.path.join(script_dir, '..', 'datasets', f'road_objects_{args.simulation_id}')
    os.makedirs(local_dataset_path, exist_ok=True)
    
    # Download data from S3
    if not download_data_from_s3(args.s3_bucket, args.simulation_id, local_dataset_path):
        logger.error("Failed to download data from S3. Exiting.")
        return 1
    
    # Prepare dataset configuration
    data_config_path = prepare_dataset_config(args.data_config, local_dataset_path, args.cyclist_focus)
    if not data_config_path:
        logger.error("Failed to prepare dataset configuration. Exiting.")
        return 1
    
    # Train YOLOv5 model
    model_path = train_yolov5(args, data_config_path)
    if not model_path:
        logger.error("Training failed. Exiting.")
        return 1
    
    # Evaluate model
    eval_results = evaluate_model(
        model_path, 
        data_config_path, 
        img_size=args.img_size, 
        batch_size=args.batch_size, 
        device=args.device
    )
    
    if not eval_results:
        logger.warning("Model evaluation failed or produced no results.")
    
    # Upload model to S3 if requested
    if not args.no_upload:
        s3_path = upload_model_to_s3(model_path, args.s3_bucket, args.simulation_id)
        if not s3_path:
            logger.warning("Failed to upload model to S3.")
    
    logger.info("Training pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
