#!/usr/bin/env python3
"""
model_utils.py - Utility functions for YOLOv5 model training and inference
with focus on road object detection and cyclist class improvement.

This module provides helper functions for:
1. Model loading and conversion
2. Class balancing and weighting
3. Specialized augmentation for cyclist class
4. Model performance analysis
5. Inference optimization
"""

import os
import sys
import yaml
import json
import numpy as np
import torch
import cv2
from pathlib import Path
import logging
import random
import shutil
from PIL import Image, ImageOps, ImageEnhance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_utils")

def load_model(model_path, device='cpu'):
    """
    Load a YOLOv5 model from path.
    
    Args:
        model_path (str): Path to the model file (.pt)
        device (str): Device to load model on ('cpu', '0', '0,1', etc.)
        
    Returns:
        model: Loaded YOLOv5 model
    """
    try:
        # Add YOLOv5 to path if needed
        sys.path.append('/opt/yolov5')
        
        # Import YOLOv5 modules
        from models.experimental import attempt_load
        
        # Load model
        device = torch.device(device if device != 'cpu' else 'cpu')
        model = attempt_load(model_path, device=device)
        
        logger.info(f"Model loaded from {model_path} to {device}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def export_model(model_path, output_format='onnx', img_size=640, batch_size=1, device='cpu'):
    """
    Export YOLOv5 model to different formats for deployment.
    
    Args:
        model_path (str): Path to the model file (.pt)
        output_format (str): Format to export to ('onnx', 'tflite', 'coreml', etc.)
        img_size (int): Image size for export
        batch_size (int): Batch size for export
        device (str): Device to use for export
        
    Returns:
        str: Path to exported model
    """
    try:
        # Construct export command
        cmd = [
            "python", "/opt/yolov5/export.py",
            "--weights", model_path,
            "--img", str(img_size),
            "--batch", str(batch_size),
            "--device", device,
            "--include", output_format
        ]
        
        # Execute export command
        logger.info(f"Exporting model to {output_format}: {' '.join(cmd)}")
        
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
            logger.error(f"Export failed with return code {process.returncode}")
            return None
        
        # Determine output path based on format
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        
        if output_format == 'onnx':
            output_path = os.path.join(model_dir, f"{model_name}.onnx")
        elif output_format == 'tflite':
            output_path = os.path.join(model_dir, f"{model_name}.tflite")
        elif output_format == 'coreml':
            output_path = os.path.join(model_dir, f"{model_name}.mlmodel")
        else:
            output_path = os.path.join(model_dir, f"{model_name}.{output_format}")
        
        if os.path.exists(output_path):
            logger.info(f"Model exported successfully to {output_path}")
            return output_path
        else:
            logger.error(f"Export completed but model not found at {output_path}")
            return None
    
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        return None

def apply_cyclist_augmentation(image_path, label_path, output_dir, num_augmentations=5):
    """
    Apply specialized augmentations to cyclist instances in images.
    
    Args:
        image_path (str): Path to the image file
        label_path (str): Path to the corresponding label file (YOLO format)
        output_dir (str): Directory to save augmented images and labels
        num_augmentations (int): Number of augmented versions to create
        
    Returns:
        list: Paths to augmented images
    """
    try:
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        
        # Load image
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        # Load labels
        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
        
        # Check if image contains cyclists
        cyclist_indices = []
        for i, label in enumerate(labels):
            if len(label) >= 5:  # Ensure label has class and bbox coordinates
                class_id = int(label[0])
                # Check if class is cyclist (assuming class ID 8 for cyclist)
                if class_id == 8:
                    cyclist_indices.append(i)
        
        # If no cyclists, return empty list
        if not cyclist_indices:
            logger.debug(f"No cyclists found in {image_path}")
            return []
        
        augmented_paths = []
        
        # Apply augmentations
        for i in range(num_augmentations):
            # Create a copy of the image for augmentation
            aug_image = image.copy()
            
            # Apply random augmentations
            # 1. Color jitter
            aug_image = ImageEnhance.Brightness(aug_image).enhance(random.uniform(0.8, 1.2))
            aug_image = ImageEnhance.Contrast(aug_image).enhance(random.uniform(0.8, 1.2))
            aug_image = ImageEnhance.Color(aug_image).enhance(random.uniform(0.8, 1.2))
            
            # 2. Random rotation (small angles to avoid distorting bounding boxes too much)
            angle = random.uniform(-10, 10)
            aug_image = aug_image.rotate(angle, resample=Image.BICUBIC, expand=False)
            
            # 3. Random scale (zoom in/out)
            scale = random.uniform(0.9, 1.1)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            aug_image = aug_image.resize((new_width, new_height), Image.BICUBIC)
            
            # Resize back to original dimensions
            aug_image = aug_image.resize((img_width, img_height), Image.BICUBIC)
            
            # Save augmented image
            image_name = os.path.basename(image_path)
            aug_image_name = f"{os.path.splitext(image_name)[0]}_aug{i}{os.path.splitext(image_name)[1]}"
            aug_image_path = os.path.join(output_dir, 'images', aug_image_name)
            aug_image.save(aug_image_path)
            
            # Save augmented labels (same as original for now, as we're not modifying bounding boxes)
            label_name = os.path.basename(label_path)
            aug_label_name = f"{os.path.splitext(label_name)[0]}_aug{i}{os.path.splitext(label_name)[1]}"
            aug_label_path = os.path.join(output_dir, 'labels', aug_label_name)
            shutil.copy(label_path, aug_label_path)
            
            augmented_paths.append(aug_image_path)
        
        logger.info(f"Created {len(augmented_paths)} augmented versions of {image_path} with cyclists")
        return augmented_paths
    
    except Exception as e:
        logger.error(f"Error applying cyclist augmentation: {str(e)}")
        return []

def balance_dataset(dataset_dir, output_dir, target_class_id=8, multiplier=3):
    """
    Balance dataset by duplicating and augmenting underrepresented classes.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        output_dir (str): Directory to save balanced dataset
        target_class_id (int): Class ID to focus on (e.g., 8 for cyclist)
        multiplier (int): Number of times to duplicate target class instances
        
    Returns:
        dict: Statistics of original and balanced dataset
    """
    try:
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
        
        # Copy validation and test sets as is
        for subset in ['val', 'test']:
            if os.path.exists(os.path.join(dataset_dir, 'images', subset)):
                os.makedirs(os.path.join(output_dir, 'images', subset), exist_ok=True)
                os.makedirs(os.path.join(output_dir, 'labels', subset), exist_ok=True)
                
                # Copy images
                for img in os.listdir(os.path.join(dataset_dir, 'images', subset)):
                    shutil.copy(
                        os.path.join(dataset_dir, 'images', subset, img),
                        os.path.join(output_dir, 'images', subset, img)
                    )
                
                # Copy labels
                for lbl in os.listdir(os.path.join(dataset_dir, 'labels', subset)):
                    shutil.copy(
                        os.path.join(dataset_dir, 'labels', subset, lbl),
                        os.path.join(output_dir, 'labels', subset, lbl)
                    )
        
        # Process training set
        train_images_dir = os.path.join(dataset_dir, 'images', 'train')
        train_labels_dir = os.path.join(dataset_dir, 'labels', 'train')
        
        # Count instances of each class
        class_counts = {}
        images_with_target = []
        
        for label_file in os.listdir(train_labels_dir):
            label_path = os.path.join(train_labels_dir, label_file)
            image_file = os.path.splitext(label_file)[0] + '.jpg'  # Assuming .jpg extension
            image_path = os.path.join(train_images_dir, image_file)
            
            if not os.path.exists(image_path):
                # Try .png extension
                image_file = os.path.splitext(label_file)[0] + '.png'
                image_path = os.path.join(train_images_dir, image_file)
                
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found for label {label_file}")
                    continue
            
            # Copy original image and label to output directory
            shutil.copy(
                image_path,
                os.path.join(output_dir, 'images', 'train', os.path.basename(image_path))
            )
            shutil.copy(
                label_path,
                os.path.join(output_dir, 'labels', 'train', os.path.basename(label_path))
            )
            
            # Count class instances
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]
            
            has_target = False
            for label in labels:
                if len(label) >= 5:  # Ensure label has class and bbox coordinates
                    class_id = int(label[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    
                    if class_id == target_class_id:
                        has_target = True
            
            if has_target:
                images_with_target.append((image_path, label_path))
        
        # Augment target class instances
        for i, (image_path, label_path) in enumerate(images_with_target):
            # Apply augmentation to create multiple versions
            apply_cyclist_augmentation(
                image_path, 
                label_path, 
                output_dir, 
                num_augmentations=multiplier
            )
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(images_with_target)} images with target class")
        
        # Calculate statistics
        balanced_class_counts = class_counts.copy()
        balanced_class_counts[target_class_id] = balanced_class_counts.get(target_class_id, 0) * (multiplier + 1)
        
        stats = {
            'original': {
                'total_images': len(os.listdir(train_images_dir)),
                'class_counts': class_counts,
                'target_class_count': class_counts.get(target_class_id, 0),
                'images_with_target': len(images_with_target)
            },
            'balanced': {
                'total_images': len(os.listdir(os.path.join(output_dir, 'images', 'train'))),
                'class_counts': balanced_class_counts,
                'target_class_count': balanced_class_counts.get(target_class_id, 0)
            }
        }
        
        # Save statistics
        stats_path = os.path.join(output_dir, 'balance_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset balancing completed. Statistics saved to {stats_path}")
        return stats
    
    except Exception as e:
        logger.error(f"Error balancing dataset: {str(e)}")
        return None

def optimize_model_for_inference(model_path, output_format='onnx', quantize=True, img_size=640):
    """
    Optimize YOLOv5 model for inference by exporting to efficient formats and quantizing.
    
    Args:
        model_path (str): Path to the model file (.pt)
        output_format (str): Format to export to ('onnx', 'tflite', 'coreml', etc.)
        quantize (bool): Whether to quantize the model
        img_size (int): Image size for export
        
    Returns:
        str: Path to optimized model
    """
    try:
        # Export model to specified format
        exported_model = export_model(
            model_path, 
            output_format=output_format, 
            img_size=img_size
        )
        
        if not exported_model:
            return None
        
        # Quantize model if requested
        if quantize and output_format == 'onnx':
            # Import onnx and onnxruntime
            try:
                import onnx
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                # Load model
                model = onnx.load(exported_model)
                
                # Quantize model
                quantized_model = os.path.splitext(exported_model)[0] + '_quantized.onnx'
                quantize_dynamic(
                    exported_model,
                    quantized_model,
                    weight_type=QuantType.QUInt8
                )
                
                logger.info(f"Model quantized successfully to {quantized_model}")
                return quantized_model
            
            except ImportError:
                logger.warning("onnx or onnxruntime not installed. Skipping quantization.")
                return exported_model
        
        return exported_model
    
    except Exception as e:
        logger.error(f"Error optimizing model for inference: {str(e)}")
        return None

def analyze_model_size_and_speed(model_path, batch_size=1, img_size=640, num_iterations=100):
    """
    Analyze model size and inference speed.
    
    Args:
        model_path (str): Path to the model file
        batch_size (int): Batch size for inference
        img_size (int): Image size for inference
        num_iterations (int): Number of iterations for speed test
        
    Returns:
        dict: Model size and speed statistics
    """
    try:
        # Get model file size
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Load model for speed test
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path.endswith('.pt'):
            # PyTorch model
            model = load_model(model_path, device=str(device))
            
            # Create dummy input
            dummy_input = torch.zeros((batch_size, 3, img_size, img_size), device=device)
            
            # Warm-up
            for _ in range(10):
                _ = model(dummy_input)
            
            # Measure inference time
            torch.cuda.synchronize() if device.type != 'cpu' else None
            start_time = torch.cuda.Event(enable_timing=True) if device.type != 'cpu' else time.time()
            end_time = torch.cuda.Event(enable_timing=True) if device.type != 'cpu' else None
            
            if device.type != 'cpu':
                start_time.record()
            
            for _ in range(num_iterations):
                _ = model(dummy_input)
            
            if device.type != 'cpu':
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            else:
                elapsed_time = time.time() - start_time
            
            inference_time = elapsed_time / num_iterations
            fps = 1 / inference_time * batch_size
        
        elif model_path.endswith('.onnx'):
            # ONNX model
            import onnxruntime as ort
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(model_path)
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Create dummy input
            dummy_input = np.zeros((batch_size, 3, img_size, img_size), dtype=np.float32)
            
            # Warm-up
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input})
            
            # Measure inference time
            import time
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = session.run(None, {input_name: dummy_input})
            
            elapsed_time = time.time() - start_time
            inference_time = elapsed_time / num_iterations
            fps = 1 / inference_time * batch_size
        
        else:
            logger.error(f"Unsupported model format: {model_path}")
            return None
        
        # Compile statistics
        stats = {
            'model_path': model_path,
            'model_format': os.path.splitext(model_path)[1],
            'model_size_bytes': model_size_bytes,
            'model_size_mb': model_size_mb,
            'device': str(device),
            'batch_size': batch_size,
            'img_size': img_size,
            'inference_time_ms': inference_time * 1000,
            'fps': fps
        }
        
        logger.info(f"Model analysis completed: {stats}")
        return stats
    
    except Exception as e:
        logger.error(f"Error analyzing model size and speed: {str(e)}")
        return None

# Additional utility functions can be added here as needed
