#!/usr/bin/env python3
"""
inference.py - Script for running inference with trained YOLOv5 models
on road object detection data with focus on cyclist detection.

This script handles:
1. Loading a trained model
2. Running inference on images or video
3. Visualizing detection results
4. Analyzing detection performance for cyclists
5. Saving results to specified output directory
"""

import argparse
import os
import sys
import yaml
import json
import numpy as np
import cv2
import torch
from pathlib import Path
import logging
import time
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("inference")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with YOLOv5 model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLOv5 model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video, or directory')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for inference')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='',
                        help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidences in --save-txt labels')
    parser.add_argument('--cyclist-focus', action='store_true',
                        help='Focus on cyclist detections in output')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='Filter by class: --classes 0, or --classes 0 2 3')
    
    return parser.parse_args()

def run_inference(args):
    """Run inference with YOLOv5 detect.py script."""
    logger.info(f"Running inference with model {args.model} on {args.source}")
    
    try:
        # Import YOLOv5 modules (assuming YOLOv5 is installed or in PYTHONPATH)
        sys.path.append('/opt/yolov5')  # Adjust path if needed
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Construct command for inference
        cmd = [
            "python", "/opt/yolov5/detect.py",
            "--weights", args.model,
            "--source", args.source,
            "--img-size", str(args.img_size),
            "--conf-thres", str(args.conf_thres),
            "--iou-thres", str(args.iou_thres),
            "--project", args.output_dir,
            "--name", "exp",
            "--exist-ok"
        ]
        
        if args.device:
            cmd.extend(["--device", args.device])
        
        if args.save_txt:
            cmd.append("--save-txt")
        
        if args.save_conf:
            cmd.append("--save-conf")
        
        if args.classes is not None:
            cmd.extend(["--classes"] + [str(c) for c in args.classes])
        
        # Execute inference command
        logger.info(f"Executing inference command: {' '.join(cmd)}")
        
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
            logger.error(f"Inference failed with return code {process.returncode}")
            return None
        
        # Determine output path
        output_path = os.path.join(args.output_dir, "exp")
        
        logger.info(f"Inference completed. Results saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return None

def analyze_cyclist_detections(output_path, conf_threshold=0.25):
    """Analyze cyclist detections in inference results."""
    if not output_path or not os.path.exists(output_path):
        logger.error(f"Output path {output_path} does not exist")
        return None
    
    try:
        # Check for labels directory
        labels_dir = os.path.join(output_path, "labels")
        if not os.path.exists(labels_dir):
            logger.warning(f"No labels directory found at {labels_dir}")
            return None
        
        # Analyze detection results
        cyclist_detections = []
        total_images = 0
        images_with_cyclists = 0
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue
            
            total_images += 1
            label_path = os.path.join(labels_dir, label_file)
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            image_cyclists = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6:  # class, x, y, w, h, conf
                    class_id = int(parts[0])
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    
                    # Check if class is cyclist (assuming class ID 8 for cyclist)
                    if class_id == 8 and conf >= conf_threshold:
                        x, y, w, h = map(float, parts[1:5])
                        image_cyclists.append({
                            'confidence': conf,
                            'bbox': [x, y, w, h]
                        })
            
            if image_cyclists:
                images_with_cyclists += 1
                cyclist_detections.append({
                    'image': os.path.splitext(label_file)[0],
                    'detections': image_cyclists,
                    'count': len(image_cyclists)
                })
        
        # Compile statistics
        stats = {
            'total_images': total_images,
            'images_with_cyclists': images_with_cyclists,
            'cyclist_detection_rate': images_with_cyclists / total_images if total_images > 0 else 0,
            'total_cyclist_detections': sum(d['count'] for d in cyclist_detections),
            'average_cyclists_per_image': sum(d['count'] for d in cyclist_detections) / images_with_cyclists if images_with_cyclists > 0 else 0,
            'average_confidence': sum(d['confidence'] for img in cyclist_detections for d in img['detections']) / sum(d['count'] for d in cyclist_detections) if sum(d['count'] for d in cyclist_detections) > 0 else 0
        }
        
        # Save statistics
        stats_path = os.path.join(output_path, 'cyclist_detection_stats.json')
        with open(stats_path, 'w') as f:
            json.dump({
                'statistics': stats,
                'detections': cyclist_detections
            }, f, indent=2)
        
        logger.info(f"Cyclist detection analysis completed. Statistics saved to {stats_path}")
        return stats
    
    except Exception as e:
        logger.error(f"Error analyzing cyclist detections: {str(e)}")
        return None

def highlight_cyclists_in_results(output_path, cyclist_class_id=8):
    """Highlight cyclist detections in output images."""
    if not output_path or not os.path.exists(output_path):
        logger.error(f"Output path {output_path} does not exist")
        return False
    
    try:
        # Create directory for highlighted images
        highlight_dir = os.path.join(output_path, "highlighted_cyclists")
        os.makedirs(highlight_dir, exist_ok=True)
        
        # Get list of output images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        for root, _, files in os.walk(output_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    # Skip images in the highlighted directory
                    if "highlighted_cyclists" not in root:
                        images.append(os.path.join(root, file))
        
        if not images:
            logger.warning(f"No images found in {output_path}")
            return False
        
        # Process each image
        for image_path in images:
            # Check if corresponding label file exists
            label_path = os.path.join(
                output_path, 
                "labels", 
                os.path.splitext(os.path.basename(image_path))[0] + ".txt"
            )
            
            if not os.path.exists(label_path):
                continue
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image {image_path}")
                continue
            
            height, width = image.shape[:2]
            
            # Load labels
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            has_cyclists = False
            
            # Draw bounding boxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    
                    # Check if class is cyclist
                    if class_id == cyclist_class_id:
                        has_cyclists = True
                        
                        # Parse normalized coordinates
                        x_center, y_center, w, h = map(float, parts[1:5])
                        
                        # Convert to pixel coordinates
                        x1 = int((x_center - w/2) * width)
                        y1 = int((y_center - h/2) * height)
                        x2 = int((x_center + w/2) * width)
                        y2 = int((y_center + h/2) * height)
                        
                        # Draw highlighted box for cyclists
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Add label
                        conf = float(parts[5]) if len(parts) >= 6 else 1.0
                        label = f"Cyclist {conf:.2f}"
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save image if it contains cyclists
            if has_cyclists:
                output_image_path = os.path.join(
                    highlight_dir, 
                    os.path.basename(image_path)
                )
                cv2.imwrite(output_image_path, image)
        
        logger.info(f"Cyclist highlighting completed. Results saved to {highlight_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error highlighting cyclists: {str(e)}")
        return False

def main():
    """Main function to run the inference pipeline."""
    args = parse_args()
    
    # Run inference
    output_path = run_inference(args)
    if not output_path:
        logger.error("Inference failed. Exiting.")
        return 1
    
    # Analyze cyclist detections if requested
    if args.cyclist_focus:
        stats = analyze_cyclist_detections(output_path, args.conf_thres)
        if not stats:
            logger.warning("Failed to analyze cyclist detections.")
        
        # Highlight cyclists in output images
        if not highlight_cyclists_in_results(output_path):
            logger.warning("Failed to highlight cyclists in output images.")
    
    logger.info("Inference pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
