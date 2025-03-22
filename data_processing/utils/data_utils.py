#!/usr/bin/env python

"""
Road Object Detection - Data Processing Utilities
This module provides utility functions for the data processing pipeline
"""

import os
import sys
import json
import logging
import numpy as np
import cv2
import shutil
import boto3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Configure logging
logger = logging.getLogger("DataUtils")

class DataUtils:
    """
    Utility functions for data processing
    """
    @staticmethod
    def create_dataset_statistics(input_dir, output_file, annotation_format=None):
        """
        Create statistics about a dataset
        
        Args:
            input_dir: Input directory containing the dataset
            output_file: Output file to save statistics
            annotation_format: Format of annotations (yolo, coco, pascal_voc)
        
        Returns:
            Dictionary containing statistics
        """
        logger.info(f"Creating statistics for dataset in {input_dir}")
        
        # Detect annotation format if not provided
        if annotation_format is None:
            annotation_format = DataUtils._detect_annotation_format(input_dir)
            if not annotation_format:
                logger.error(f"Could not detect annotation format in {input_dir}")
                return None
        
        # Get statistics based on format
        if annotation_format == 'yolo':
            stats = DataUtils._get_yolo_statistics(input_dir)
        elif annotation_format == 'coco':
            stats = DataUtils._get_coco_statistics(input_dir)
        elif annotation_format == 'pascal_voc':
            stats = DataUtils._get_pascal_voc_statistics(input_dir)
        else:
            logger.error(f"Unsupported annotation format: {annotation_format}")
            return None
        
        # Save statistics to file
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {output_file}")
        
        return stats
    
    @staticmethod
    def visualize_dataset_statistics(stats_file, output_dir):
        """
        Visualize dataset statistics
        
        Args:
            stats_file: JSON file containing dataset statistics
            output_dir: Directory to save visualizations
        
        Returns:
            List of generated visualization files
        """
        logger.info(f"Visualizing statistics from {stats_file}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load statistics
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        visualization_files = []
        
        # Class distribution
        if 'class_counts' in stats:
            plt.figure(figsize=(12, 8))
            sns.barplot(x=list(stats['class_counts'].keys()), y=list(stats['class_counts'].values()))
            plt.title('Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            class_dist_file = os.path.join(output_dir, 'class_distribution.png')
            plt.savefig(class_dist_file)
            plt.close()
            
            visualization_files.append(class_dist_file)
        
        # Image size distribution
        if 'image_sizes' in stats:
            widths = [size[0] for size in stats['image_sizes']]
            heights = [size[1] for size in stats['image_sizes']]
            
            plt.figure(figsize=(12, 8))
            plt.scatter(widths, heights, alpha=0.5)
            plt.title('Image Size Distribution')
            plt.xlabel('Width')
            plt.ylabel('Height')
            plt.grid(True)
            plt.tight_layout()
            
            size_dist_file = os.path.join(output_dir, 'size_distribution.png')
            plt.savefig(size_dist_file)
            plt.close()
            
            visualization_files.append(size_dist_file)
        
        # Objects per image distribution
        if 'objects_per_image' in stats:
            plt.figure(figsize=(12, 8))
            sns.histplot(stats['objects_per_image'], kde=True)
            plt.title('Objects per Image Distribution')
            plt.xlabel('Number of Objects')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            
            obj_dist_file = os.path.join(output_dir, 'objects_distribution.png')
            plt.savefig(obj_dist_file)
            plt.close()
            
            visualization_files.append(obj_dist_file)
        
        # Class-specific statistics
        if 'class_specific' in stats:
            for class_name, class_stats in stats['class_specific'].items():
                if 'bbox_sizes' in class_stats:
                    widths = [size[0] for size in class_stats['bbox_sizes']]
                    heights = [size[1] for size in class_stats['bbox_sizes']]
                    
                    plt.figure(figsize=(12, 8))
                    plt.scatter(widths, heights, alpha=0.5)
                    plt.title(f'Bounding Box Size Distribution for {class_name}')
                    plt.xlabel('Width')
                    plt.ylabel('Height')
                    plt.grid(True)
                    plt.tight_layout()
                    
                    bbox_dist_file = os.path.join(output_dir, f'{class_name}_bbox_distribution.png')
                    plt.savefig(bbox_dist_file)
                    plt.close()
                    
                    visualization_files.append(bbox_dist_file)
        
        logger.info(f"Visualizations saved to {output_dir}")
        
        return visualization_files
    
    @staticmethod
    def upload_dataset_to_s3(input_dir, bucket_name, prefix, include_visualizations=True):
        """
        Upload dataset to S3
        
        Args:
            input_dir: Input directory containing the dataset
            bucket_name: S3 bucket name
            prefix: S3 prefix
            include_visualizations: Whether to include visualizations
        
        Returns:
            Success status and message
        """
        logger.info(f"Uploading dataset from {input_dir} to S3 bucket {bucket_name}/{prefix}")
        
        try:
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            # Create statistics and visualizations if requested
            if include_visualizations:
                stats_file = os.path.join(input_dir, 'statistics.json')
                vis_dir = os.path.join(input_dir, 'visualizations')
                
                # Create statistics if not exists
                if not os.path.exists(stats_file):
                    stats = DataUtils.create_dataset_statistics(input_dir, stats_file)
                    if stats:
                        DataUtils.visualize_dataset_statistics(stats_file, vis_dir)
            
            # Walk through directory and upload files
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    s3_key = os.path.join(prefix, os.path.relpath(local_path, input_dir))
                    
                    # Upload file
                    s3_client.upload_file(local_path, bucket_name, s3_key)
                    logger.debug(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
            
            logger.info(f"Dataset uploaded successfully to S3 bucket {bucket_name}/{prefix}")
            return {'success': True, 'message': f"Dataset uploaded successfully to S3 bucket {bucket_name}/{prefix}"}
            
        except Exception as e:
            error_msg = f"Error uploading dataset to S3: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    @staticmethod
    def download_dataset_from_s3(bucket_name, prefix, output_dir):
        """
        Download dataset from S3
        
        Args:
            bucket_name: S3 bucket name
            prefix: S3 prefix
            output_dir: Output directory to save dataset
        
        Returns:
            Success status and message
        """
        logger.info(f"Downloading dataset from S3 bucket {bucket_name}/{prefix} to {output_dir}")
        
        try:
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # List objects in bucket with prefix
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            # Download each object
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        local_path = os.path.join(output_dir, os.path.relpath(s3_key, prefix))
                        
                        # Create directory if not exists
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        
                        # Download file
                        s3_client.download_file(bucket_name, s3_key, local_path)
                        logger.debug(f"Downloaded s3://{bucket_name}/{s3_key} to {local_path}")
            
            logger.info(f"Dataset downloaded successfully from S3 bucket {bucket_name}/{prefix}")
            return {'success': True, 'message': f"Dataset downloaded successfully from S3 bucket {bucket_name}/{prefix}"}
            
        except Exception as e:
            error_msg = f"Error downloading dataset from S3: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    @staticmethod
    def _detect_annotation_format(input_dir):
        """Detect the annotation format in the input directory"""
        # Check for YOLO format
        if os.path.exists(os.path.join(input_dir, 'labels')) or os.path.exists(os.path.join(input_dir, 'yolo')):
            return 'yolo'
        
        # Check for COCO format
        coco_files = [f for f in os.listdir(input_dir) if f.endswith('.json') and 'coco' in f.lower()]
        if coco_files or os.path.exists(os.path.join(input_dir, 'annotations')) and any(f.endswith('.json') for f in os.listdir(os.path.join(input_dir, 'annotations'))):
            return 'coco'
        
        # Check for Pascal VOC format
        if os.path.exists(os.path.join(input_dir, 'Annotations')) or os.path.exists(os.path.join(input_dir, 'pascal_voc')):
            return 'pascal_voc'
        
        # Try to infer from file extensions
        annotation_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt') and not file == 'classes.txt':
                    annotation_files.append(('yolo', os.path.join(root, file)))
                elif file.endswith('.xml'):
                    annotation_files.append(('pascal_voc', os.path.join(root, file)))
                elif file.endswith('.json') and not file.startswith('.'):
                    with open(os.path.join(root, file), 'r') as f:
                        try:
                            content = json.load(f)
                            if 'annotations' in content or 'categories' in content:
                                annotation_files.append(('coco', os.path.join(root, file)))
                        except:
                            pass
        
        # Count formats
        format_counts = {'yolo': 0, 'coco': 0, 'pascal_voc': 0}
        for fmt, _ in annotation_files:
            format_counts[fmt] += 1
        
        # Return the most common format
        if format_counts['yolo'] > format_counts['coco'] and format_counts['yolo'] > format_counts['pascal_voc']:
            return 'yolo'
        elif format_counts['coco'] > format_counts['yolo'] and format_counts['coco'] > format_counts['pascal_voc']:
            return 'coco'
        elif format_counts['pascal_voc'] > format_counts['yolo'] and format_counts['pascal_voc'] > format_counts['coco']:
            return 'pascal_voc'
        
        return None
    
    @staticmethod
    def _get_yolo_statistics(input_dir):
        """Get statistics for YOLO format dataset"""
        # Find images and labels directories
        images_dir = None
        labels_dir = None
        
        # Check common directory structures
        if os.path.exists(os.path.join(input_dir, 'images')):
            images_dir = os.path.join(input_dir, 'images')
        elif os.path.exists(os.path.join(input_dir, 'rgb')):
            images_dir = os.path.join(input_dir, 'rgb')
        else:
            # Look for directory with image files
            for root, dirs, files in os.walk(input_dir):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in files):
                    images_dir = root
                    break
        
        if os.path.exists(os.path.join(input_dir, 'labels')):
            labels_dir = os.path.join(input_dir, 'labels')
        elif os.path.exists(os.path.join(input_dir, 'yolo')):
            labels_dir = os.path.join(input_dir, 'yolo')
        else:
            # Look for directory with label files
            for root, dirs, files in os.walk(input_dir):
                if any(f.lower().endswith('.txt') and not f == 'classes.txt' for f in files):
                    labels_dir = root
                    break
        
        if not images_dir or not labels_dir:
            logger.error(f"Could not find images and labels directories in {input_dir}")
            return None
        
        # Find class mapping
        classes_file = None
        if os.path.exists(os.path.join(input_dir, 'classes.txt')):
            classes_file = os.path.join(input_dir, 'classes.txt')
        elif os.path.exists(os.path.join(labels_dir, 'classes.txt')):
            classes_file = os.path.join(labels_dir, 'classes.txt')
        
        class_names = []
        if classes_file:
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        
        # Initialize statistics
        stats = {
            'dataset_format': 'yolo',
            'total_images': 0,
            'total_annotations': 0,
            'class_counts': {},
            'image_sizes': [],
            'objects_per_image': [],
            'class_specific': {}
        }
        
        # Get all image files
        image_files = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        stats['total_images'] = len(image_files)
        
        # Process each image and its annotation
        for image_file in image_files:
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            label_file = os.path.join(labels_dir, f"{image_name}.txt")
            
            # Check if label file exists
            if not os.path.exists(label_file):
                continue
            
            # Get image size
            try:
                img = Image.open(image_file)
                width, height = img.size
                stats['image_sizes'].append([width, height])
            except:
                continue
            
            # Process annotations
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                stats['objects_per_image'].append(len(lines))
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                    
                    # Get class name
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    
                    # Update class counts
                    stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
                    
                    # Update class-specific statistics
                    if class_name not in stats['class_specific']:
                        stats['class_specific'][class_name] = {
                            'count': 0,
                            'bbox_sizes': []
                        }
                    
                    stats['class_specific'][class_name]['count'] += 1
                    
                    # Convert relative bbox to absolute
                    abs_width = bbox_width * width
                    abs_height = bbox_height * height
                    stats['class_specific'][class_name]['bbox_sizes'].append([abs_width, abs_height])
                    
                    stats['total_annotations'] += 1
                    
            except:
                continue
        
        # Calculate average objects per image
        if stats['objects_per_image']:
            stats['avg_objects_per_image'] = sum(stats['objects_per_image']) / len(stats['objects_per_image'])
        
        # Calculate average image size
        if stats['image_sizes']:
            avg_width = sum(size[0] for size in stats['image_sizes']) / len(stats['image_sizes'])
            avg_height = sum(size[1] for size in stats['image_sizes']) / len(stats['image_sizes'])
            stats['avg_image_size'] = [avg_width, avg_height]
        
        # Calculate class-specific averages
        for class_name, class_stats in stats['class_specific'].items():
            if class_stats['bbox_sizes']:
                avg_width = sum(size[0] for size in class_stats['bbox_sizes']) / len(class_stats['bbox_sizes'])
                avg_height = sum(size[1] for size in class_stats['bbox_sizes']) / len(class_stats['bbox_sizes'])
                class_stats['avg_bbox_size'] = [avg_width, avg_height]
        
        return stats
    
    @staticmethod
    def _get_coco_statistics(input_dir):
        """Get statistics for COCO format dataset"""
        # Find COCO annotation file
        coco_file = None
        
        # Check common locations
        if os.path.exists(os.path.join(input_dir, 'annotations', 'instances_default.json')):
            coco_file = os.path.join(input_dir, 'annotations', 'instances_default.json')
        elif os.path.exists(os.path.join(input_dir, 'annotations.json')):
            coco_file = os.path.join(input_dir, 'annotations.json')
        else:
            # Look for JSON files that might be COCO annotations
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.endswith('.json'):
                        try:
                            with open(os.path.join(root, file), 'r') as f:
                                data = json.load(f)
                                if 'images' in data and 'annotations' in data and 'categories' in data:
                                    coco_file = os.path.join(root, file)
                                    break
                        except:
                            continue
                if coco_file:
                    break
        
        if not coco_file:
            logger.error(f"Could not find COCO annotation file in {input_dir}")
            return None
        
        # Load COCO annotations
        try:
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Check required fields
            if not all(key in coco_data for key in ['images', 'annotations', 'categories']):
                logger.error(f"Invalid COCO format in {coco_file}: missing required fields")
                return None
            
            # Initialize statistics
            stats = {
                'dataset_format': 'coco',
                'total_images': len(coco_data['images']),
                'total_annotations': len(coco_data['annotations']),
                'class_counts': {},
                'image_sizes': [],
                'objects_per_image': [],
                'class_specific': {}
            }
            
            # Create category ID to name mapping
            category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
            
            # Create image ID to annotations mapping
            image_to_annotations = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in image_to_annotations:
                    image_to_annotations[image_id] = []
                image_to_annotations[image_id].append(ann)
            
            # Process each image
            for img in coco_data['images']:
                image_id = img['id']
                width = img['width']
                height = img['height']
                
                stats['image_sizes'].append([width, height])
                
                # Get annotations for this image
                annotations = image_to_annotations.get(image_id, [])
                stats['objects_per_image'].append(len(annotations))
                
                # Process annotations
                for ann in annotations:
                    category_id = ann['category_id']
                    bbox = ann['bbox']  # [x, y, width, height] in COCO format
                    
                    # Get category name
                    category_name = category_map.get(category_id, f"category_{category_id}")
                    
                    # Update class counts
                    stats['class_counts'][category_name] = stats['class_counts'].get(category_name, 0) + 1
                    
                    # Update class-specific statistics
                    if category_name not in stats['class_specific']:
                        stats['class_specific'][category_name] = {
                            'count': 0,
                            'bbox_sizes': []
                        }
                    
                    stats['class_specific'][category_name]['count'] += 1
                    stats['class_specific'][category_name]['bbox_sizes'].append([bbox[2], bbox[3]])
            
            # Calculate average objects per image
            if stats['objects_per_image']:
                stats['avg_objects_per_image'] = sum(stats['objects_per_image']) / len(stats['objects_per_image'])
            
            # Calculate average image size
            if stats['image_sizes']:
                avg_width = sum(size[0] for size in stats['image_sizes']) / len(stats['image_sizes'])
                avg_height = sum(size[1] for size in stats['image_sizes']) / len(stats['image_sizes'])
                stats['avg_image_size'] = [avg_width, avg_height]
            
            # Calculate class-specific averages
            for class_name, class_stats in stats['class_specific'].items():
                if class_stats['bbox_sizes']:
                    avg_width = sum(size[0] for size in class_stats['bbox_sizes']) / len(class_stats['bbox_sizes'])
                    avg_height = sum(size[1] for size in class_stats['bbox_sizes']) / len(class_stats['bbox_sizes'])
                    class_stats['avg_bbox_size'] = [avg_width, avg_height]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing COCO annotations: {str(e)}")
            return None
    
    @staticmethod
    def _get_pascal_voc_statistics(input_dir):
        """Get statistics for Pascal VOC format dataset"""
        # Find annotations and images directories
        annotations_dir = None
        images_dir = None
        
        # Check common directory structures
        if os.path.exists(os.path.join(input_dir, 'Annotations')):
            annotations_dir = os.path.join(input_dir, 'Annotations')
        elif os.path.exists(os.path.join(input_dir, 'pascal_voc')):
            annotations_dir = os.path.join(input_dir, 'pascal_voc')
        else:
            # Look for directory with XML files
            for root, dirs, files in os.walk(input_dir):
                if any(f.lower().endswith('.xml') for f in files):
                    annotations_dir = root
                    break
        
        if os.path.exists(os.path.join(input_dir, 'JPEGImages')):
            images_dir = os.path.join(input_dir, 'JPEGImages')
        elif os.path.exists(os.path.join(input_dir, 'images')):
            images_dir = os.path.join(input_dir, 'images')
        elif os.path.exists(os.path.join(input_dir, 'rgb')):
            images_dir = os.path.join(input_dir, 'rgb')
        else:
            # Look for directory with image files
            for root, dirs, files in os.walk(input_dir):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in files):
                    images_dir = root
                    break
        
        if not annotations_dir:
            logger.error(f"Could not find annotations directory in {input_dir}")
            return None
        
        # Initialize statistics
        stats = {
            'dataset_format': 'pascal_voc',
            'total_images': 0,
            'total_annotations': 0,
            'class_counts': {},
            'image_sizes': [],
            'objects_per_image': [],
            'class_specific': {}
        }
        
        # Get all annotation files
        annotation_files = []
        for root, _, files in os.walk(annotations_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    annotation_files.append(os.path.join(root, file))
        
        stats['total_images'] = len(annotation_files)
        
        # Process each annotation
        for annotation_file in annotation_files:
            try:
                # Parse XML
                tree = ET.parse(annotation_file)
                xml_root = tree.getroot()
                
                # Get image size
                size_elem = xml_root.find('size')
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
                
                stats['image_sizes'].append([width, height])
                
                # Get objects
                objects = xml_root.findall('object')
                stats['objects_per_image'].append(len(objects))
                
                # Process objects
                for obj in objects:
                    class_name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    
                    # Get bounding box coordinates
                    x_min = float(bbox.find('xmin').text)
                    y_min = float(bbox.find('ymin').text)
                    x_max = float(bbox.find('xmax').text)
                    y_max = float(bbox.find('ymax').text)
                    
                    # Calculate width and height
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    
                    # Update class counts
                    stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
                    
                    # Update class-specific statistics
                    if class_name not in stats['class_specific']:
                        stats['class_specific'][class_name] = {
                            'count': 0,
                            'bbox_sizes': []
                        }
                    
                    stats['class_specific'][class_name]['count'] += 1
                    stats['class_specific'][class_name]['bbox_sizes'].append([bbox_width, bbox_height])
                    
                    stats['total_annotations'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {annotation_file}: {str(e)}")
                continue
        
        # Calculate average objects per image
        if stats['objects_per_image']:
            stats['avg_objects_per_image'] = sum(stats['objects_per_image']) / len(stats['objects_per_image'])
        
        # Calculate average image size
        if stats['image_sizes']:
            avg_width = sum(size[0] for size in stats['image_sizes']) / len(stats['image_sizes'])
            avg_height = sum(size[1] for size in stats['image_sizes']) / len(stats['image_sizes'])
            stats['avg_image_size'] = [avg_width, avg_height]
        
        # Calculate class-specific averages
        for class_name, class_stats in stats['class_specific'].items():
            if class_stats['bbox_sizes']:
                avg_width = sum(size[0] for size in class_stats['bbox_sizes']) / len(class_stats['bbox_sizes'])
                avg_height = sum(size[1] for size in class_stats['bbox_sizes']) / len(class_stats['bbox_sizes'])
                class_stats['avg_bbox_size'] = [avg_width, avg_height]
        
        return stats
