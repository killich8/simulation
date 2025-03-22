#!/usr/bin/env python

"""
Road Object Detection - Data Validator
This script validates the input data for the processing pipeline
"""

import os
import sys
import json
import logging
import numpy as np
import cv2
from PIL import Image
import xml.etree.ElementTree as ET

# Configure logging
logger = logging.getLogger("DataValidator")

class DataValidator:
    """
    Class for validating input data
    """
    def __init__(self, config):
        """Initialize the data validator"""
        self.config = config
        self.min_image_size = config.get('min_image_size', [640, 480])
        self.max_image_size = config.get('max_image_size', [4096, 4096])
        self.min_objects_per_image = config.get('min_objects_per_image', 1)
        self.required_classes = config.get('required_classes', ['car', 'pedestrian', 'bicycle'])
        self.min_class_instances = config.get('min_class_instances', {'bicycle': 100})
        self.annotation_formats = config.get('annotation_formats', ['yolo', 'coco', 'pascal_voc'])
        
        # Statistics
        self.statistics = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'total_annotations': 0,
            'valid_annotations': 0,
            'invalid_annotations': 0,
            'class_counts': {},
            'errors': []
        }
    
    def validate(self, input_dir):
        """Validate the input data"""
        logger.info(f"Validating data in {input_dir}")
        
        # Reset statistics
        self.statistics = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'total_annotations': 0,
            'valid_annotations': 0,
            'invalid_annotations': 0,
            'class_counts': {},
            'errors': []
        }
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            error_msg = f"Input directory does not exist: {input_dir}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        # Detect annotation format
        annotation_format = self._detect_annotation_format(input_dir)
        if not annotation_format:
            error_msg = f"Could not detect annotation format in {input_dir}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        logger.info(f"Detected annotation format: {annotation_format}")
        
        # Validate images and annotations
        if annotation_format == 'yolo':
            result = self._validate_yolo_dataset(input_dir)
        elif annotation_format == 'coco':
            result = self._validate_coco_dataset(input_dir)
        elif annotation_format == 'pascal_voc':
            result = self._validate_pascal_voc_dataset(input_dir)
        else:
            error_msg = f"Unsupported annotation format: {annotation_format}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        # Check if we have enough instances of required classes
        for class_name, min_count in self.min_class_instances.items():
            actual_count = self.statistics['class_counts'].get(class_name, 0)
            if actual_count < min_count:
                error_msg = f"Not enough instances of class '{class_name}': {actual_count} < {min_count}"
                logger.warning(error_msg)
                self.statistics['errors'].append(error_msg)
                # Don't fail validation for this, just warn
        
        # Check if all required classes are present
        for class_name in self.required_classes:
            if class_name not in self.statistics['class_counts'] or self.statistics['class_counts'][class_name] == 0:
                error_msg = f"Required class '{class_name}' not found in dataset"
                logger.warning(error_msg)
                self.statistics['errors'].append(error_msg)
                # Don't fail validation for this, just warn
        
        # Log validation results
        logger.info(f"Validation completed: {self.statistics['valid_images']} valid images, "
                   f"{self.statistics['invalid_images']} invalid images")
        
        return result
    
    def _detect_annotation_format(self, input_dir):
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
    
    def _validate_yolo_dataset(self, input_dir):
        """Validate YOLO format dataset"""
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
            error_msg = f"Could not find images and labels directories in {input_dir}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
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
        
        # Get all image files
        image_files = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        self.statistics['total_images'] = len(image_files)
        
        # Validate each image and its annotation
        for image_file in image_files:
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            label_file = os.path.join(labels_dir, f"{image_name}.txt")
            
            # Check if label file exists
            if not os.path.exists(label_file):
                error_msg = f"Label file not found for image: {image_file}"
                logger.warning(error_msg)
                self.statistics['errors'].append(error_msg)
                self.statistics['invalid_annotations'] += 1
                self.statistics['invalid_images'] += 1
                continue
            
            self.statistics['total_annotations'] += 1
            
            # Validate image
            try:
                img = Image.open(image_file)
                width, height = img.size
                
                # Check image size
                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    error_msg = f"Image too small: {image_file} ({width}x{height})"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_images'] += 1
                    continue
                
                if width > self.max_image_size[0] or height > self.max_image_size[1]:
                    error_msg = f"Image too large: {image_file} ({width}x{height})"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_images'] += 1
                    continue
                
                # Validate annotation
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) < self.min_objects_per_image:
                    error_msg = f"Too few objects in image: {image_file} ({len(lines)} < {self.min_objects_per_image})"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_annotations'] += 1
                    continue
                
                # Parse annotations
                valid_annotation = True
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        error_msg = f"Invalid annotation format in {label_file}: {line}"
                        logger.warning(error_msg)
                        self.statistics['errors'].append(error_msg)
                        valid_annotation = False
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:5])
                    
                    # Check if values are in range [0, 1]
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        error_msg = f"Invalid bounding box values in {label_file}: {line}"
                        logger.warning(error_msg)
                        self.statistics['errors'].append(error_msg)
                        valid_annotation = False
                        continue
                    
                    # Update class statistics
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    self.statistics['class_counts'][class_name] = self.statistics['class_counts'].get(class_name, 0) + 1
                
                if valid_annotation:
                    self.statistics['valid_annotations'] += 1
                    self.statistics['valid_images'] += 1
                else:
                    self.statistics['invalid_annotations'] += 1
                    self.statistics['invalid_images'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {image_file}: {str(e)}"
                logger.error(error_msg)
                self.statistics['errors'].append(error_msg)
                self.statistics['invalid_images'] += 1
        
        # Check if we have any valid images
        if self.statistics['valid_images'] == 0:
            error_msg = "No valid images found in dataset"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        return {'success': True, 'message': f"Validated {self.statistics['valid_images']} images successfully"}
    
    def _validate_coco_dataset(self, input_dir):
        """Validate COCO format dataset"""
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
            error_msg = f"Could not find COCO annotation file in {input_dir}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        # Load COCO annotations
        try:
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Check required fields
            if not all(key in coco_data for key in ['images', 'annotations', 'categories']):
                error_msg = f"Invalid COCO format in {coco_file}: missing required fields"
                logger.error(error_msg)
                self.statistics['errors'].append(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Get image directory
            images_dir = os.path.dirname(coco_file)
            if 'images' in os.listdir(os.path.dirname(images_dir)):
                images_dir = os.path.join(os.path.dirname(images_dir), 'images')
            
            # Create category ID to name mapping
            category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
            
            # Validate images and annotations
            self.statistics['total_images'] = len(coco_data['images'])
            self.statistics['total_annotations'] = len(coco_data['annotations'])
            
            # Create image ID to file mapping
            image_map = {img['id']: img for img in coco_data['images']}
            
            # Count annotations per image
            annotations_per_image = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                annotations_per_image[image_id] = annotations_per_image.get(image_id, 0) + 1
                
                # Update class statistics
                category_id = ann['category_id']
                category_name = category_map.get(category_id, f"category_{category_id}")
                self.statistics['class_counts'][category_name] = self.statistics['class_counts'].get(category_name, 0) + 1
            
            # Validate each image
            for img in coco_data['images']:
                image_id = img['id']
                file_name = img['file_name']
                width = img['width']
                height = img['height']
                
                # Check image size
                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    error_msg = f"Image too small: {file_name} ({width}x{height})"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_images'] += 1
                    continue
                
                if width > self.max_image_size[0] or height > self.max_image_size[1]:
                    error_msg = f"Image too large: {file_name} ({width}x{height})"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_images'] += 1
                    continue
                
                # Check if image file exists
                image_path = os.path.join(images_dir, file_name)
                if not os.path.exists(image_path):
                    error_msg = f"Image file not found: {image_path}"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_images'] += 1
                    continue
                
                # Check number of annotations
                if image_id not in annotations_per_image or annotations_per_image[image_id] < self.min_objects_per_image:
                    error_msg = f"Too few objects in image: {file_name} ({annotations_per_image.get(image_id, 0)} < {self.min_objects_per_image})"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_images'] += 1
                    continue
                
                # Image is valid
                self.statistics['valid_images'] += 1
            
            # All annotations are considered valid in COCO format
            self.statistics['valid_annotations'] = self.statistics['total_annotations']
            
        except Exception as e:
            error_msg = f"Error processing COCO annotations: {str(e)}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        # Check if we have any valid images
        if self.statistics['valid_images'] == 0:
            error_msg = "No valid images found in dataset"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        return {'success': True, 'message': f"Validated {self.statistics['valid_images']} images successfully"}
    
    def _validate_pascal_voc_dataset(self, input_dir):
        """Validate Pascal VOC format dataset"""
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
        
        if not annotations_dir or not images_dir:
            error_msg = f"Could not find annotations and images directories in {input_dir}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        # Get all annotation files
        annotation_files = []
        for root, _, files in os.walk(annotations_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    annotation_files.append(os.path.join(root, file))
        
        self.statistics['total_annotations'] = len(annotation_files)
        
        # Validate each annotation and its image
        for annotation_file in annotation_files:
            try:
                # Parse XML
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                
                # Get image filename
                filename = root.find('filename').text
                
                # Get image path
                image_path = os.path.join(images_dir, filename)
                if not os.path.exists(image_path):
                    # Try with different extensions
                    base_name = os.path.splitext(filename)[0]
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        alt_path = os.path.join(images_dir, base_name + ext)
                        if os.path.exists(alt_path):
                            image_path = alt_path
                            break
                
                if not os.path.exists(image_path):
                    error_msg = f"Image file not found for annotation: {annotation_file}"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_annotations'] += 1
                    continue
                
                self.statistics['total_images'] += 1
                
                # Get image size
                size_elem = root.find('size')
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
                
                # Check image size
                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    error_msg = f"Image too small: {image_path} ({width}x{height})"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_images'] += 1
                    continue
                
                if width > self.max_image_size[0] or height > self.max_image_size[1]:
                    error_msg = f"Image too large: {image_path} ({width}x{height})"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_images'] += 1
                    continue
                
                # Get objects
                objects = root.findall('object')
                
                if len(objects) < self.min_objects_per_image:
                    error_msg = f"Too few objects in image: {image_path} ({len(objects)} < {self.min_objects_per_image})"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    self.statistics['invalid_annotations'] += 1
                    self.statistics['invalid_images'] += 1
                    continue
                
                # Validate objects
                valid_annotation = True
                for obj in objects:
                    name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    
                    if bbox is None:
                        error_msg = f"Missing bounding box in {annotation_file} for object {name}"
                        logger.warning(error_msg)
                        self.statistics['errors'].append(error_msg)
                        valid_annotation = False
                        continue
                    
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # Check if bounding box is valid
                    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                        error_msg = f"Invalid bounding box in {annotation_file}: {xmin},{ymin},{xmax},{ymax}"
                        logger.warning(error_msg)
                        self.statistics['errors'].append(error_msg)
                        valid_annotation = False
                        continue
                    
                    # Update class statistics
                    self.statistics['class_counts'][name] = self.statistics['class_counts'].get(name, 0) + 1
                
                if valid_annotation:
                    self.statistics['valid_annotations'] += 1
                    self.statistics['valid_images'] += 1
                else:
                    self.statistics['invalid_annotations'] += 1
                    self.statistics['invalid_images'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {annotation_file}: {str(e)}"
                logger.error(error_msg)
                self.statistics['errors'].append(error_msg)
                self.statistics['invalid_annotations'] += 1
        
        # Check if we have any valid images
        if self.statistics['valid_images'] == 0:
            error_msg = "No valid images found in dataset"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        return {'success': True, 'message': f"Validated {self.statistics['valid_images']} images successfully"}
    
    def get_statistics(self):
        """Get validation statistics"""
        return self.statistics
