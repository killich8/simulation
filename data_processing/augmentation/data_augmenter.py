#!/usr/bin/env python

"""
Road Object Detection - Data Augmenter
This script augments the input data for the processing pipeline
"""

import os
import sys
import json
import logging
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
import albumentations as A

# Configure logging
logger = logging.getLogger("DataAugmenter")

class DataAugmenter:
    """
    Class for augmenting input data
    """
    def __init__(self, config):
        """Initialize the data augmenter"""
        self.config = config
        self.augmentation_types = config.get('augmentation_types', ['flip', 'rotate', 'brightness', 'contrast', 'noise'])
        self.augmentation_per_image = config.get('augmentation_per_image', 3)
        self.class_specific_augmentations = config.get('class_specific_augmentations', {})
        self.cyclist_augmentation_factor = config.get('cyclist_augmentation_factor', 2.0)
        self.include_original = config.get('include_original', True)
        
        # Statistics
        self.statistics = {
            'total_images': 0,
            'augmented_images': 0,
            'total_annotations': 0,
            'augmented_annotations': 0,
            'class_counts': {},
            'augmentation_counts': {},
            'errors': []
        }
        
        # Initialize augmentation pipeline
        self._init_augmentation_pipeline()
    
    def _init_augmentation_pipeline(self):
        """Initialize augmentation pipeline"""
        # Basic augmentations
        self.basic_augmentations = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2)
        ]
        
        # Cyclist-specific augmentations (more aggressive)
        self.cyclist_augmentations = [
            A.HorizontalFlip(p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussNoise(var_limit=(10.0, 60.0), p=0.4),
            A.Blur(blur_limit=4, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.7),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.4),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.4),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=1, num_flare_circles_upper=3, src_radius=100, src_color=(255, 255, 255), p=0.2)
        ]
        
        # Combine augmentations into a composition
        self.transform = A.Compose(
            self.basic_augmentations,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        
        self.cyclist_transform = A.Compose(
            self.cyclist_augmentations,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
    
    def augment(self, input_dir, output_dir):
        """Augment the input data"""
        logger.info(f"Augmenting data from {input_dir} to {output_dir}")
        
        # Reset statistics
        self.statistics = {
            'total_images': 0,
            'augmented_images': 0,
            'total_annotations': 0,
            'augmented_annotations': 0,
            'class_counts': {},
            'augmentation_counts': {},
            'errors': []
        }
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            error_msg = f"Input directory does not exist: {input_dir}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect annotation format
        annotation_format = self._detect_annotation_format(input_dir)
        if not annotation_format:
            error_msg = f"Could not detect annotation format in {input_dir}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        logger.info(f"Detected annotation format: {annotation_format}")
        
        # Augment data based on format
        if annotation_format == 'yolo':
            result = self._augment_yolo_dataset(input_dir, output_dir)
        elif annotation_format == 'coco':
            result = self._augment_coco_dataset(input_dir, output_dir)
        elif annotation_format == 'pascal_voc':
            result = self._augment_pascal_voc_dataset(input_dir, output_dir)
        else:
            error_msg = f"Unsupported annotation format: {annotation_format}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        # Log augmentation results
        logger.info(f"Augmentation completed: {self.statistics['augmented_images']} images generated")
        
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
    
    def _augment_yolo_dataset(self, input_dir, output_dir):
        """Augment YOLO format dataset"""
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
        
        # Create output directories
        output_images_dir = os.path.join(output_dir, 'images')
        output_labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
        
        # Copy classes.txt if it exists
        if classes_file:
            shutil.copy(classes_file, os.path.join(output_dir, 'classes.txt'))
        
        # Get all image files
        image_files = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        self.statistics['total_images'] = len(image_files)
        
        # Augment each image and its annotation
        for image_file in image_files:
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            label_file = os.path.join(labels_dir, f"{image_name}.txt")
            
            # Check if label file exists
            if not os.path.exists(label_file):
                error_msg = f"Label file not found for image: {image_file}"
                logger.warning(error_msg)
                self.statistics['errors'].append(error_msg)
                continue
            
            self.statistics['total_annotations'] += 1
            
            try:
                # Load image
                img = cv2.imread(image_file)
                if img is None:
                    error_msg = f"Could not read image: {image_file}"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Load annotations
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                bboxes = []
                class_labels = []
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
                
                # Check if image contains cyclists
                has_cyclist = False
                cyclist_class_id = None
                
                if class_names:
                    for i, name in enumerate(class_names):
                        if name.lower() in ['bicycle', 'cyclist', 'bike']:
                            cyclist_class_id = i
                            break
                
                if cyclist_class_id is not None and cyclist_class_id in class_labels:
                    has_cyclist = True
                
                # Include original image if requested
                if self.include_original:
                    # Save original image
                    output_image_path = os.path.join(output_images_dir, f"{image_name}.jpg")
                    cv2.imwrite(output_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    # Save original annotation
                    output_label_path = os.path.join(output_labels_dir, f"{image_name}.txt")
                    with open(output_label_path, 'w') as f:
                        f.write(''.join(lines))
                    
                    self.statistics['augmented_images'] += 1
                    self.statistics['augmented_annotations'] += 1
                
                # Determine number of augmentations
                num_augmentations = self.augmentation_per_image
                if has_cyclist:
                    num_augmentations = int(num_augmentations * self.cyclist_augmentation_factor)
                
                # Apply augmentations
                for i in range(num_augmentations):
                    # Choose appropriate transform
                    transform = self.cyclist_transform if has_cyclist else self.transform
                    
                    # Apply augmentation
                    augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                    augmented_img = augmented['image']
                    augmented_bboxes = augmented['bboxes']
                    augmented_class_labels = augmented['class_labels']
                    
                    # Save augmented image
                    aug_image_name = f"{image_name}_aug_{i+1}"
                    output_image_path = os.path.join(output_images_dir, f"{aug_image_name}.jpg")
                    cv2.imwrite(output_image_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
                    
                    # Save augmented annotation
                    output_label_path = os.path.join(output_labels_dir, f"{aug_image_name}.txt")
                    with open(output_label_path, 'w') as f:
                        for j in range(len(augmented_bboxes)):
                            bbox = augmented_bboxes[j]
                            class_id = augmented_class_labels[j]
                            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                    
                    self.statistics['augmented_images'] += 1
                    self.statistics['augmented_annotations'] += 1
                    
                    # Update augmentation statistics
                    self.statistics['augmentation_counts'][f'aug_{i+1}'] = self.statistics['augmentation_counts'].get(f'aug_{i+1}', 0) + 1
                
                # Update class statistics
                for class_id in class_labels:
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    self.statistics['class_counts'][class_name] = self.statistics['class_counts'].get(class_name, 0) + 1
                
            except Exception as e:
                error_msg = f"Error processing {image_file}: {str(e)}"
                logger.error(error_msg)
                self.statistics['errors'].append(error_msg)
        
        # Check if we have any augmented images
        if self.statistics['augmented_images'] == 0:
            error_msg = "No images were augmented"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        return {'success': True, 'message': f"Augmented {self.statistics['augmented_images']} images successfully"}
    
    def _augment_coco_dataset(self, input_dir, output_dir):
        """Augment COCO format dataset"""
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
            
            # Create output directories
            output_images_dir = os.path.join(output_dir, 'images')
            output_annotations_dir = os.path.join(output_dir, 'annotations')
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_annotations_dir, exist_ok=True)
            
            # Create category ID to name mapping
            category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
            
            # Find cyclist category ID
            cyclist_category_id = None
            for cat in coco_data['categories']:
                if cat['name'].lower() in ['bicycle', 'cyclist', 'bike']:
                    cyclist_category_id = cat['id']
                    break
            
            # Create image ID to annotations mapping
            image_to_annotations = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in image_to_annotations:
                    image_to_annotations[image_id] = []
                image_to_annotations[image_id].append(ann)
            
            self.statistics['total_images'] = len(coco_data['images'])
            self.statistics['total_annotations'] = len(coco_data['annotations'])
            
            # Create new COCO data structure for augmented dataset
            augmented_coco_data = {
                'info': coco_data.get('info', {}),
                'licenses': coco_data.get('licenses', []),
                'categories': coco_data['categories'],
                'images': [],
                'annotations': []
            }
            
            next_image_id = max([img['id'] for img in coco_data['images']]) + 1
            next_annotation_id = max([ann['id'] for ann in coco_data['annotations']]) + 1
            
            # Augment each image and its annotations
            for img_info in coco_data['images']:
                image_id = img_info['id']
                file_name = img_info['file_name']
                width = img_info['width']
                height = img_info['height']
                
                # Find image file
                image_path = os.path.join(images_dir, file_name)
                if not os.path.exists(image_path):
                    error_msg = f"Image file not found: {image_path}"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    continue
                
                try:
                    # Load image
                    img = cv2.imread(image_path)
                    if img is None:
                        error_msg = f"Could not read image: {image_path}"
                        logger.warning(error_msg)
                        self.statistics['errors'].append(error_msg)
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get annotations for this image
                    annotations = image_to_annotations.get(image_id, [])
                    
                    # Convert COCO annotations to YOLO format for augmentation
                    bboxes = []
                    class_labels = []
                    
                    for ann in annotations:
                        category_id = ann['category_id']
                        bbox = ann['bbox']  # [x, y, width, height] in COCO format
                        
                        # Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height]
                        x_center = (bbox[0] + bbox[2] / 2) / width
                        y_center = (bbox[1] + bbox[3] / 2) / height
                        bbox_width = bbox[2] / width
                        bbox_height = bbox[3] / height
                        
                        bboxes.append([x_center, y_center, bbox_width, bbox_height])
                        class_labels.append(category_id)
                    
                    # Check if image contains cyclists
                    has_cyclist = cyclist_category_id in class_labels if cyclist_category_id is not None else False
                    
                    # Include original image if requested
                    if self.include_original:
                        # Add original image to augmented dataset
                        augmented_coco_data['images'].append(img_info)
                        
                        # Add original annotations to augmented dataset
                        for ann in annotations:
                            augmented_coco_data['annotations'].append(ann)
                        
                        # Copy original image to output directory
                        shutil.copy(image_path, os.path.join(output_images_dir, file_name))
                        
                        self.statistics['augmented_images'] += 1
                        self.statistics['augmented_annotations'] += len(annotations)
                    
                    # Determine number of augmentations
                    num_augmentations = self.augmentation_per_image
                    if has_cyclist:
                        num_augmentations = int(num_augmentations * self.cyclist_augmentation_factor)
                    
                    # Apply augmentations
                    for i in range(num_augmentations):
                        # Choose appropriate transform
                        transform = self.cyclist_transform if has_cyclist else self.transform
                        
                        # Apply augmentation
                        augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                        augmented_img = augmented['image']
                        augmented_bboxes = augmented['bboxes']
                        augmented_class_labels = augmented['class_labels']
                        
                        # Create augmented image filename
                        base_name = os.path.splitext(file_name)[0]
                        aug_file_name = f"{base_name}_aug_{i+1}.jpg"
                        
                        # Save augmented image
                        output_image_path = os.path.join(output_images_dir, aug_file_name)
                        cv2.imwrite(output_image_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
                        
                        # Add augmented image to COCO data
                        augmented_image = {
                            'id': next_image_id,
                            'file_name': aug_file_name,
                            'width': width,
                            'height': height,
                            'date_captured': img_info.get('date_captured', '')
                        }
                        augmented_coco_data['images'].append(augmented_image)
                        
                        # Add augmented annotations to COCO data
                        for j in range(len(augmented_bboxes)):
                            bbox = augmented_bboxes[j]
                            category_id = augmented_class_labels[j]
                            
                            # Convert YOLO bbox [x_center, y_center, width, height] to COCO format [x, y, width, height]
                            x = (bbox[0] - bbox[2] / 2) * width
                            y = (bbox[1] - bbox[3] / 2) * height
                            bbox_width = bbox[2] * width
                            bbox_height = bbox[3] * height
                            
                            augmented_annotation = {
                                'id': next_annotation_id,
                                'image_id': next_image_id,
                                'category_id': category_id,
                                'bbox': [x, y, bbox_width, bbox_height],
                                'area': bbox_width * bbox_height,
                                'segmentation': [],
                                'iscrowd': 0
                            }
                            augmented_coco_data['annotations'].append(augmented_annotation)
                            next_annotation_id += 1
                        
                        next_image_id += 1
                        self.statistics['augmented_images'] += 1
                        self.statistics['augmented_annotations'] += len(augmented_bboxes)
                        
                        # Update augmentation statistics
                        self.statistics['augmentation_counts'][f'aug_{i+1}'] = self.statistics['augmentation_counts'].get(f'aug_{i+1}', 0) + 1
                    
                    # Update class statistics
                    for category_id in class_labels:
                        category_name = category_map.get(category_id, f"category_{category_id}")
                        self.statistics['class_counts'][category_name] = self.statistics['class_counts'].get(category_name, 0) + 1
                
                except Exception as e:
                    error_msg = f"Error processing {image_path}: {str(e)}"
                    logger.error(error_msg)
                    self.statistics['errors'].append(error_msg)
            
            # Save augmented COCO data
            with open(os.path.join(output_annotations_dir, 'instances_default.json'), 'w') as f:
                json.dump(augmented_coco_data, f, indent=2)
            
            # Check if we have any augmented images
            if self.statistics['augmented_images'] == 0:
                error_msg = "No images were augmented"
                logger.error(error_msg)
                self.statistics['errors'].append(error_msg)
                return {'success': False, 'message': error_msg}
            
            return {'success': True, 'message': f"Augmented {self.statistics['augmented_images']} images successfully"}
            
        except Exception as e:
            error_msg = f"Error augmenting COCO dataset: {str(e)}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
    
    def _augment_pascal_voc_dataset(self, input_dir, output_dir):
        """Augment Pascal VOC format dataset"""
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
        
        # Create output directories
        output_images_dir = os.path.join(output_dir, 'JPEGImages')
        output_annotations_dir = os.path.join(output_dir, 'Annotations')
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_annotations_dir, exist_ok=True)
        
        # Get all annotation files
        annotation_files = []
        for root, _, files in os.walk(annotations_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    annotation_files.append(os.path.join(root, file))
        
        self.statistics['total_annotations'] = len(annotation_files)
        
        # Augment each annotation and its image
        for annotation_file in annotation_files:
            try:
                # Parse XML
                tree = ET.parse(annotation_file)
                xml_root = tree.getroot()
                
                # Get image filename
                filename = xml_root.find('filename').text
                
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
                    continue
                
                self.statistics['total_images'] += 1
                
                # Get image size
                size_elem = xml_root.find('size')
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
                
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    error_msg = f"Could not read image: {image_path}"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get objects
                objects = xml_root.findall('object')
                
                # Convert Pascal VOC annotations to YOLO format for augmentation
                bboxes = []
                class_labels = []
                class_names = []
                
                for obj in objects:
                    class_name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    
                    # Get bounding box coordinates
                    x_min = float(bbox.find('xmin').text)
                    y_min = float(bbox.find('ymin').text)
                    x_max = float(bbox.find('xmax').text)
                    y_max = float(bbox.find('ymax').text)
                    
                    # Convert to YOLO format [x_center, y_center, width, height]
                    x_center = ((x_min + x_max) / 2) / width
                    y_center = ((y_min + y_max) / 2) / height
                    bbox_width = (x_max - x_min) / width
                    bbox_height = (y_max - y_min) / height
                    
                    # Use class name as label for now, will convert to numeric ID later
                    bboxes.append([x_center, y_center, bbox_width, bbox_height])
                    class_labels.append(len(class_names))  # Use index as temporary class ID
                    class_names.append(class_name)
                
                # Check if image contains cyclists
                has_cyclist = any(name.lower() in ['bicycle', 'cyclist', 'bike'] for name in class_names)
                
                # Include original image if requested
                if self.include_original:
                    # Copy original image to output directory
                    output_image_name = os.path.basename(image_path)
                    shutil.copy(image_path, os.path.join(output_images_dir, output_image_name))
                    
                    # Copy original annotation to output directory
                    output_annotation_name = os.path.basename(annotation_file)
                    shutil.copy(annotation_file, os.path.join(output_annotations_dir, output_annotation_name))
                    
                    self.statistics['augmented_images'] += 1
                    self.statistics['augmented_annotations'] += 1
                
                # Determine number of augmentations
                num_augmentations = self.augmentation_per_image
                if has_cyclist:
                    num_augmentations = int(num_augmentations * self.cyclist_augmentation_factor)
                
                # Apply augmentations
                for i in range(num_augmentations):
                    # Choose appropriate transform
                    transform = self.cyclist_transform if has_cyclist else self.transform
                    
                    # Apply augmentation
                    augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                    augmented_img = augmented['image']
                    augmented_bboxes = augmented['bboxes']
                    augmented_class_labels = augmented['class_labels']
                    
                    # Create augmented image filename
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    aug_file_name = f"{base_name}_aug_{i+1}.jpg"
                    
                    # Save augmented image
                    output_image_path = os.path.join(output_images_dir, aug_file_name)
                    cv2.imwrite(output_image_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
                    
                    # Create new XML for augmented annotation
                    new_root = ET.Element('annotation')
                    
                    ET.SubElement(new_root, 'folder').text = 'JPEGImages'
                    ET.SubElement(new_root, 'filename').text = aug_file_name
                    ET.SubElement(new_root, 'path').text = output_image_path
                    
                    source = ET.SubElement(new_root, 'source')
                    ET.SubElement(source, 'database').text = xml_root.find('source/database').text if xml_root.find('source/database') is not None else 'Unknown'
                    
                    size = ET.SubElement(new_root, 'size')
                    ET.SubElement(size, 'width').text = str(width)
                    ET.SubElement(size, 'height').text = str(height)
                    ET.SubElement(size, 'depth').text = '3'
                    
                    ET.SubElement(new_root, 'segmented').text = '0'
                    
                    # Add augmented objects
                    for j in range(len(augmented_bboxes)):
                        bbox = augmented_bboxes[j]
                        class_idx = augmented_class_labels[j]
                        class_name = class_names[class_idx]
                        
                        # Convert YOLO bbox [x_center, y_center, width, height] to Pascal VOC format [xmin, ymin, xmax, ymax]
                        x_min = int((bbox[0] - bbox[2] / 2) * width)
                        y_min = int((bbox[1] - bbox[3] / 2) * height)
                        x_max = int((bbox[0] + bbox[2] / 2) * width)
                        y_max = int((bbox[1] + bbox[3] / 2) * height)
                        
                        # Ensure coordinates are within image bounds
                        x_min = max(0, min(width - 1, x_min))
                        y_min = max(0, min(height - 1, y_min))
                        x_max = max(0, min(width, x_max))
                        y_max = max(0, min(height, y_max))
                        
                        # Create object element
                        obj = ET.SubElement(new_root, 'object')
                        ET.SubElement(obj, 'name').text = class_name
                        ET.SubElement(obj, 'pose').text = 'Unspecified'
                        ET.SubElement(obj, 'truncated').text = '0'
                        ET.SubElement(obj, 'difficult').text = '0'
                        
                        bndbox = ET.SubElement(obj, 'bndbox')
                        ET.SubElement(bndbox, 'xmin').text = str(x_min)
                        ET.SubElement(bndbox, 'ymin').text = str(y_min)
                        ET.SubElement(bndbox, 'xmax').text = str(x_max)
                        ET.SubElement(bndbox, 'ymax').text = str(y_max)
                    
                    # Save augmented XML
                    xml_str = minidom.parseString(ET.tostring(new_root)).toprettyxml(indent="  ")
                    output_xml_path = os.path.join(output_annotations_dir, f"{base_name}_aug_{i+1}.xml")
                    with open(output_xml_path, 'w') as f:
                        f.write(xml_str)
                    
                    self.statistics['augmented_images'] += 1
                    self.statistics['augmented_annotations'] += 1
                    
                    # Update augmentation statistics
                    self.statistics['augmentation_counts'][f'aug_{i+1}'] = self.statistics['augmentation_counts'].get(f'aug_{i+1}', 0) + 1
                
                # Update class statistics
                for class_name in class_names:
                    self.statistics['class_counts'][class_name] = self.statistics['class_counts'].get(class_name, 0) + 1
                
            except Exception as e:
                error_msg = f"Error processing {annotation_file}: {str(e)}"
                logger.error(error_msg)
                self.statistics['errors'].append(error_msg)
        
        # Check if we have any augmented images
        if self.statistics['augmented_images'] == 0:
            error_msg = "No images were augmented"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        return {'success': True, 'message': f"Augmented {self.statistics['augmented_images']} images successfully"}
    
    def get_statistics(self):
        """Get augmentation statistics"""
        return self.statistics
