#!/usr/bin/env python

"""
Road Object Detection - Data Transformer
This script transforms the input data for the processing pipeline
"""

import os
import sys
import json
import logging
import numpy as np
import cv2
from PIL import Image
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Configure logging
logger = logging.getLogger("DataTransformer")

class DataTransformer:
    """
    Class for transforming input data
    """
    def __init__(self, config):
        """Initialize the data transformer"""
        self.config = config
        self.target_size = config.get('target_size', [640, 640])
        self.normalize = config.get('normalize', True)
        self.convert_to_format = config.get('convert_to_format', 'yolo')
        self.include_original = config.get('include_original', False)
        self.class_mapping = config.get('class_mapping', {})
        
        # Statistics
        self.statistics = {
            'total_images': 0,
            'transformed_images': 0,
            'total_annotations': 0,
            'transformed_annotations': 0,
            'class_counts': {},
            'errors': []
        }
    
    def transform(self, input_dir, output_dir):
        """Transform the input data"""
        logger.info(f"Transforming data from {input_dir} to {output_dir}")
        
        # Reset statistics
        self.statistics = {
            'total_images': 0,
            'transformed_images': 0,
            'total_annotations': 0,
            'transformed_annotations': 0,
            'class_counts': {},
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
        
        # Transform data based on format
        if annotation_format == 'yolo':
            result = self._transform_yolo_dataset(input_dir, output_dir)
        elif annotation_format == 'coco':
            result = self._transform_coco_dataset(input_dir, output_dir)
        elif annotation_format == 'pascal_voc':
            result = self._transform_pascal_voc_dataset(input_dir, output_dir)
        else:
            error_msg = f"Unsupported annotation format: {annotation_format}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        # Log transformation results
        logger.info(f"Transformation completed: {self.statistics['transformed_images']} images transformed")
        
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
    
    def _transform_yolo_dataset(self, input_dir, output_dir):
        """Transform YOLO format dataset"""
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
        
        # Transform each image and its annotation
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
                
                original_height, original_width = img.shape[:2]
                
                # Resize image
                img_resized = cv2.resize(img, (self.target_size[0], self.target_size[1]))
                
                # Normalize if requested
                if self.normalize:
                    img_resized = img_resized.astype(np.float32) / 255.0
                
                # Save transformed image
                output_image_path = os.path.join(output_images_dir, f"{image_name}.jpg")
                cv2.imwrite(output_image_path, img_resized * 255 if self.normalize else img_resized)
                
                # Transform annotation
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                transformed_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Apply class mapping if provided
                    if class_names and class_id < len(class_names):
                        class_name = class_names[class_id]
                        if class_name in self.class_mapping:
                            class_id = self.class_mapping[class_name]
                    
                    # No need to transform coordinates for YOLO format if aspect ratio is preserved
                    # If aspect ratio changes, we need to adjust the coordinates
                    if self.target_size[0] / self.target_size[1] != original_width / original_height:
                        # Calculate scaling factors
                        scale_x = self.target_size[0] / original_width
                        scale_y = self.target_size[1] / original_height
                        
                        # Calculate new center coordinates
                        new_x_center = x_center * original_width * scale_x / self.target_size[0]
                        new_y_center = y_center * original_height * scale_y / self.target_size[1]
                        
                        # Calculate new width and height
                        new_width = width * original_width * scale_x / self.target_size[0]
                        new_height = height * original_height * scale_y / self.target_size[1]
                        
                        # Ensure values are within [0, 1]
                        new_x_center = max(0, min(1, new_x_center))
                        new_y_center = max(0, min(1, new_y_center))
                        new_width = max(0, min(1, new_width))
                        new_height = max(0, min(1, new_height))
                        
                        transformed_lines.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}")
                    else:
                        transformed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                    # Update class statistics
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    self.statistics['class_counts'][class_name] = self.statistics['class_counts'].get(class_name, 0) + 1
                
                # Save transformed annotation
                output_label_path = os.path.join(output_labels_dir, f"{image_name}.txt")
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(transformed_lines))
                
                self.statistics['transformed_images'] += 1
                self.statistics['transformed_annotations'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {image_file}: {str(e)}"
                logger.error(error_msg)
                self.statistics['errors'].append(error_msg)
        
        # Check if we have any transformed images
        if self.statistics['transformed_images'] == 0:
            error_msg = "No images were transformed"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        return {'success': True, 'message': f"Transformed {self.statistics['transformed_images']} images successfully"}
    
    def _transform_coco_dataset(self, input_dir, output_dir):
        """Transform COCO format dataset"""
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
            
            # Create output directories based on target format
            if self.convert_to_format == 'yolo':
                output_images_dir = os.path.join(output_dir, 'images')
                output_labels_dir = os.path.join(output_dir, 'labels')
            elif self.convert_to_format == 'pascal_voc':
                output_images_dir = os.path.join(output_dir, 'JPEGImages')
                output_annotations_dir = os.path.join(output_dir, 'Annotations')
            else:  # Keep COCO format
                output_images_dir = os.path.join(output_dir, 'images')
                output_annotations_dir = os.path.join(output_dir, 'annotations')
            
            os.makedirs(output_images_dir, exist_ok=True)
            
            if self.convert_to_format == 'yolo':
                os.makedirs(output_labels_dir, exist_ok=True)
                
                # Create classes.txt
                with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
                    for category in sorted(coco_data['categories'], key=lambda x: x['id']):
                        f.write(f"{category['name']}\n")
            elif self.convert_to_format == 'pascal_voc':
                os.makedirs(output_annotations_dir, exist_ok=True)
            else:  # Keep COCO format
                os.makedirs(output_annotations_dir, exist_ok=True)
            
            # Create category ID to name mapping
            category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
            
            # Create image ID to annotations mapping
            image_to_annotations = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in image_to_annotations:
                    image_to_annotations[image_id] = []
                image_to_annotations[image_id].append(ann)
            
            self.statistics['total_images'] = len(coco_data['images'])
            self.statistics['total_annotations'] = len(coco_data['annotations'])
            
            # Transform each image and its annotations
            transformed_coco_data = {
                'info': coco_data.get('info', {}),
                'licenses': coco_data.get('licenses', []),
                'categories': coco_data['categories'],
                'images': [],
                'annotations': []
            }
            
            annotation_id = 1
            
            for img_info in coco_data['images']:
                image_id = img_info['id']
                file_name = img_info['file_name']
                
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
                    
                    original_height, original_width = img.shape[:2]
                    
                    # Resize image
                    img_resized = cv2.resize(img, (self.target_size[0], self.target_size[1]))
                    
                    # Normalize if requested
                    if self.normalize:
                        img_resized = img_resized.astype(np.float32) / 255.0
                    
                    # Save transformed image
                    output_image_name = f"{os.path.splitext(file_name)[0]}.jpg"
                    output_image_path = os.path.join(output_images_dir, output_image_name)
                    cv2.imwrite(output_image_path, img_resized * 255 if self.normalize else img_resized)
                    
                    # Get annotations for this image
                    annotations = image_to_annotations.get(image_id, [])
                    
                    if self.convert_to_format == 'yolo':
                        # Convert to YOLO format
                        yolo_annotations = []
                        
                        for ann in annotations:
                            category_id = ann['category_id']
                            bbox = ann['bbox']  # [x, y, width, height] in COCO format
                            
                            # Apply class mapping if provided
                            category_name = category_map.get(category_id, f"category_{category_id}")
                            if category_name in self.class_mapping:
                                category_id = self.class_mapping[category_name]
                            
                            # Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height]
                            x_center = (bbox[0] + bbox[2] / 2) / original_width
                            y_center = (bbox[1] + bbox[3] / 2) / original_height
                            width = bbox[2] / original_width
                            height = bbox[3] / original_height
                            
                            # Ensure values are within [0, 1]
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            
                            yolo_annotations.append(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                            
                            # Update class statistics
                            self.statistics['class_counts'][category_name] = self.statistics['class_counts'].get(category_name, 0) + 1
                        
                        # Save YOLO annotations
                        output_label_path = os.path.join(output_labels_dir, f"{os.path.splitext(file_name)[0]}.txt")
                        with open(output_label_path, 'w') as f:
                            f.write('\n'.join(yolo_annotations))
                        
                    elif self.convert_to_format == 'pascal_voc':
                        # Convert to Pascal VOC format
                        root = ET.Element('annotation')
                        
                        ET.SubElement(root, 'folder').text = 'JPEGImages'
                        ET.SubElement(root, 'filename').text = output_image_name
                        ET.SubElement(root, 'path').text = output_image_path
                        
                        source = ET.SubElement(root, 'source')
                        ET.SubElement(source, 'database').text = 'COCO'
                        
                        size = ET.SubElement(root, 'size')
                        ET.SubElement(size, 'width').text = str(self.target_size[0])
                        ET.SubElement(size, 'height').text = str(self.target_size[1])
                        ET.SubElement(size, 'depth').text = '3'
                        
                        ET.SubElement(root, 'segmented').text = '0'
                        
                        for ann in annotations:
                            category_id = ann['category_id']
                            bbox = ann['bbox']  # [x, y, width, height] in COCO format
                            
                            # Apply class mapping if provided
                            category_name = category_map.get(category_id, f"category_{category_id}")
                            if category_name in self.class_mapping:
                                category_id = self.class_mapping[category_name]
                                category_name = list(self.class_mapping.keys())[list(self.class_mapping.values()).index(category_id)]
                            
                            # Scale bbox to new image size
                            x_scale = self.target_size[0] / original_width
                            y_scale = self.target_size[1] / original_height
                            
                            x_min = int(bbox[0] * x_scale)
                            y_min = int(bbox[1] * y_scale)
                            x_max = int((bbox[0] + bbox[2]) * x_scale)
                            y_max = int((bbox[1] + bbox[3]) * y_scale)
                            
                            # Ensure coordinates are within image bounds
                            x_min = max(0, min(self.target_size[0] - 1, x_min))
                            y_min = max(0, min(self.target_size[1] - 1, y_min))
                            x_max = max(0, min(self.target_size[0], x_max))
                            y_max = max(0, min(self.target_size[1], y_max))
                            
                            # Create object element
                            obj = ET.SubElement(root, 'object')
                            ET.SubElement(obj, 'name').text = category_name
                            ET.SubElement(obj, 'pose').text = 'Unspecified'
                            ET.SubElement(obj, 'truncated').text = '0'
                            ET.SubElement(obj, 'difficult').text = '0'
                            
                            bndbox = ET.SubElement(obj, 'bndbox')
                            ET.SubElement(bndbox, 'xmin').text = str(x_min)
                            ET.SubElement(bndbox, 'ymin').text = str(y_min)
                            ET.SubElement(bndbox, 'xmax').text = str(x_max)
                            ET.SubElement(bndbox, 'ymax').text = str(y_max)
                            
                            # Update class statistics
                            self.statistics['class_counts'][category_name] = self.statistics['class_counts'].get(category_name, 0) + 1
                        
                        # Save Pascal VOC annotation
                        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
                        output_xml_path = os.path.join(output_annotations_dir, f"{os.path.splitext(file_name)[0]}.xml")
                        with open(output_xml_path, 'w') as f:
                            f.write(xml_str)
                    
                    else:  # Keep COCO format but transform
                        # Add transformed image to new COCO data
                        transformed_img_info = {
                            'id': image_id,
                            'file_name': output_image_name,
                            'width': self.target_size[0],
                            'height': self.target_size[1],
                            'date_captured': img_info.get('date_captured', '')
                        }
                        transformed_coco_data['images'].append(transformed_img_info)
                        
                        # Transform annotations
                        for ann in annotations:
                            category_id = ann['category_id']
                            bbox = ann['bbox']  # [x, y, width, height] in COCO format
                            
                            # Apply class mapping if provided
                            category_name = category_map.get(category_id, f"category_{category_id}")
                            if category_name in self.class_mapping:
                                category_id = self.class_mapping[category_name]
                            
                            # Scale bbox to new image size
                            x_scale = self.target_size[0] / original_width
                            y_scale = self.target_size[1] / original_height
                            
                            new_bbox = [
                                bbox[0] * x_scale,
                                bbox[1] * y_scale,
                                bbox[2] * x_scale,
                                bbox[3] * y_scale
                            ]
                            
                            # Create transformed annotation
                            transformed_ann = {
                                'id': annotation_id,
                                'image_id': image_id,
                                'category_id': category_id,
                                'bbox': new_bbox,
                                'area': new_bbox[2] * new_bbox[3],
                                'segmentation': [],
                                'iscrowd': 0
                            }
                            transformed_coco_data['annotations'].append(transformed_ann)
                            annotation_id += 1
                            
                            # Update class statistics
                            self.statistics['class_counts'][category_name] = self.statistics['class_counts'].get(category_name, 0) + 1
                    
                    self.statistics['transformed_images'] += 1
                    self.statistics['transformed_annotations'] += len(annotations)
                    
                except Exception as e:
                    error_msg = f"Error processing {image_path}: {str(e)}"
                    logger.error(error_msg)
                    self.statistics['errors'].append(error_msg)
            
            # Save transformed COCO data if keeping COCO format
            if self.convert_to_format == 'coco':
                with open(os.path.join(output_annotations_dir, 'instances_default.json'), 'w') as f:
                    json.dump(transformed_coco_data, f, indent=2)
            
            # Check if we have any transformed images
            if self.statistics['transformed_images'] == 0:
                error_msg = "No images were transformed"
                logger.error(error_msg)
                self.statistics['errors'].append(error_msg)
                return {'success': False, 'message': error_msg}
            
            return {'success': True, 'message': f"Transformed {self.statistics['transformed_images']} images successfully"}
            
        except Exception as e:
            error_msg = f"Error transforming COCO dataset: {str(e)}"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
    
    def _transform_pascal_voc_dataset(self, input_dir, output_dir):
        """Transform Pascal VOC format dataset"""
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
        
        # Create output directories based on target format
        if self.convert_to_format == 'yolo':
            output_images_dir = os.path.join(output_dir, 'images')
            output_labels_dir = os.path.join(output_dir, 'labels')
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_labels_dir, exist_ok=True)
            
            # Create classes.txt by scanning all XML files for unique class names
            class_names = set()
            for root, _, files in os.walk(annotations_dir):
                for file in files:
                    if file.lower().endswith('.xml'):
                        try:
                            tree = ET.parse(os.path.join(root, file))
                            xml_root = tree.getroot()
                            for obj in xml_root.findall('object'):
                                class_name = obj.find('name').text
                                class_names.add(class_name)
                        except:
                            continue
            
            # Write classes.txt
            with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
                for i, name in enumerate(sorted(class_names)):
                    f.write(f"{name}\n")
            
            # Create class name to ID mapping
            class_to_id = {name: i for i, name in enumerate(sorted(class_names))}
            
        elif self.convert_to_format == 'coco':
            output_images_dir = os.path.join(output_dir, 'images')
            output_annotations_dir = os.path.join(output_dir, 'annotations')
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_annotations_dir, exist_ok=True)
            
            # Initialize COCO data structure
            coco_data = {
                'info': {
                    'description': 'Converted from Pascal VOC',
                    'version': '1.0',
                    'year': 2023,
                    'contributor': 'DataTransformer',
                    'date_created': ''
                },
                'licenses': [],
                'images': [],
                'annotations': [],
                'categories': []
            }
            
            # Scan all XML files for unique class names
            class_names = set()
            for root, _, files in os.walk(annotations_dir):
                for file in files:
                    if file.lower().endswith('.xml'):
                        try:
                            tree = ET.parse(os.path.join(root, file))
                            xml_root = tree.getroot()
                            for obj in xml_root.findall('object'):
                                class_name = obj.find('name').text
                                class_names.add(class_name)
                        except:
                            continue
            
            # Create COCO categories
            for i, name in enumerate(sorted(class_names)):
                coco_data['categories'].append({
                    'id': i,
                    'name': name,
                    'supercategory': 'none'
                })
            
            # Create class name to ID mapping
            class_to_id = {name: i for i, name in enumerate(sorted(class_names))}
            
        else:  # Keep Pascal VOC format
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
        
        # Transform each annotation and its image
        image_id = 1
        annotation_id = 1
        
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
                
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    error_msg = f"Could not read image: {image_path}"
                    logger.warning(error_msg)
                    self.statistics['errors'].append(error_msg)
                    continue
                
                # Get original image size
                size_elem = xml_root.find('size')
                original_width = int(size_elem.find('width').text)
                original_height = int(size_elem.find('height').text)
                
                # Resize image
                img_resized = cv2.resize(img, (self.target_size[0], self.target_size[1]))
                
                # Normalize if requested
                if self.normalize:
                    img_resized = img_resized.astype(np.float32) / 255.0
                
                # Save transformed image
                output_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
                output_image_path = os.path.join(output_images_dir, output_image_name)
                cv2.imwrite(output_image_path, img_resized * 255 if self.normalize else img_resized)
                
                # Get objects
                objects = xml_root.findall('object')
                
                if self.convert_to_format == 'yolo':
                    # Convert to YOLO format
                    yolo_annotations = []
                    
                    for obj in objects:
                        class_name = obj.find('name').text
                        bbox = obj.find('bndbox')
                        
                        # Apply class mapping if provided
                        if class_name in self.class_mapping:
                            class_id = self.class_mapping[class_name]
                        else:
                            class_id = class_to_id[class_name]
                        
                        # Get bounding box coordinates
                        x_min = float(bbox.find('xmin').text)
                        y_min = float(bbox.find('ymin').text)
                        x_max = float(bbox.find('xmax').text)
                        y_max = float(bbox.find('ymax').text)
                        
                        # Convert to YOLO format [x_center, y_center, width, height]
                        x_center = ((x_min + x_max) / 2) / original_width
                        y_center = ((y_min + y_max) / 2) / original_height
                        width = (x_max - x_min) / original_width
                        height = (y_max - y_min) / original_height
                        
                        # Ensure values are within [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        
                        # Update class statistics
                        self.statistics['class_counts'][class_name] = self.statistics['class_counts'].get(class_name, 0) + 1
                    
                    # Save YOLO annotations
                    output_label_path = os.path.join(output_labels_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                
                elif self.convert_to_format == 'coco':
                    # Add image to COCO data
                    coco_data['images'].append({
                        'id': image_id,
                        'file_name': output_image_name,
                        'width': self.target_size[0],
                        'height': self.target_size[1],
                        'date_captured': ''
                    })
                    
                    # Convert objects to COCO annotations
                    for obj in objects:
                        class_name = obj.find('name').text
                        bbox = obj.find('bndbox')
                        
                        # Apply class mapping if provided
                        if class_name in self.class_mapping:
                            category_id = self.class_mapping[class_name]
                        else:
                            category_id = class_to_id[class_name]
                        
                        # Get bounding box coordinates
                        x_min = float(bbox.find('xmin').text)
                        y_min = float(bbox.find('ymin').text)
                        x_max = float(bbox.find('xmax').text)
                        y_max = float(bbox.find('ymax').text)
                        
                        # Scale to new image size
                        x_scale = self.target_size[0] / original_width
                        y_scale = self.target_size[1] / original_height
                        
                        # Convert to COCO format [x, y, width, height]
                        x = x_min * x_scale
                        y = y_min * y_scale
                        width = (x_max - x_min) * x_scale
                        height = (y_max - y_min) * y_scale
                        
                        # Create COCO annotation
                        coco_data['annotations'].append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': category_id,
                            'bbox': [x, y, width, height],
                            'area': width * height,
                            'segmentation': [],
                            'iscrowd': 0
                        })
                        
                        annotation_id += 1
                        
                        # Update class statistics
                        self.statistics['class_counts'][class_name] = self.statistics['class_counts'].get(class_name, 0) + 1
                    
                    image_id += 1
                
                else:  # Keep Pascal VOC format
                    # Create new XML with transformed data
                    new_root = ET.Element('annotation')
                    
                    ET.SubElement(new_root, 'folder').text = 'JPEGImages'
                    ET.SubElement(new_root, 'filename').text = output_image_name
                    ET.SubElement(new_root, 'path').text = output_image_path
                    
                    source = ET.SubElement(new_root, 'source')
                    ET.SubElement(source, 'database').text = xml_root.find('source/database').text if xml_root.find('source/database') is not None else 'Unknown'
                    
                    size = ET.SubElement(new_root, 'size')
                    ET.SubElement(size, 'width').text = str(self.target_size[0])
                    ET.SubElement(size, 'height').text = str(self.target_size[1])
                    ET.SubElement(size, 'depth').text = '3'
                    
                    ET.SubElement(new_root, 'segmented').text = '0'
                    
                    # Transform objects
                    for obj in objects:
                        class_name = obj.find('name').text
                        bbox = obj.find('bndbox')
                        
                        # Apply class mapping if provided
                        if class_name in self.class_mapping:
                            class_name = list(self.class_mapping.keys())[list(self.class_mapping.values()).index(self.class_mapping[class_name])]
                        
                        # Get bounding box coordinates
                        x_min = float(bbox.find('xmin').text)
                        y_min = float(bbox.find('ymin').text)
                        x_max = float(bbox.find('xmax').text)
                        y_max = float(bbox.find('ymax').text)
                        
                        # Scale to new image size
                        x_scale = self.target_size[0] / original_width
                        y_scale = self.target_size[1] / original_height
                        
                        new_x_min = int(x_min * x_scale)
                        new_y_min = int(y_min * y_scale)
                        new_x_max = int(x_max * x_scale)
                        new_y_max = int(y_max * y_scale)
                        
                        # Ensure coordinates are within image bounds
                        new_x_min = max(0, min(self.target_size[0] - 1, new_x_min))
                        new_y_min = max(0, min(self.target_size[1] - 1, new_y_min))
                        new_x_max = max(0, min(self.target_size[0], new_x_max))
                        new_y_max = max(0, min(self.target_size[1], new_y_max))
                        
                        # Create object element
                        new_obj = ET.SubElement(new_root, 'object')
                        ET.SubElement(new_obj, 'name').text = class_name
                        ET.SubElement(new_obj, 'pose').text = obj.find('pose').text if obj.find('pose') is not None else 'Unspecified'
                        ET.SubElement(new_obj, 'truncated').text = obj.find('truncated').text if obj.find('truncated') is not None else '0'
                        ET.SubElement(new_obj, 'difficult').text = obj.find('difficult').text if obj.find('difficult') is not None else '0'
                        
                        new_bbox = ET.SubElement(new_obj, 'bndbox')
                        ET.SubElement(new_bbox, 'xmin').text = str(new_x_min)
                        ET.SubElement(new_bbox, 'ymin').text = str(new_y_min)
                        ET.SubElement(new_bbox, 'xmax').text = str(new_x_max)
                        ET.SubElement(new_bbox, 'ymax').text = str(new_y_max)
                        
                        # Update class statistics
                        self.statistics['class_counts'][class_name] = self.statistics['class_counts'].get(class_name, 0) + 1
                    
                    # Save transformed XML
                    xml_str = minidom.parseString(ET.tostring(new_root)).toprettyxml(indent="  ")
                    output_xml_path = os.path.join(output_annotations_dir, f"{os.path.splitext(os.path.basename(annotation_file))[0]}.xml")
                    with open(output_xml_path, 'w') as f:
                        f.write(xml_str)
                
                self.statistics['transformed_images'] += 1
                self.statistics['transformed_annotations'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {annotation_file}: {str(e)}"
                logger.error(error_msg)
                self.statistics['errors'].append(error_msg)
        
        # Save COCO data if converting to COCO
        if self.convert_to_format == 'coco':
            with open(os.path.join(output_annotations_dir, 'instances_default.json'), 'w') as f:
                json.dump(coco_data, f, indent=2)
        
        # Check if we have any transformed images
        if self.statistics['transformed_images'] == 0:
            error_msg = "No images were transformed"
            logger.error(error_msg)
            self.statistics['errors'].append(error_msg)
            return {'success': False, 'message': error_msg}
        
        return {'success': True, 'message': f"Transformed {self.statistics['transformed_images']} images successfully"}
    
    def get_statistics(self):
        """Get transformation statistics"""
        return self.statistics
