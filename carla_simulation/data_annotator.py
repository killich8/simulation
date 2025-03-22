#!/usr/bin/env python

"""
Road Object Detection - Data Annotation Script
This script handles the annotation of data collected from CARLA simulator
with a focus on creating high-quality annotations for underrepresented classes like cyclists.
"""

import os
import sys
import glob
import json
import argparse
import logging
import numpy as np
import cv2
from datetime import datetime
import boto3
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Configure logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class DataAnnotator:
    """
    Class for annotating data collected from CARLA simulator
    """
    def __init__(self, args):
        self.args = args
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.format = args.format.lower()
        self.s3_bucket = args.s3_bucket
        self.s3_prefix = args.s3_prefix
        
        # Validate input directory
        if not os.path.exists(self.input_dir):
            logging.error(f"Input directory does not exist: {self.input_dir}")
            sys.exit(1)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize S3 client if needed
        self.s3_client = None
        if self.s3_bucket:
            self.s3_client = boto3.client('s3')
            logging.info(f"S3 upload enabled to bucket: {self.s3_bucket}")
        
        # Define class mapping
        self.class_mapping = {
            'car': 0,
            'bicycle': 1,
            'motorcycle': 2,
            'pedestrian': 3,
            'truck': 4,
            'bus': 5,
            'traffic_light': 6,
            'traffic_sign': 7
        }
        
        # Define class colors for visualization
        self.class_colors = {
            'car': (0, 0, 255),        # Red
            'bicycle': (0, 255, 0),    # Green
            'motorcycle': (255, 0, 0), # Blue
            'pedestrian': (255, 255, 0), # Yellow
            'truck': (255, 0, 255),    # Magenta
            'bus': (0, 255, 255),      # Cyan
            'traffic_light': (128, 128, 0), # Olive
            'traffic_sign': (128, 0, 128)  # Purple
        }
    
    def process_dataset(self):
        """Process the entire dataset and create annotations"""
        logging.info(f"Processing dataset in {self.input_dir}")
        
        # Get all annotation files
        annotation_files = sorted(glob.glob(os.path.join(self.input_dir, 'annotations', '*.json')))
        
        if not annotation_files:
            logging.error(f"No annotation files found in {self.input_dir}/annotations")
            return False
        
        # Process each annotation file
        for annotation_file in annotation_files:
            frame_id = os.path.splitext(os.path.basename(annotation_file))[0]
            logging.info(f"Processing frame {frame_id}")
            
            # Load annotation
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
            
            # Load corresponding RGB image
            rgb_image_path = os.path.join(self.input_dir, 'rgb', f"{frame_id}.png")
            if not os.path.exists(rgb_image_path):
                logging.warning(f"RGB image not found for frame {frame_id}, skipping")
                continue
            
            # Create annotations in the specified format
            if self.format == 'coco':
                self.create_coco_annotation(annotation, rgb_image_path, frame_id)
            elif self.format == 'pascal_voc':
                self.create_pascal_voc_annotation(annotation, rgb_image_path, frame_id)
            elif self.format == 'yolo':
                self.create_yolo_annotation(annotation, rgb_image_path, frame_id)
            else:
                logging.error(f"Unsupported annotation format: {self.format}")
                return False
            
            # Create visualization if requested
            if self.args.visualize:
                self.create_visualization(annotation, rgb_image_path, frame_id)
        
        # Create dataset-level annotation file if needed
        if self.format == 'coco':
            self.create_coco_dataset_file()
        
        logging.info("Dataset annotation completed successfully")
        
        # Upload to S3 if requested
        if self.s3_bucket:
            self.upload_annotations_to_s3()
        
        return True
    
    def create_coco_annotation(self, annotation, image_path, frame_id):
        """Create annotation in COCO format"""
        # Load image to get dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Create annotation directory
        coco_dir = os.path.join(self.output_dir, 'coco')
        os.makedirs(coco_dir, exist_ok=True)
        
        # Create frame-specific annotation
        coco_annotation = {
            'info': {
                'description': 'CARLA Synthetic Dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'CARLA Simulator',
                'date_created': datetime.now().isoformat()
            },
            'images': [{
                'id': int(frame_id),
                'file_name': f"{frame_id}.png",
                'width': width,
                'height': height,
                'date_captured': datetime.now().isoformat()
            }],
            'annotations': [],
            'categories': [
                {'id': 0, 'name': 'car', 'supercategory': 'vehicle'},
                {'id': 1, 'name': 'bicycle', 'supercategory': 'vehicle'},
                {'id': 2, 'name': 'motorcycle', 'supercategory': 'vehicle'},
                {'id': 3, 'name': 'pedestrian', 'supercategory': 'person'},
                {'id': 4, 'name': 'truck', 'supercategory': 'vehicle'},
                {'id': 5, 'name': 'bus', 'supercategory': 'vehicle'},
                {'id': 6, 'name': 'traffic_light', 'supercategory': 'traffic'},
                {'id': 7, 'name': 'traffic_sign', 'supercategory': 'traffic'}
            ]
        }
        
        # Add annotations for each object
        annotation_id = 1
        for obj in annotation['objects']:
            obj_type = obj['type']
            bbox = obj['bbox']
            
            # Convert [x_min, y_min, x_max, y_max] to [x, y, width, height]
            coco_bbox = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]
            ]
            
            # Calculate area
            area = coco_bbox[2] * coco_bbox[3]
            
            # Add to annotations
            coco_annotation['annotations'].append({
                'id': annotation_id,
                'image_id': int(frame_id),
                'category_id': self.class_mapping.get(obj_type, 0),
                'bbox': coco_bbox,
                'area': area,
                'iscrowd': 0
            })
            
            annotation_id += 1
        
        # Save annotation to file
        with open(os.path.join(coco_dir, f"{frame_id}.json"), 'w') as f:
            json.dump(coco_annotation, f, indent=2)
        
        # Store annotation for dataset-level file
        self.coco_annotations = self.coco_annotations if hasattr(self, 'coco_annotations') else []
        self.coco_annotations.append(coco_annotation)
    
    def create_coco_dataset_file(self):
        """Create a single COCO dataset file from all annotations"""
        if not hasattr(self, 'coco_annotations') or not self.coco_annotations:
            logging.warning("No COCO annotations to combine")
            return
        
        # Create combined annotation
        combined_annotation = {
            'info': self.coco_annotations[0]['info'],
            'images': [],
            'annotations': [],
            'categories': self.coco_annotations[0]['categories']
        }
        
        # Combine all annotations
        annotation_id = 1
        for coco_annotation in self.coco_annotations:
            combined_annotation['images'].extend(coco_annotation['images'])
            
            # Update annotation IDs
            for ann in coco_annotation['annotations']:
                ann['id'] = annotation_id
                combined_annotation['annotations'].append(ann)
                annotation_id += 1
        
        # Save combined annotation to file
        with open(os.path.join(self.output_dir, 'coco', 'dataset.json'), 'w') as f:
            json.dump(combined_annotation, f, indent=2)
        
        logging.info(f"Created COCO dataset file with {len(combined_annotation['images'])} images and {len(combined_annotation['annotations'])} annotations")
    
    def create_pascal_voc_annotation(self, annotation, image_path, frame_id):
        """Create annotation in Pascal VOC format"""
        # Load image to get dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Create annotation directory
        voc_dir = os.path.join(self.output_dir, 'pascal_voc')
        os.makedirs(voc_dir, exist_ok=True)
        
        # Create XML structure
        root = ET.Element('annotation')
        
        # Add basic information
        ET.SubElement(root, 'folder').text = 'rgb'
        ET.SubElement(root, 'filename').text = f"{frame_id}.png"
        ET.SubElement(root, 'path').text = image_path
        
        # Add source information
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = 'CARLA Simulator'
        
        # Add size information
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'
        
        # Add segmented information
        ET.SubElement(root, 'segmented').text = '0'
        
        # Add object annotations
        for obj in annotation['objects']:
            obj_type = obj['type']
            bbox = obj['bbox']
            
            # Create object element
            obj_elem = ET.SubElement(root, 'object')
            ET.SubElement(obj_elem, 'name').text = obj_type
            ET.SubElement(obj_elem, 'pose').text = 'Unspecified'
            ET.SubElement(obj_elem, 'truncated').text = '0'
            ET.SubElement(obj_elem, 'difficult').text = '0'
            
            # Add bounding box
            bndbox = ET.SubElement(obj_elem, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
            ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
            ET.SubElement(bndbox, 'xmax').text = str(int(bbox[2]))
            ET.SubElement(bndbox, 'ymax').text = str(int(bbox[3]))
        
        # Convert to pretty XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        
        # Save to file
        with open(os.path.join(voc_dir, f"{frame_id}.xml"), 'w') as f:
            f.write(xml_str)
    
    def create_yolo_annotation(self, annotation, image_path, frame_id):
        """Create annotation in YOLO format"""
        # Load image to get dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Create annotation directory
        yolo_dir = os.path.join(self.output_dir, 'yolo')
        os.makedirs(yolo_dir, exist_ok=True)
        
        # Create class mapping file if it doesn't exist
        classes_file = os.path.join(yolo_dir, 'classes.txt')
        if not os.path.exists(classes_file):
            with open(classes_file, 'w') as f:
                for class_name in sorted(self.class_mapping.keys(), key=lambda x: self.class_mapping[x]):
                    f.write(f"{class_name}\n")
        
        # Create annotation file
        with open(os.path.join(yolo_dir, f"{frame_id}.txt"), 'w') as f:
            for obj in annotation['objects']:
                obj_type = obj['type']
                bbox = obj['bbox']
                
                # Convert [x_min, y_min, x_max, y_max] to YOLO format [x_center, y_center, width, height]
                x_center = ((bbox[0] + bbox[2]) / 2) / width
                y_center = ((bbox[1] + bbox[3]) / 2) / height
                bbox_width = (bbox[2] - bbox[0]) / width
                bbox_height = (bbox[3] - bbox[1]) / height
                
                # Get class ID
                class_id = self.class_mapping.get(obj_type, 0)
                
                # Write to file
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
    
    def create_visualization(self, annotation, image_path, frame_id):
        """Create visualization of annotations"""
        # Create visualization directory
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load image
        img = cv2.imread(image_path)
        
        # Draw bounding boxes
        for obj in annotation['objects']:
            obj_type = obj['type']
            bbox = obj['bbox']
            
            # Get color for class
            color = self.class_colors.get(obj_type, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(
                img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2
            )
            
            # Draw label
            cv2.putText(
                img,
                obj_type,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Save visualization
        cv2.imwrite(os.path.join(vis_dir, f"{frame_id}.png"), img)
    
    def upload_annotations_to_s3(self):
        """Upload annotations to S3"""
        if self.s3_client is None:
            return False
        
        logging.info(f"Uploading annotations to S3 bucket: {self.s3_bucket}/{self.s3_prefix}")
        
        # Walk through output directory and upload all files
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_key = os.path.join(
                    self.s3_prefix,
                    'annotations',
                    os.path.relpath(local_path, self.output_dir)
                )
                
                try:
                    self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                    logging.debug(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_key}")
                except Exception as e:
                    logging.error(f"Error uploading to S3: {e}")
        
        logging.info("Annotation upload completed")
        return True


def main():
    """Main function"""
    argparser = argparse.ArgumentParser(description='CARLA Data Annotation')
    argparser.add_argument('--input-dir', required=True, help='Input directory containing collected data')
    argparser.add_argument('--output-dir', default='./annotations', help='Output directory for annotations')
    argparser.add_argument('--format', default='yolo', choices=['coco', 'pascal_voc', 'yolo'], help='Annotation format')
    argparser.add_argument('--visualize', action='store_true', help='Create visualizations of annotations')
    argparser.add_argument('--s3-bucket', help='S3 bucket for annotation upload')
    argparser.add_argument('--s3-prefix', default='carla_data', help='S3 prefix for annotation upload')
    
    args = argparser.parse_args()
    
    # Create annotator and process dataset
    annotator = DataAnnotator(args)
    success = annotator.process_dataset()
    
    if success:
        logging.info("Annotation completed successfully")
        return 0
    else:
        logging.error("Annotation failed")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.info("User interrupted the annotation process")
        sys.exit(0)
