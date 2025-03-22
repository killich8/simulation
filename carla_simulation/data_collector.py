#!/usr/bin/env python

"""
Road Object Detection - Data Collection Script
This script handles the collection and processing of data from CARLA simulator
"""

import os
import sys
import glob
import time
import json
import argparse
import logging
import numpy as np
import cv2
from datetime import datetime
import queue
import shutil
import boto3
from PIL import Image

# Find CARLA module
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

# Configure logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class DataCollector:
    """
    Class for collecting and processing data from CARLA simulator
    """
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        self.s3_bucket = args.s3_bucket
        self.s3_prefix = args.s3_prefix
        
        # Create output directories
        self._create_output_dirs()
        
        # Initialize S3 client if needed
        self.s3_client = None
        if self.s3_bucket:
            self.s3_client = boto3.client('s3')
            logging.info(f"S3 upload enabled to bucket: {self.s3_bucket}")
    
    def _create_output_dirs(self):
        """Create output directories for data collection"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'semantic'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'lidar'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'annotations'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'metadata'), exist_ok=True)
        logging.info(f"Created output directories in {self.output_dir}")
    
    def process_rgb_image(self, image, frame_id):
        """Process and save RGB image"""
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        
        # Save image
        img_path = os.path.join(self.output_dir, 'rgb', f'{frame_id:06d}.png')
        cv2.imwrite(img_path, array)
        
        return img_path, array
    
    def process_depth_image(self, image, frame_id):
        """Process and save depth image"""
        image.convert(cc.LogarithmicDepth)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        
        # Save image
        img_path = os.path.join(self.output_dir, 'depth', f'{frame_id:06d}.png')
        cv2.imwrite(img_path, array)
        
        return img_path, array
    
    def process_semantic_image(self, image, frame_id):
        """Process and save semantic segmentation image"""
        image.convert(cc.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        
        # Save image
        img_path = os.path.join(self.output_dir, 'semantic', f'{frame_id:06d}.png')
        cv2.imwrite(img_path, array)
        
        return img_path, array
    
    def process_lidar_data(self, point_cloud, frame_id):
        """Process and save LiDAR point cloud data"""
        # Save point cloud as PLY file
        ply_path = os.path.join(self.output_dir, 'lidar', f'{frame_id:06d}.ply')
        point_cloud.save_to_disk(ply_path)
        
        return ply_path
    
    def generate_annotations(self, world, frame_id, rgb_image):
        """Generate annotations for objects in the scene"""
        # Get all actors in the world
        vehicles = world.get_actors().filter('vehicle.*')
        pedestrians = world.get_actors().filter('walker.pedestrian.*')
        
        # Get camera sensor to project 3D points to 2D
        camera = None
        for sensor in world.get_actors().filter('sensor.camera.rgb'):
            camera = sensor
            break
        
        if camera is None:
            logging.error("RGB camera not found, cannot generate annotations")
            return None
        
        # Get camera matrix
        camera_transform = camera.get_transform()
        camera_intrinsics = camera.attributes
        width = int(camera_intrinsics['image_size_x'])
        height = int(camera_intrinsics['image_size_y'])
        fov = float(camera_intrinsics['fov'])
        
        # Create annotation dictionary
        annotations = {
            'frame_id': frame_id,
            'image_width': width,
            'image_height': height,
            'objects': []
        }
        
        # Process vehicles
        for vehicle in vehicles:
            # Skip ego vehicle
            if vehicle.attributes.get('role_name') == 'hero':
                continue
            
            # Get vehicle bounding box and transform to camera view
            bbox = vehicle.bounding_box
            bbox_transform = carla.Transform(bbox.location, bbox.rotation)
            bbox_vehicle_transform = carla.Transform(
                vehicle.get_transform().location,
                vehicle.get_transform().rotation
            )
            bbox_world_transform = bbox_vehicle_transform.transform(bbox_transform)
            
            # Get 3D bounding box corners in world coordinates
            corners = [
                carla.Location(x=-bbox.extent.x, y=-bbox.extent.y, z=-bbox.extent.z),
                carla.Location(x=bbox.extent.x, y=-bbox.extent.y, z=-bbox.extent.z),
                carla.Location(x=bbox.extent.x, y=bbox.extent.y, z=-bbox.extent.z),
                carla.Location(x=-bbox.extent.x, y=bbox.extent.y, z=-bbox.extent.z),
                carla.Location(x=-bbox.extent.x, y=-bbox.extent.y, z=bbox.extent.z),
                carla.Location(x=bbox.extent.x, y=-bbox.extent.y, z=bbox.extent.z),
                carla.Location(x=bbox.extent.x, y=bbox.extent.y, z=bbox.extent.z),
                carla.Location(x=-bbox.extent.x, y=bbox.extent.y, z=bbox.extent.z)
            ]
            
            # Transform corners to world coordinates
            corners_world = [bbox_world_transform.transform(corner) for corner in corners]
            
            # Project 3D points to 2D camera view
            corners_2d = []
            for corner in corners_world:
                corner_vector = carla.Vector3D(
                    x=corner.x - camera_transform.location.x,
                    y=corner.y - camera_transform.location.y,
                    z=corner.z - camera_transform.location.z
                )
                
                # Transform to camera coordinates
                camera_rotation = camera_transform.rotation
                forward = carla.Vector3D(x=1.0, y=0.0, z=0.0)
                forward = camera_rotation.get_forward_vector()
                
                # Check if point is behind camera
                if forward.dot(corner_vector) <= 0:
                    continue
                
                # Project to 2D
                point_2d = self._world_to_camera(corner, camera_transform, width, height, fov)
                if point_2d is not None:
                    corners_2d.append(point_2d)
            
            # If we have at least 2 corners visible, create a 2D bounding box
            if len(corners_2d) >= 2:
                x_coords = [p[0] for p in corners_2d]
                y_coords = [p[1] for p in corners_2d]
                
                # Calculate 2D bounding box
                x_min = max(0, min(x_coords))
                y_min = max(0, min(y_coords))
                x_max = min(width, max(x_coords))
                y_max = min(height, max(y_coords))
                
                # Determine object type
                obj_type = 'car'
                if 'bicycle' in vehicle.type_id or 'bike' in vehicle.type_id:
                    obj_type = 'bicycle'
                elif 'motorcycle' in vehicle.type_id:
                    obj_type = 'motorcycle'
                
                # Add to annotations
                annotations['objects'].append({
                    'type': obj_type,
                    'id': vehicle.id,
                    'bbox': [float(x_min), float(y_min), float(x_max), float(y_max)],
                    '3d_location': [float(vehicle.get_location().x),
                                   float(vehicle.get_location().y),
                                   float(vehicle.get_location().z)]
                })
        
        # Process pedestrians
        for pedestrian in pedestrians:
            # Get pedestrian bounding box and transform to camera view
            bbox = pedestrian.bounding_box
            bbox_transform = carla.Transform(bbox.location, bbox.rotation)
            bbox_pedestrian_transform = carla.Transform(
                pedestrian.get_transform().location,
                pedestrian.get_transform().rotation
            )
            bbox_world_transform = bbox_pedestrian_transform.transform(bbox_transform)
            
            # Get 3D bounding box corners in world coordinates
            corners = [
                carla.Location(x=-bbox.extent.x, y=-bbox.extent.y, z=-bbox.extent.z),
                carla.Location(x=bbox.extent.x, y=-bbox.extent.y, z=-bbox.extent.z),
                carla.Location(x=bbox.extent.x, y=bbox.extent.y, z=-bbox.extent.z),
                carla.Location(x=-bbox.extent.x, y=bbox.extent.y, z=-bbox.extent.z),
                carla.Location(x=-bbox.extent.x, y=-bbox.extent.y, z=bbox.extent.z),
                carla.Location(x=bbox.extent.x, y=-bbox.extent.y, z=bbox.extent.z),
                carla.Location(x=bbox.extent.x, y=bbox.extent.y, z=bbox.extent.z),
                carla.Location(x=-bbox.extent.x, y=bbox.extent.y, z=bbox.extent.z)
            ]
            
            # Transform corners to world coordinates
            corners_world = [bbox_world_transform.transform(corner) for corner in corners]
            
            # Project 3D points to 2D camera view
            corners_2d = []
            for corner in corners_world:
                corner_vector = carla.Vector3D(
                    x=corner.x - camera_transform.location.x,
                    y=corner.y - camera_transform.location.y,
                    z=corner.z - camera_transform.location.z
                )
                
                # Transform to camera coordinates
                camera_rotation = camera_transform.rotation
                forward = carla.Vector3D(x=1.0, y=0.0, z=0.0)
                forward = camera_rotation.get_forward_vector()
                
                # Check if point is behind camera
                if forward.dot(corner_vector) <= 0:
                    continue
                
                # Project to 2D
                point_2d = self._world_to_camera(corner, camera_transform, width, height, fov)
                if point_2d is not None:
                    corners_2d.append(point_2d)
            
            # If we have at least 2 corners visible, create a 2D bounding box
            if len(corners_2d) >= 2:
                x_coords = [p[0] for p in corners_2d]
                y_coords = [p[1] for p in corners_2d]
                
                # Calculate 2D bounding box
                x_min = max(0, min(x_coords))
                y_min = max(0, min(y_coords))
                x_max = min(width, max(x_coords))
                y_max = min(height, max(y_coords))
                
                # Add to annotations
                annotations['objects'].append({
                    'type': 'pedestrian',
                    'id': pedestrian.id,
                    'bbox': [float(x_min), float(y_min), float(x_max), float(y_max)],
                    '3d_location': [float(pedestrian.get_location().x),
                                   float(pedestrian.get_location().y),
                                   float(pedestrian.get_location().z)]
                })
        
        # Save annotations to JSON file
        annotation_path = os.path.join(self.output_dir, 'annotations', f'{frame_id:06d}.json')
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        return annotation_path
    
    def _world_to_camera(self, location, camera_transform, width, height, fov):
        """Convert world coordinates to camera coordinates"""
        # Calculate camera matrix
        K = np.identity(3)
        K[0, 0] = K[1, 1] = width / (2.0 * np.tan(fov * np.pi / 360.0))
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0
        
        # Get world to camera matrix
        world_to_camera = np.array(camera_transform.get_inverse_matrix())
        
        # Convert location to homogeneous coordinates
        point = np.array([location.x, location.y, location.z, 1])
        
        # Transform to camera coordinates
        point_camera = np.dot(world_to_camera, point)
        
        # Handle points behind the camera
        if point_camera[2] <= 0:
            return None
        
        # Project to 2D
        point_2d = np.dot(K, point_camera[:3] / point_camera[2])
        
        return (int(point_2d[0]), int(point_2d[1]))
    
    def save_metadata(self, world, frame_id):
        """Save metadata about the simulation"""
        # Get world information
        weather = world.get_weather()
        
        # Create metadata dictionary
        metadata = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'map_name': world.get_map().name,
            'weather': {
                'cloudiness': weather.cloudiness,
                'precipitation': weather.precipitation,
                'precipitation_deposits': weather.precipitation_deposits,
                'wind_intensity': weather.wind_intensity,
                'sun_azimuth_angle': weather.sun_azimuth_angle,
                'sun_altitude_angle': weather.sun_altitude_angle,
                'fog_density': weather.fog_density,
                'fog_distance': weather.fog_distance,
                'wetness': weather.wetness
            }
        }
        
        # Save metadata to JSON file
        metadata_path = os.path.join(self.output_dir, 'metadata', f'{frame_id:06d}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def upload_to_s3(self, local_path, s3_key):
        """Upload file to S3 bucket"""
        if self.s3_client is None:
            return False
        
        try:
            self.s3_client.upload_file(local_path, self.s3_bucket, f"{self.s3_prefix}/{s3_key}")
            logging.debug(f"Uploaded {local_path} to s3://{self.s3_bucket}/{self.s3_prefix}/{s3_key}")
            return True
        except Exception as e:
            logging.error(f"Error uploading to S3: {e}")
            return False
    
    def upload_dataset_to_s3(self):
        """Upload entire dataset to S3"""
        if self.s3_client is None:
            return False
        
        logging.info(f"Uploading dataset to S3 bucket: {self.s3_bucket}/{self.s3_prefix}")
        
        # Walk through output directory and upload all files
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_key = os.path.relpath(local_path, self.output_dir)
                self.upload_to_s3(local_path, s3_key)
        
        logging.info("Dataset upload completed")
        return True
    
    def create_dataset_archive(self, archive_path=None):
        """Create a compressed archive of the dataset"""
        if archive_path is None:
            archive_path = f"{self.output_dir}.tar.gz"
        
        logging.info(f"Creating dataset archive: {archive_path}")
        
        # Create archive
        shutil.make_archive(
            os.path.splitext(archive_path)[0],
            'gztar',
            os.path.dirname(self.output_dir),
            os.path.basename(self.output_dir)
        )
        
        logging.info(f"Dataset archive created: {archive_path}")
        return archive_path


def main():
    """Main function"""
    argparser = argparse.ArgumentParser(description='CARLA Data Collection')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--output-dir', default='./output', help='Output directory for collected data')
    argparser.add_argument('--s3-bucket', help='S3 bucket for data upload')
    argparser.add_argument('--s3-prefix', default='carla_data', help='S3 prefix for data upload')
    argparser.add_argument('--create-archive', action='store_true', help='Create compressed archive of dataset')
    argparser.add_argument('--archive-path', help='Path for dataset archive')
    
    args = argparser.parse_args()
    
    # Create data collector
    collector = DataCollector(args)
    
    # Connect to CARLA server
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Example of processing a single frame
        # In a real scenario, this would be integrated with the simulation loop
        frame_id = 0
        
        # Get RGB camera
        rgb_camera = None
        for sensor in world.get_actors().filter('sensor.camera.rgb'):
            rgb_camera = sensor
            break
        
        if rgb_camera is None:
            logging.error("RGB camera not found")
            return 1
        
        # Process RGB image
        image = rgb_camera.listen(lambda image: collector.process_rgb_image(image, frame_id))
        
        # Generate annotations
        collector.generate_annotations(world, frame_id, image)
        
        # Save metadata
        collector.save_metadata(world, frame_id)
        
        # Upload to S3 if requested
        if args.s3_bucket:
            collector.upload_dataset_to_s3()
        
        # Create archive if requested
        if args.create_archive:
            collector.create_dataset_archive(args.archive_path)
        
        return 0
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.info("User interrupted the data collection")
        sys.exit(0)
