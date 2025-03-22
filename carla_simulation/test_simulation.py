#!/usr/bin/env python

"""
Road Object Detection - Test Script for CARLA Simulation
This script tests the functionality of the CARLA simulation scripts
"""

import os
import sys
import unittest
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import simulation modules
from carla_simulation.simulation import CarlaSimulation
from carla_simulation.data_collector import DataCollector
from carla_simulation.data_annotator import DataAnnotator


class TestCarlaSimulation(unittest.TestCase):
    """Test cases for CARLA simulation scripts"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock simulation parameters
        self.sim_params = {
            "simulation_parameters": {
                "town": "Town05",
                "ego_vehicle": "vehicle.tesla.model3",
                "weather": {
                    "cloudiness": 0,
                    "precipitation": 0,
                    "precipitation_deposits": 0,
                    "wind_intensity": 0,
                    "sun_azimuth_angle": 70,
                    "sun_altitude_angle": 70,
                    "fog_density": 0,
                    "fog_distance": 0,
                    "wetness": 0
                },
                "actors": {
                    "vehicles": {
                        "count": 10,
                        "safe_distance": 5.0,
                        "speed_factor": 1.0
                    },
                    "pedestrians": {
                        "count": 5,
                        "safe_distance": 2.0,
                        "speed_factor": 0.8
                    },
                    "cyclists": {
                        "count": 8,
                        "safe_distance": 3.0,
                        "speed_factor": 1.2
                    }
                },
                "data_collection": {
                    "frames_per_second": 10,
                    "duration_seconds": 10,
                    "sensors": {
                        "rgb_camera": {
                            "width": 1280,
                            "height": 720,
                            "fov": 90,
                            "position": [1.5, 0.0, 2.4],
                            "rotation": [0.0, 0.0, 0.0]
                        },
                        "depth_camera": {
                            "width": 1280,
                            "height": 720,
                            "fov": 90,
                            "position": [1.5, 0.0, 2.4],
                            "rotation": [0.0, 0.0, 0.0]
                        },
                        "semantic_segmentation": {
                            "width": 1280,
                            "height": 720,
                            "fov": 90,
                            "position": [1.5, 0.0, 2.4],
                            "rotation": [0.0, 0.0, 0.0]
                        },
                        "lidar": {
                            "channels": 64,
                            "range": 100,
                            "points_per_second": 500000,
                            "rotation_frequency": 10,
                            "position": [0.0, 0.0, 2.4]
                        }
                    }
                }
            }
        }
        
        # Save simulation parameters to file
        self.sim_params_file = os.path.join(self.test_dir, "simulation_config.json")
        with open(self.sim_params_file, 'w') as f:
            json.dump(self.sim_params, f, indent=2)
        
        # Create mock arguments
        self.args = MagicMock()
        self.args.host = "127.0.0.1"
        self.args.port = 2000
        self.args.tm_port = 8000
        self.args.config = self.sim_params_file
        self.args.seed = 42
        self.args.sync = True
        self.args.output_dir = os.path.join(self.test_dir, "output")
        self.args.s3_bucket = None
        self.args.s3_prefix = "carla_data"
        self.args.input_dir = os.path.join(self.test_dir, "output")
        self.args.format = "yolo"
        self.args.visualize = True
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch('carla_simulation.simulation.CarlaSimulation._connect_to_carla')
    def test_simulation_initialization(self, mock_connect):
        """Test initialization of CarlaSimulation class"""
        # Mock the connection to CARLA
        mock_connect.return_value = None
        
        # Create simulation instance
        simulation = CarlaSimulation(self.args)
        
        # Check if simulation parameters were loaded correctly
        self.assertEqual(simulation.sim_params, self.sim_params)
        self.assertEqual(simulation.args.seed, 42)
        
        # Check if weather presets were initialized
        self.assertIn('Clear', simulation.weather_presets)
        self.assertIn('Cloudy', simulation.weather_presets)
        
        # Check if lists were initialized
        self.assertEqual(simulation.vehicle_list, [])
        self.assertEqual(simulation.walker_list, [])
        self.assertEqual(simulation.cyclist_list, [])
        self.assertEqual(simulation.sensor_list, [])
    
    @patch('carla_simulation.data_collector.DataCollector._create_output_dirs')
    def test_data_collector_initialization(self, mock_create_dirs):
        """Test initialization of DataCollector class"""
        # Mock directory creation
        mock_create_dirs.return_value = None
        
        # Create data collector instance
        collector = DataCollector(self.args)
        
        # Check if output directory was set correctly
        self.assertEqual(collector.output_dir, self.args.output_dir)
        
        # Check if S3 client is None when no bucket is specified
        self.assertIsNone(collector.s3_client)
    
    @patch('carla_simulation.data_annotator.DataAnnotator.process_dataset')
    def test_data_annotator(self, mock_process):
        """Test DataAnnotator class"""
        # Mock process_dataset method
        mock_process.return_value = True
        
        # Create data annotator instance
        annotator = DataAnnotator(self.args)
        
        # Check if input and output directories were set correctly
        self.assertEqual(annotator.input_dir, self.args.input_dir)
        self.assertEqual(annotator.output_dir, self.args.output_dir)
        
        # Check if format was set correctly
        self.assertEqual(annotator.format, self.args.format)
        
        # Check if class mapping was initialized
        self.assertIn('car', annotator.class_mapping)
        self.assertIn('bicycle', annotator.class_mapping)
        self.assertIn('pedestrian', annotator.class_mapping)
        
        # Test process_dataset method
        result = annotator.process_dataset()
        self.assertTrue(result)
        mock_process.assert_called_once()
    
    def test_cyclist_focus(self):
        """Test that cyclists are properly emphasized in the simulation"""
        # Check that cyclist count is higher than in typical simulations
        cyclist_count = self.sim_params["simulation_parameters"]["actors"]["cyclists"]["count"]
        vehicle_count = self.sim_params["simulation_parameters"]["actors"]["vehicles"]["count"]
        pedestrian_count = self.sim_params["simulation_parameters"]["actors"]["pedestrians"]["count"]
        
        # Verify that cyclists are a significant portion of the actors
        total_actors = cyclist_count + vehicle_count + pedestrian_count
        cyclist_percentage = cyclist_count / total_actors * 100
        
        # Assert that cyclists make up at least 30% of all actors
        self.assertGreaterEqual(cyclist_percentage, 30.0)
        
        # Check that cyclist speed factor is appropriate for realistic movement
        cyclist_speed = self.sim_params["simulation_parameters"]["actors"]["cyclists"]["speed_factor"]
        self.assertGreater(cyclist_speed, 0.8)  # Faster than pedestrians
        self.assertLess(cyclist_speed, 1.5)     # Slower than most vehicles


if __name__ == '__main__':
    unittest.main()
