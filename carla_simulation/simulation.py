#!/usr/bin/env python

"""
Road Object Detection - CARLA Simulation Script
This script handles the generation of synthetic data using CARLA simulator
with a focus on underrepresented classes like cyclists.
"""

import glob
import os
import sys
import time
import json
import argparse
import random
import logging
import numpy as np
from datetime import datetime

# Find CARLA module
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls
from carla import ColorConverter as cc

# Configure logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors.
    """
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)
            return q

        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class CarlaSimulation:
    """
    Main class for CARLA simulation and data generation
    """
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.map = None
        self.blueprint_library = None
        self.traffic_manager = None
        self.vehicle_list = []
        self.walker_list = []
        self.cyclist_list = []
        self.sensor_list = []
        self.all_id = []
        self.weather_presets = self._get_weather_presets()
        
        # Set random seed for reproducibility
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
        
        # Load simulation parameters
        self.sim_params = self._load_simulation_parameters(args.config)
        
        # Connect to CARLA server
        self._connect_to_carla()

    def _connect_to_carla(self):
        """Connect to CARLA server"""
        try:
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(10.0)
            logging.info(f"Connected to CARLA server: {self.args.host}:{self.args.port}")
            
            # Get world and map
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            logging.info(f"Current map: {self.map.name}")
            
            # Get blueprint library
            self.blueprint_library = self.world.get_blueprint_library()
            
            # Set up traffic manager
            self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            self.traffic_manager.set_hybrid_physics_mode(True)
            if self.args.seed is not None:
                self.traffic_manager.set_random_device_seed(self.args.seed)
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
        except Exception as e:
            logging.error(f"Error connecting to CARLA: {e}")
            sys.exit(1)

    def _load_simulation_parameters(self, config_path):
        """Load simulation parameters from JSON file"""
        try:
            with open(config_path, 'r') as f:
                params = json.load(f)
            logging.info(f"Loaded simulation parameters from {config_path}")
            return params
        except Exception as e:
            logging.error(f"Error loading simulation parameters: {e}")
            sys.exit(1)

    def _get_weather_presets(self):
        """Define weather presets for simulation"""
        return {
            'Clear': carla.WeatherParameters.ClearNoon,
            'Cloudy': carla.WeatherParameters.CloudyNoon,
            'Wet': carla.WeatherParameters.WetNoon,
            'WetCloudy': carla.WeatherParameters.WetCloudyNoon,
            'SoftRain': carla.WeatherParameters.SoftRainNoon,
            'MidRain': carla.WeatherParameters.MidRainyNoon,
            'HardRain': carla.WeatherParameters.HardRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
            'CloudySunset': carla.WeatherParameters.CloudySunset,
            'WetSunset': carla.WeatherParameters.WetSunset,
            'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
            'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
            'MidRainSunset': carla.WeatherParameters.MidRainSunset,
            'HardRainSunset': carla.WeatherParameters.HardRainSunset,
        }

    def set_weather(self, weather_name=None):
        """Set weather conditions"""
        if weather_name and weather_name in self.weather_presets:
            weather = self.weather_presets[weather_name]
        else:
            # Randomly select a weather preset
            weather_name = random.choice(list(self.weather_presets.keys()))
            weather = self.weather_presets[weather_name]
        
        self.world.set_weather(weather)
        logging.info(f"Weather set to {weather_name}")
        return weather_name

    def spawn_ego_vehicle(self):
        """Spawn ego vehicle for data collection"""
        # Get vehicle blueprint
        vehicle_bp = self.blueprint_library.find(self.sim_params['simulation_parameters']['ego_vehicle'])
        
        # Get spawn points and randomly select one
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logging.error("No spawn points available in the map")
            return None
        
        spawn_point = random.choice(spawn_points)
        
        # Spawn the vehicle
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            logging.error("Failed to spawn ego vehicle")
            return None
        
        logging.info(f"Spawned ego vehicle: {vehicle.type_id}")
        self.vehicle_list.append(vehicle)
        
        # Set autopilot
        vehicle.set_autopilot(True, self.traffic_manager.get_port())
        
        # Set vehicle light state
        light_state = vls.Position | vls.LowBeam
        vehicle.set_light_state(carla.VehicleLightState(light_state))
        
        return vehicle

    def setup_sensors(self, vehicle):
        """Set up sensors for data collection"""
        sensors = []
        sensor_configs = self.sim_params['simulation_parameters']['data_collection']['sensors']
        
        # RGB Camera
        if 'rgb_camera' in sensor_configs:
            rgb_cam_bp = self.blueprint_library.find('sensor.camera.rgb')
            rgb_cam_bp.set_attribute('image_size_x', str(sensor_configs['rgb_camera']['width']))
            rgb_cam_bp.set_attribute('image_size_y', str(sensor_configs['rgb_camera']['height']))
            rgb_cam_bp.set_attribute('fov', str(sensor_configs['rgb_camera']['fov']))
            
            cam_transform = carla.Transform(
                carla.Location(
                    x=sensor_configs['rgb_camera']['position'][0],
                    y=sensor_configs['rgb_camera']['position'][1],
                    z=sensor_configs['rgb_camera']['position'][2]
                ),
                carla.Rotation(
                    pitch=sensor_configs['rgb_camera']['rotation'][0],
                    yaw=sensor_configs['rgb_camera']['rotation'][1],
                    roll=sensor_configs['rgb_camera']['rotation'][2]
                )
            )
            
            rgb_camera = self.world.spawn_actor(rgb_cam_bp, cam_transform, attach_to=vehicle)
            sensors.append(rgb_camera)
            self.sensor_list.append(rgb_camera)
            logging.info("RGB camera sensor attached")
        
        # Depth Camera
        if 'depth_camera' in sensor_configs:
            depth_cam_bp = self.blueprint_library.find('sensor.camera.depth')
            depth_cam_bp.set_attribute('image_size_x', str(sensor_configs['depth_camera']['width']))
            depth_cam_bp.set_attribute('image_size_y', str(sensor_configs['depth_camera']['height']))
            depth_cam_bp.set_attribute('fov', str(sensor_configs['depth_camera']['fov']))
            
            cam_transform = carla.Transform(
                carla.Location(
                    x=sensor_configs['depth_camera']['position'][0],
                    y=sensor_configs['depth_camera']['position'][1],
                    z=sensor_configs['depth_camera']['position'][2]
                ),
                carla.Rotation(
                    pitch=sensor_configs['depth_camera']['rotation'][0],
                    yaw=sensor_configs['depth_camera']['rotation'][1],
                    roll=sensor_configs['depth_camera']['rotation'][2]
                )
            )
            
            depth_camera = self.world.spawn_actor(depth_cam_bp, cam_transform, attach_to=vehicle)
            depth_camera.listen(lambda image: image.save_to_disk(f'output/depth/{image.frame:06d}.png', cc.LogarithmicDepth))
            sensors.append(depth_camera)
            self.sensor_list.append(depth_camera)
            logging.info("Depth camera sensor attached")
        
        # Semantic Segmentation Camera
        if 'semantic_segmentation' in sensor_configs:
            sem_cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            sem_cam_bp.set_attribute('image_size_x', str(sensor_configs['semantic_segmentation']['width']))
            sem_cam_bp.set_attribute('image_size_y', str(sensor_configs['semantic_segmentation']['height']))
            sem_cam_bp.set_attribute('fov', str(sensor_configs['semantic_segmentation']['fov']))
            
            cam_transform = carla.Transform(
                carla.Location(
                    x=sensor_configs['semantic_segmentation']['position'][0],
                    y=sensor_configs['semantic_segmentation']['position'][1],
                    z=sensor_configs['semantic_segmentation']['position'][2]
                ),
                carla.Rotation(
                    pitch=sensor_configs['semantic_segmentation']['rotation'][0],
                    yaw=sensor_configs['semantic_segmentation']['rotation'][1],
                    roll=sensor_configs['semantic_segmentation']['rotation'][2]
                )
            )
            
            sem_camera = self.world.spawn_actor(sem_cam_bp, cam_transform, attach_to=vehicle)
            sem_camera.listen(lambda image: image.save_to_disk(f'output/semantic/{image.frame:06d}.png', cc.CityScapesPalette))
            sensors.append(sem_camera)
            self.sensor_list.append(sem_camera)
            logging.info("Semantic segmentation camera sensor attached")
        
        # LiDAR
        if 'lidar' in sensor_configs:
            lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels', str(sensor_configs['lidar']['channels']))
            lidar_bp.set_attribute('range', str(sensor_configs['lidar']['range']))
            lidar_bp.set_attribute('points_per_second', str(sensor_configs['lidar']['points_per_second']))
            lidar_bp.set_attribute('rotation_frequency', str(sensor_configs['lidar']['rotation_frequency']))
            
            lidar_transform = carla.Transform(
                carla.Location(
                    x=sensor_configs['lidar']['position'][0],
                    y=sensor_configs['lidar']['position'][1],
                    z=sensor_configs['lidar']['position'][2]
                )
            )
            
            lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
            lidar.listen(lambda point_cloud: point_cloud.save_to_disk(f'output/lidar/{point_cloud.frame:06d}.ply'))
            sensors.append(lidar)
            self.sensor_list.append(lidar)
            logging.info("LiDAR sensor attached")
        
        return sensors

    def spawn_npc_vehicles(self):
        """Spawn NPC vehicles in the simulation"""
        vehicle_count = self.sim_params['simulation_parameters']['actors']['vehicles']['count']
        
        # Get vehicle blueprints
        vehicle_bps = self.blueprint_library.filter('vehicle.*')
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels').as_int()) == 4]
        
        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        if len(spawn_points) < vehicle_count:
            vehicle_count = len(spawn_points)
            logging.warning(f"Not enough spawn points for vehicles. Spawning {vehicle_count} vehicles instead.")
        
        # Spawn vehicles
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor
        
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= vehicle_count:
                break
            
            vehicle_bp = random.choice(vehicle_bps)
            
            # Set random color
            if vehicle_bp.has_attribute('color'):
                color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)
            
            # Set role name
            vehicle_bp.set_attribute('role_name', 'npc_vehicle')
            
            # Set vehicle light state
            light_state = vls.Position | vls.LowBeam
            
            # Add command to spawn vehicle with autopilot
            batch.append(
                SpawnActor(vehicle_bp, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state))
            )
        
        # Apply batch and store vehicle IDs
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(f"Error spawning NPC vehicle: {response.error}")
            else:
                self.vehicle_list.append(self.world.get_actor(response.actor_id))
        
        logging.info(f"Spawned {len(self.vehicle_list) - 1} NPC vehicles")  # -1 for ego vehicle
        
        # Set traffic manager behavior
        speed_factor = self.sim_params['simulation_parameters']['actors']['vehicles']['speed_factor']
        self.traffic_manager.global_percentage_speed_difference(100.0 * (1.0 - speed_factor))
        
        return self.vehicle_list

    def spawn_npc_pedestrians(self):
        """Spawn NPC pedestrians in the simulation"""
        pedestrian_count = self.sim_params['simulation_parameters']['actors']['pedestrians']['count']
        
        # Get pedestrian blueprints
        pedestrian_bps = self.blueprint_library.filter('walker.pedestrian.*')
        
        # Spawn pedestrians
        spawn_points = []
        for i in range(pedestrian_count):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        
        # Spawn walker actors
        batch = []
        walker_speeds = []
        
        for spawn_point in spawn_points:
            walker_bp = random.choice(pedestrian_bps)
            
            # Set not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            
            # Set speed
            if walker_bp.has_attribute('speed'):
                walker_speed = float(walker_bp.get_attribute('speed').recommended_values[1])
                walker_speeds.append(walker_speed)
            else:
                walker_speeds.append(1.0)
            
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        
        # Apply batch and store walker IDs
        walkers = []
        for i, response in enumerate(self.client.apply_batch_sync(batch, True)):
            if response.error:
                logging.error(f"Error spawning pedestrian: {response.error}")
            else:
                walkers.append({"id": response.actor_id, "speed": walker_speeds[i]})
        
        # Spawn walker controllers
        batch = []
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        
        for walker in walkers:
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker["id"]))
        
        # Apply batch and store controller IDs
        for i, response in enumerate(self.client.apply_batch_sync(batch, True)):
            if response.error:
                logging.error(f"Error spawning pedestrian controller: {response.error}")
            else:
                walkers[i]["con"] = response.actor_id
                self.all_id.append(response.actor_id)
                self.all_id.append(walkers[i]["id"])
        
        # Get all walker actors
        all_actors = self.world.get_actors(self.all_id)
        self.walker_list = all_actors
        
        # Start walker controllers
        for i in range(0, len(self.all_id), 2):
            # Set walker controller
            all_actors[i].start()
            # Set random destination
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # Set walking speed
            all_actors[i].set_max_speed(walkers[int(i/2)]["speed"])
        
        logging.info(f"Spawned {len(walkers)} pedestrians")
        return self.walker_list

    def spawn_npc_cyclists(self):
        """Spawn NPC cyclists in the simulation with focus on this underrepresented class"""
        cyclist_count = self.sim_params['simulation_parameters']['actors']['cyclists']['count']
        
        # Get bicycle blueprints
        bicycle_bps = self.blueprint_library.filter('vehicle.bh.crossbike')
        if len(bicycle_bps) == 0:
            bicycle_bps = self.blueprint_library.filter('vehicle.diamondback.century')
        
        # Get pedestrian blueprints for cyclists
        cyclist_bps = self.blueprint_library.filter('walker.pedestrian.*')
        
        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        if len(spawn_points) < cyclist_count:
            cyclist_count = len(spawn_points)
            logging.warning(f"Not enough spawn points for cyclists. Spawning {cyclist_count} cyclists instead.")
        
        # Spawn cyclists (bicycle + rider)
        for i in range(cyclist_count):
            if i >= len(spawn_points):
                break
            
            # Spawn bicycle
            bicycle_bp = random.choice(bicycle_bps)
            bicycle_bp.set_attribute('role_name', 'npc_cyclist')
            
            bicycle = self.world.try_spawn_actor(bicycle_bp, spawn_points[i])
            if bicycle is None:
                logging.warning(f"Failed to spawn bicycle at {i}")
                continue
            
            self.cyclist_list.append(bicycle)
            
            # Set bicycle autopilot
            bicycle.set_autopilot(True, self.traffic_manager.get_port())
            
            # Set bicycle speed (slower than cars)
            speed_factor = self.sim_params['simulation_parameters']['actors']['cyclists']['speed_factor']
            self.traffic_manager.vehicle_percentage_speed_difference(bicycle, 100.0 * (1.0 - speed_factor))
            
            # Set bicycle to keep right lane
            self.traffic_manager.set_desired_speed(bicycle, 15.0)  # km/h
            self.traffic_manager.force_lane_change(bicycle, False)
            self.traffic_manager.set_route_option(bicycle, carla.MapConnectionOption.RoadOption.STRAIGHT)
            
            # Spawn rider on bicycle
            cyclist_bp = random.choice(cyclist_bps)
            cyclist_transform = carla.Transform(
                carla.Location(z=0.5),  # Offset to place rider on bicycle
                carla.Rotation()
            )
            
            cyclist = self.world.try_spawn_actor(cyclist_bp, cyclist_transform, attach_to=bicycle)
            if cyclist is not None:
                self.cyclist_list.append(cyclist)
        
        logging.info(f"Spawned {len(self.cyclist_list)} cyclist actors")
        return self.cyclist_list

    def generate_data(self):
        """Main method to generate synthetic data"""
        try:
            # Create output directories
            output_dir = os.path.join(os.getcwd(), 'output')
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'semantic'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'lidar'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
            
            # Set weather
            weather_name = self.set_weather()
            
            # Spawn ego vehicle
            ego_vehicle = self.spawn_ego_vehicle()
            if ego_vehicle is None:
                return False
            
            # Setup sensors
            sensors = self.setup_sensors(ego_vehicle)
            
            # Spawn NPCs
            self.spawn_npc_vehicles()
            self.spawn_npc_pedestrians()
            self.spawn_npc_cyclists()  # Focus on cyclists as underrepresented class
            
            # Generate data
            frames_per_second = self.sim_params['simulation_parameters']['data_collection']['frames_per_second']
            duration_seconds = self.sim_params['simulation_parameters']['data_collection']['duration_seconds']
            total_frames = frames_per_second * duration_seconds
            
            logging.info(f"Generating {total_frames} frames of data at {frames_per_second} FPS")
            
            # Main data collection loop
            for frame in range(total_frames):
                # Tick the world
                self.world.tick()
                
                # Process sensor data and generate annotations
                if frame % (frames_per_second // 10) == 0:  # Log progress every 10% of a second
                    logging.info(f"Generated frame {frame}/{total_frames}")
                
                # Sleep to maintain real-time factor if needed
                time.sleep(0.001)
            
            logging.info("Data generation completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error during data generation: {e}")
            return False
        finally:
            # Clean up
            self.cleanup()

    def cleanup(self):
        """Clean up all actors"""
        logging.info("Cleaning up actors...")
        
        # Destroy sensors
        for sensor in self.sensor_list:
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        
        # Destroy vehicles
        for vehicle in self.vehicle_list:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
        
        # Destroy walkers and controllers
        for actor in self.walker_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        
        # Destroy cyclists
        for cyclist in self.cyclist_list:
            if cyclist is not None and cyclist.is_alive:
                cyclist.destroy()
        
        logging.info("Cleanup completed")


def main():
    """Main function"""
    argparser = argparse.ArgumentParser(description='CARLA Synthetic Data Generator')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--tm-port', default=8000, type=int, help='Port to communicate with TM (default: 8000)')
    argparser.add_argument('--config', default='simulation_config.json', help='Path to simulation configuration file')
    argparser.add_argument('--seed', type=int, help='Set seed for reproducibility')
    argparser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    
    args = argparser.parse_args()
    
    # Create and run simulation
    simulation = CarlaSimulation(args)
    success = simulation.generate_data()
    
    if success:
        logging.info("Simulation completed successfully")
        return 0
    else:
        logging.error("Simulation failed")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.info("User interrupted the simulation")
        sys.exit(0)
