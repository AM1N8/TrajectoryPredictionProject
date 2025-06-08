#!/usr/bin/env python

"""
CARLA Data Collection Script
----------------------------
This script collects comprehensive data from a CARLA simulation including:
- Ego vehicle data (position, control, orientation, velocity, acceleration)
- Map and road data (lane geometry, lane information)
- LiDAR 360-degree point cloud data (replaces spatial points)
- Options to spawn additional vehicles and control traffic lights

The script can disable traffic lights and stop signs for uninterrupted driving.
Data is stored in CSV format for time-series analysis, and LiDAR data in PLY format.

Requirements:
- CARLA 0.9.14
- Python 3.7+
- numpy, csv, math, time modules
"""

import glob
import os
import sys
import numpy as np
import csv
import math
import time
import json
import argparse
import random
from datetime import datetime

try:
    # Add CARLA Python API to path
    carla_path = glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(carla_path)
except IndexError:
    # For Windows, try the default location
    try:
        sys.path.append(glob.glob('PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass

import carla


class CARLADataCollector:
    def __init__(self, args):
        # Initialize configuration parameters
        self.args = args
        self.world = None
        self.client = None
        self.ego_vehicle = None
        self.spawn_point = None
        self.map = None
        self.data_path = args.output_dir
        self.fps = args.fps
        self.radius = args.radius
        self.synchronous_mode = args.sync
        self.current_waypoint = None
        self.destination = None
        self.traffic_manager = None
        self.other_vehicles = []
        self.lidar_sensor = None
        self.keep_traffic_lights = args.keep_traffic_lights
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            
        # Create LiDAR data directory
        self.lidar_path = os.path.join(self.data_path, "lidar")
        if not os.path.exists(self.lidar_path):
            os.makedirs(self.lidar_path)
            
        # Initialize data files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ego_data_file = os.path.join(self.data_path, f"ego_data_{timestamp}.csv")
        self.lane_data_file = os.path.join(self.data_path, f"lane_data_{timestamp}.csv")
        
        # Initialize CSV writers
        self._init_csv_files()
        
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Ego vehicle data CSV
        with open(self.ego_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'x', 'y', 'z', 'steering', 'throttle', 'brake',
                'yaw', 'heading', 'velocity_x', 'velocity_y', 'velocity_z',
                'accel_x', 'accel_y', 'accel_z'
            ])
            
        # Lane data CSV
        with open(self.lane_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'lane_id', 'lane_width', 'waypoint_x', 'waypoint_y',
                'waypoint_z', 'is_junction', 'lane_type', 'is_drivable'
            ])
    
    def connect_to_carla(self):
        """Connect to CARLA server and setup the simulation"""
        try:
            # Connect to CARLA server
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(10.0)
            
            # Get world and map references
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            
            # Set up traffic manager
            self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
            self.traffic_manager.set_random_device_seed(self.args.seed)
            
            # Set synchronous mode if enabled
            settings = self.world.get_settings()
            if self.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / self.fps
                self.traffic_manager.set_synchronous_mode(True)
            self.world.apply_settings(settings)
            
            print(f"Connected to CARLA server: {self.world.get_map().name}")
            return True
        except Exception as e:
            print(f"Error connecting to CARLA: {e}")
            return False
    
    def spawn_ego_vehicle(self):
        """Spawn the ego vehicle at a suitable spawn point"""
        try:
            # Get blueprint for ego vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('model3')[0]  # Tesla Model 3
            if not vehicle_bp:
                # Fallback to a default vehicle if Model 3 is not available
                vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            
            # Set vehicle to always spawn
            if vehicle_bp.has_attribute('role_name'):
                vehicle_bp.set_attribute('role_name', 'ego')
            
            # Get a valid spawn point
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                print("No spawn points found, using default location")
                self.spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2), carla.Rotation())
            else:
                self.spawn_point = random.choice(spawn_points)
            
            # Spawn the vehicle
            self.ego_vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
            
            # Set up autopilot
            self.ego_vehicle.set_autopilot(True, self.args.tm_port)
            
            # Get initial waypoint
            self.current_waypoint = self.map.get_waypoint(self.ego_vehicle.get_location())
            
            # Set initial destination
            self.set_new_destination()
            
            # Attach LiDAR sensor
            self.attach_lidar_sensor()
            
            print(f"Spawned ego vehicle at {self.spawn_point.location}")
            return True
        except Exception as e:
            print(f"Error spawning vehicle: {e}")
            return False
    
    def attach_lidar_sensor(self):
        """Attach LiDAR sensor to the ego vehicle"""
        try:
            if self.ego_vehicle:
                blueprint_library = self.world.get_blueprint_library()
                lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
                
                # Configure LiDAR parameters
                lidar_bp.set_attribute('channels', '64')  # Number of lasers/channels
                lidar_bp.set_attribute('points_per_second', '100000')
                lidar_bp.set_attribute('rotation_frequency', '10')  # 10 Hz rotation
                lidar_bp.set_attribute('range', str(self.radius))  # Same as the radius parameter
                lidar_bp.set_attribute('upper_fov', '10')
                lidar_bp.set_attribute('lower_fov', '-30')
                
                # Location relative to the vehicle
                lidar_transform = carla.Transform(carla.Location(x=0, z=2))  # 2m above the vehicle
                
                # Spawn and attach the LiDAR sensor
                self.lidar_sensor = self.world.spawn_actor(
                    lidar_bp, 
                    lidar_transform, 
                    attach_to=self.ego_vehicle
                )
                
                # Set up LiDAR callback function
                self.lidar_sensor.listen(self.process_lidar_data)
                
                print("LiDAR sensor attached to ego vehicle")
            else:
                print("Ego vehicle not available, couldn't attach LiDAR sensor")
        except Exception as e:
            print(f"Error attaching LiDAR sensor: {e}")
    
    def process_lidar_data(self, point_cloud):
        """Process LiDAR point cloud data and save to PLY file"""
        try:
            # Get snapshot timestamp
            timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
            
            # Convert point cloud to numpy array
            data = np.fromstring(bytes(point_cloud.raw_data), dtype=np.float32)
            data = np.reshape(data, (int(data.shape[0] / 4), 4))
            
            # Extract points
            points = data[:, :3]  # X, Y, Z
            intensities = data[:, 3]  # Intensity
            
            # Create PLY file path
            filename = os.path.join(self.lidar_path, f"lidar_{timestamp:.6f}.ply")
            
            # Save to PLY format
            self.save_to_ply(filename, points, intensities)
            
        except Exception as e:
            print(f"Error processing LiDAR data: {e}")
    
    def save_to_ply(self, filename, points, intensities):
        """Save point cloud to PLY file format"""
        try:
            # Number of points
            num_points = len(points)
            
            # Write PLY header
            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {num_points}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property float intensity\n")
                f.write("end_header\n")
                
                # Write point data
                for i in range(num_points):
                    x, y, z = points[i]
                    intensity = intensities[i]
                    f.write(f"{x} {y} {z} {intensity}\n")
                    
        except Exception as e:
            print(f"Error saving to PLY: {e}")
    
    def spawn_other_vehicles(self):
        """Spawn additional vehicles"""
        if self.args.num_vehicles <= 0:
            return
            
        try:
            print(f"Spawning {self.args.num_vehicles} additional vehicles...")
            
            # Get blueprint library
            blueprint_library = self.world.get_blueprint_library()
            car_blueprints = blueprint_library.filter('vehicle.*')
            
            # Get spawn points
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                print("No spawn points available for other vehicles")
                return
                
            # Limit number of vehicles to spawn points available
            num_to_spawn = min(self.args.num_vehicles, len(spawn_points) - 1)  # -1 for ego vehicle
            
            # Prepare batch of spawn commands
            batch = []
            for i in range(num_to_spawn):
                vehicle_bp = random.choice(car_blueprints)
                
                # Try to spawn vehicles with autopilot
                if vehicle_bp.has_attribute('color'):
                    color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                    vehicle_bp.set_attribute('color', color)
                
                # Set role name to identify traffic vehicles
                if vehicle_bp.has_attribute('role_name'):
                    vehicle_bp.set_attribute('role_name', 'traffic')
                    
                # Choose a spawn point different from ego's
                spawn_point = random.choice(spawn_points)
                while spawn_point.location.distance(self.spawn_point.location) < 5.0:
                    spawn_point = random.choice(spawn_points)
                
                batch.append(carla.command.SpawnActor(vehicle_bp, spawn_point)
                             .then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.args.tm_port)))
            
            # Execute batch and get the IDs of spawned vehicles
            results = self.client.apply_batch_sync(batch, True)
            
            # Store the IDs of spawned vehicles
            for result in results:
                if not result.error:
                    self.other_vehicles.append(result.actor_id)
            
            print(f"Successfully spawned {len(self.other_vehicles)} additional vehicles")
            
        except Exception as e:
            print(f"Error spawning other vehicles: {e}")
    
    def manage_traffic_lights(self):
        """Manage traffic lights based on user preference"""
        if not self.keep_traffic_lights:
            # Disable traffic rules
            self.disable_traffic_rules()
        else:
            print("Traffic lights are kept active")
    
    def disable_traffic_rules(self):
        """Disable traffic lights and stop signs"""
        try:
            # Set all traffic lights to green
            for traffic_light in self.world.get_actors().filter('traffic.traffic_light'):
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.freeze(True)  # Keep it green permanently
            
            # Configure the ego vehicle's autopilot to ignore traffic lights and stop signs
            if self.traffic_manager and self.ego_vehicle:
                self.traffic_manager.ignore_lights_percentage(self.ego_vehicle, 100)
                self.traffic_manager.ignore_signs_percentage(self.ego_vehicle, 100)
                self.traffic_manager.ignore_vehicles_percentage(self.ego_vehicle, 50)  # Partially ignore other vehicles
                
                # Set global speed limit factor
                self.traffic_manager.global_percentage_speed_difference(-30)  # 30% faster than speed limit
                
                print("Traffic rules disabled for ego vehicle")
            else:
                print("Traffic manager or ego vehicle not available, couldn't disable traffic rules")
        except Exception as e:
            print(f"Error disabling traffic rules: {e}")
    
    def set_new_destination(self):
        """Set a new random destination for the ego vehicle"""
        if not self.ego_vehicle or not self.map:
            return
            
        try:
            # Get random spawn point as destination
            spawn_points = self.map.get_spawn_points()
            if spawn_points:
                # Choose a destination different from current position
                destinations = spawn_points.copy()
                current_location = self.ego_vehicle.get_location()
                
                # Filter out spawn points that are too close
                valid_destinations = [d for d in destinations 
                                     if d.location.distance(current_location) > 100.0]
                
                if valid_destinations:
                    self.destination = random.choice(valid_destinations).location
                else:
                    # If no valid destinations found, pick a random one
                    self.destination = random.choice(destinations).location
                
                # Set destination for traffic manager
                if self.traffic_manager:
                    self.traffic_manager.set_path(self.ego_vehicle, [self.destination])
                    print(f"Set new destination at {self.destination}")
            else:
                print("No spawn points available for destination")
        except Exception as e:
            print(f"Error setting new destination: {e}")
    
    def check_destination_reached(self):
        """Check if current destination is reached and set a new one if needed"""
        if not self.ego_vehicle or not self.destination:
            return
            
        current_location = self.ego_vehicle.get_location()
        if current_location.distance(self.destination) < 10.0:  # Within 10 meters
            print("Destination reached, setting new destination")
            self.set_new_destination()
    
    def collect_ego_data(self, timestamp):
        """Collect and save ego vehicle data"""
        if not self.ego_vehicle:
            return
        
        try:
            # Get vehicle transform, velocity, and control
            transform = self.ego_vehicle.get_transform()
            velocity = self.ego_vehicle.get_velocity()
            acceleration = self.ego_vehicle.get_acceleration()
            control = self.ego_vehicle.get_control()
            
            # Calculate heading from forward vector
            forward_vec = transform.get_forward_vector()
            heading = math.degrees(math.atan2(forward_vec.y, forward_vec.x))
            
            # Write data to CSV
            with open(self.ego_data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    transform.location.x, transform.location.y, transform.location.z,
                    control.steer, control.throttle, control.brake,
                    transform.rotation.yaw, heading,
                    velocity.x, velocity.y, velocity.z,
                    acceleration.x, acceleration.y, acceleration.z
                ])
        except Exception as e:
            print(f"Error collecting ego data: {e}")
    
    def collect_lane_data(self, timestamp):
        """Collect and save lane data around ego vehicle"""
        if not self.ego_vehicle or not self.map:
            return
        
        try:
            # Get vehicle location
            vehicle_location = self.ego_vehicle.get_location()
            
            # Get current vehicle waypoint
            current_waypoint = self.map.get_waypoint(vehicle_location)
            self.current_waypoint = current_waypoint
            
            # Get nearby waypoints
            waypoints = []
            
            # Add current waypoint
            waypoints.append(current_waypoint)
            
            # Add waypoints ahead
            next_waypoints = current_waypoint.next(2.0)  # 2m spacing
            for _ in range(10):  # Get 10 waypoints ahead
                if next_waypoints:
                    waypoint = next_waypoints[0]
                    waypoints.append(waypoint)
                    next_waypoints = waypoint.next(2.0)
                else:
                    break
            
            # Add waypoints behind
            prev_waypoints = [current_waypoint.previous(2.0)[0]] if current_waypoint.previous(2.0) else []
            for _ in range(5):  # Get 5 waypoints behind
                if prev_waypoints:
                    waypoint = prev_waypoints[0]
                    waypoints.append(waypoint)
                    prev_waypoints = waypoint.previous(2.0) if waypoint.previous(2.0) else []
                else:
                    break
            
            # Add waypoints on left and right lanes if they exist
            if current_waypoint.lane_change & carla.LaneChange.Left:
                left_waypoint = current_waypoint.get_left_lane()
                if left_waypoint:
                    waypoints.append(left_waypoint)
            
            if current_waypoint.lane_change & carla.LaneChange.Right:
                right_waypoint = current_waypoint.get_right_lane()
                if right_waypoint:
                    waypoints.append(right_waypoint)
            
            # Write lane data to CSV
            with open(self.lane_data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                for i, waypoint in enumerate(waypoints):
                    lane_id = f"{waypoint.road_id}_{waypoint.lane_id}"
                    
                    writer.writerow([
                        timestamp,
                        lane_id,
                        waypoint.lane_width,
                        waypoint.transform.location.x,
                        waypoint.transform.location.y,
                        waypoint.transform.location.z,
                        waypoint.is_junction,
                        str(waypoint.lane_type),
                        waypoint.lane_type != carla.LaneType.Shoulder and 
                        waypoint.lane_type != carla.LaneType.Parking
                    ])
        except Exception as e:
            print(f"Error collecting lane data: {e}")
    
    def run_simulation(self):
        """Run the data collection simulation"""
        frame_count = 0
        try:
            print("Starting data collection simulation...")
            last_destination_check = 0
            
            while True:
                # Update simulation if in synchronous mode
                if self.synchronous_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                
                # Get simulation timestamp
                timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
                
                # Check if we need a new destination (every 10 seconds)
                if timestamp - last_destination_check > 10.0:
                    self.check_destination_reached()
                    last_destination_check = timestamp
                
                # Collect data
                self.collect_ego_data(timestamp)
                self.collect_lane_data(timestamp)
                # LiDAR data is collected automatically via callback
                
                frame_count += 1
                if frame_count % self.fps == 0:
                    print(f"Collected data for frame {frame_count}, time: {timestamp:.2f}s")
                
                # Optional: exit after certain frames
                if self.args.max_frames > 0 and frame_count >= self.args.max_frames:
                    print(f"Reached maximum frames ({self.args.max_frames}), exiting...")
                    break
                    
                # Sleep to maintain frame rate in asynchronous mode
                if not self.synchronous_mode:
                    time.sleep(1.0 / self.fps)
                    
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        finally:
            # Clean up
            self.cleanup()
            print(f"Simulation ended. Collected data for {frame_count} frames.")
            print(f"Data saved to {self.data_path}")
            print(f"LiDAR data saved to {self.lidar_path}")
    
    def cleanup(self):
        """Clean up all spawned actors"""
        try:
            # Destroy LiDAR sensor
            if self.lidar_sensor:
                self.lidar_sensor.destroy()
                
            # Destroy ego vehicle
            if self.ego_vehicle:
                self.ego_vehicle.destroy()
                
            # Destroy other vehicles
            if self.other_vehicles:
                for vehicle_id in self.other_vehicles:
                    actor = self.world.get_actor(vehicle_id)
                    if actor:
                        actor.destroy()
                        
            # Reset synchronous mode
            if self.synchronous_mode:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                
            print("All spawned actors have been cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--tm-port', default=8000, type=int, help='Traffic Manager port')
    parser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    parser.add_argument('--fps', default=10, type=int, help='Simulation frames per second')
    parser.add_argument('--radius', default=100.0, type=float, help='LiDAR range (meters)')
    parser.add_argument('--output-dir', default='./carla_data', help='Output directory for data files')
    parser.add_argument('--max-frames', default=0, type=int, help='Maximum frames to collect (0 for unlimited)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for traffic manager')
    parser.add_argument('--num-vehicles', default=0, type=int, help='Number of other vehicles to spawn')
    parser.add_argument('--keep-traffic-lights', action='store_true', help='Keep traffic lights active (disable by default)')
    args = parser.parse_args()
    
    # Create and run data collector
    collector = CARLADataCollector(args)
    if collector.connect_to_carla():
        if collector.spawn_ego_vehicle():
            collector.spawn_other_vehicles()
            collector.manage_traffic_lights()
            collector.run_simulation()
        else:
            print("Failed to spawn ego vehicle. Exiting.")
    else:
        print("Failed to connect to CARLA. Exiting.")

if __name__ == '__main__':
    main()