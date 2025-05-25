#!/usr/bin/env python

"""
CARLA Test Sequence Generator
-----------------------------
This script processes collected CARLA data and creates test sequences for trajectory prediction models.
Each test sequence contains:
- 10 past ego vehicle data points (CSV)
- 10 corresponding LiDAR point clouds (PLY files)
- Ground truth future positions for evaluation

The script organizes data into folders for easy model testing and evaluation.

Requirements:
- pandas
- numpy
- os, shutil, glob modules
- Input data from CARLA data collection script
"""

import os
import pandas as pd
import numpy as np
import shutil
import glob
import argparse
from datetime import datetime
import json


class CARLATestSequenceGenerator:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.sequence_length = args.sequence_length  # Past data points (default: 10)
        self.prediction_length = args.prediction_length  # Future data points to predict (default: 10)
        self.step_size = args.step_size  # Step between sequences (default: 1)
        self.min_velocity = args.min_velocity  # Minimum velocity to consider (filter stationary periods)
        
        # File paths
        self.ego_csv_path = None
        self.lidar_dir = None
        self.ego_data = None
        self.lidar_files = []
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def find_data_files(self):
        """Find ego data CSV and LiDAR directory"""
        try:
            # Find ego data CSV file
            ego_files = glob.glob(os.path.join(self.data_dir, "ego_data_*.csv"))
            if not ego_files:
                print(f"No ego data CSV files found in {self.data_dir}")
                return False
            
            # Use the most recent file if multiple exist
            self.ego_csv_path = max(ego_files, key=os.path.getctime)
            print(f"Found ego data file: {self.ego_csv_path}")
            
            # Find LiDAR directory
            self.lidar_dir = os.path.join(self.data_dir, "lidar")
            if not os.path.exists(self.lidar_dir):
                print(f"LiDAR directory not found: {self.lidar_dir}")
                return False
            
            # Get LiDAR files
            self.lidar_files = glob.glob(os.path.join(self.lidar_dir, "lidar_*.ply"))
            self.lidar_files.sort()  # Sort by filename (timestamp)
            
            print(f"Found {len(self.lidar_files)} LiDAR files")
            return True
            
        except Exception as e:
            print(f"Error finding data files: {e}")
            return False
    
    def load_ego_data(self):
        """Load and process ego vehicle data"""
        try:
            # Load CSV data
            self.ego_data = pd.read_csv(self.ego_csv_path)
            print(f"Loaded ego data with {len(self.ego_data)} records")
            
            # Calculate velocity magnitude
            self.ego_data['velocity_magnitude'] = np.sqrt(
                self.ego_data['velocity_x']**2 + 
                self.ego_data['velocity_y']**2 + 
                self.ego_data['velocity_z']**2
            )
            
            # Filter out stationary periods if requested
            if self.min_velocity > 0:
                moving_data = self.ego_data[self.ego_data['velocity_magnitude'] >= self.min_velocity]
                print(f"Filtered to {len(moving_data)} records with velocity >= {self.min_velocity} m/s")
                self.ego_data = moving_data.reset_index(drop=True)
            
            return True
            
        except Exception as e:
            print(f"Error loading ego data: {e}")
            return False
    
    def match_lidar_to_ego_data(self):
        """Match LiDAR files to ego data timestamps"""
        try:
            # Extract timestamps from LiDAR filenames
            lidar_timestamps = []
            lidar_file_map = {}
            
            for lidar_file in self.lidar_files:
                filename = os.path.basename(lidar_file)
                # Extract timestamp from filename (format: lidar_timestamp.ply)
                timestamp_str = filename.replace('lidar_', '').replace('.ply', '')
                try:
                    timestamp = float(timestamp_str)
                    lidar_timestamps.append(timestamp)
                    lidar_file_map[timestamp] = lidar_file
                except ValueError:
                    continue
            
            # Match ego data timestamps to closest LiDAR timestamps
            matched_indices = []
            matched_lidar_files = []
            
            for idx, ego_timestamp in enumerate(self.ego_data['timestamp']):
                # Find closest LiDAR timestamp
                closest_lidar_timestamp = min(lidar_timestamps, 
                                            key=lambda x: abs(x - ego_timestamp))
                
                # Only include if the time difference is reasonable (e.g., < 0.5 seconds)
                time_diff = abs(closest_lidar_timestamp - ego_timestamp)
                if time_diff < 0.5:  # 0.5 second tolerance
                    matched_indices.append(idx)
                    matched_lidar_files.append(lidar_file_map[closest_lidar_timestamp])
            
            # Filter ego data to only matched records
            self.ego_data = self.ego_data.iloc[matched_indices].reset_index(drop=True)
            self.lidar_files = matched_lidar_files
            
            print(f"Matched {len(self.ego_data)} ego data records with LiDAR files")
            return True
            
        except Exception as e:
            print(f"Error matching LiDAR to ego data: {e}")
            return False
    
    def create_test_sequences(self):
        """Create test sequences for model evaluation"""
        try:
            total_sequences = 0
            valid_sequences = 0
            
            # Calculate maximum possible sequences
            max_sequences = len(self.ego_data) - self.sequence_length - self.prediction_length + 1
            
            if max_sequences <= 0:
                print(f"Not enough data points. Need at least {self.sequence_length + self.prediction_length} points")
                return False
            
            print(f"Creating test sequences from {len(self.ego_data)} data points...")
            print(f"Sequence length: {self.sequence_length}, Prediction length: {self.prediction_length}")
            
            # Create sequences
            for i in range(0, max_sequences, self.step_size):
                try:
                    sequence_id = f"sequence_{i:06d}"
                    sequence_dir = os.path.join(self.output_dir, sequence_id)
                    
                    # Create sequence directory
                    if not os.path.exists(sequence_dir):
                        os.makedirs(sequence_dir)
                    
                    # Create subdirectories
                    past_dir = os.path.join(sequence_dir, "past")
                    future_dir = os.path.join(sequence_dir, "ground_truth")
                    lidar_dir = os.path.join(sequence_dir, "lidar")
                    
                    for dir_path in [past_dir, future_dir, lidar_dir]:
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                    
                    # Extract past data (input for model)
                    past_start = i
                    past_end = i + self.sequence_length
                    past_data = self.ego_data.iloc[past_start:past_end].copy()
                    
                    # Extract future data (ground truth for evaluation)
                    future_start = past_end
                    future_end = future_start + self.prediction_length
                    future_data = self.ego_data.iloc[future_start:future_end].copy()
                    
                    # Check if we have enough future data
                    if len(future_data) < self.prediction_length:
                        continue
                    
                    # Save past ego data (CSV for model input)
                    past_csv_path = os.path.join(past_dir, "ego_data.csv")
                    past_data.to_csv(past_csv_path, index=False)
                    
                    # Save future ego data (ground truth for evaluation)
                    future_csv_path = os.path.join(future_dir, "ego_data_future.csv")
                    future_data.to_csv(future_csv_path, index=False)
                    
                    # Copy corresponding LiDAR files
                    for j, lidar_idx in enumerate(range(past_start, past_end)):
                        if lidar_idx < len(self.lidar_files):
                            src_lidar = self.lidar_files[lidar_idx]
                            dst_lidar = os.path.join(lidar_dir, f"lidar_{j:03d}.ply")
                            shutil.copy2(src_lidar, dst_lidar)
                    
                    # Create sequence metadata
                    metadata = {
                        "sequence_id": sequence_id,
                        "past_length": self.sequence_length,
                        "prediction_length": self.prediction_length,
                        "past_timestamps": past_data['timestamp'].tolist(),
                        "future_timestamps": future_data['timestamp'].tolist(),
                        "start_position": {
                            "x": float(past_data.iloc[0]['x']),
                            "y": float(past_data.iloc[0]['y']),
                            "z": float(past_data.iloc[0]['z'])
                        },
                        "end_position": {
                            "x": float(future_data.iloc[-1]['x']),
                            "y": float(future_data.iloc[-1]['y']),
                            "z": float(future_data.iloc[-1]['z'])
                        },
                        "avg_velocity": float(past_data['velocity_magnitude'].mean()),
                        "total_distance": float(np.sqrt(
                            (future_data.iloc[-1]['x'] - past_data.iloc[0]['x'])**2 +
                            (future_data.iloc[-1]['y'] - past_data.iloc[0]['y'])**2
                        ))
                    }
                    
                    # Save metadata
                    metadata_path = os.path.join(sequence_dir, "metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    valid_sequences += 1
                    
                    if valid_sequences % 50 == 0:
                        print(f"Created {valid_sequences} sequences...")
                    
                except Exception as e:
                    print(f"Error creating sequence {i}: {e}")
                    continue
                
                total_sequences += 1
            
            print(f"Successfully created {valid_sequences} test sequences out of {total_sequences} attempts")
            
            # Create summary file
            self.create_summary_file(valid_sequences)
            
            return True
            
        except Exception as e:
            print(f"Error creating test sequences: {e}")
            return False
    
    def create_summary_file(self, num_sequences):
        """Create a summary file with dataset statistics"""
        try:
            summary = {
                "dataset_info": {
                    "source_data_dir": self.data_dir,
                    "output_dir": self.output_dir,
                    "generation_time": datetime.now().isoformat(),
                    "total_sequences": num_sequences
                },
                "sequence_parameters": {
                    "past_length": self.sequence_length,
                    "prediction_length": self.prediction_length,
                    "step_size": self.step_size,
                    "min_velocity_filter": self.min_velocity
                },
                "data_statistics": {
                    "total_ego_records": len(self.ego_data),
                    "total_lidar_files": len(self.lidar_files),
                    "avg_velocity": float(self.ego_data['velocity_magnitude'].mean()),
                    "max_velocity": float(self.ego_data['velocity_magnitude'].max()),
                    "time_span": {
                        "start": float(self.ego_data['timestamp'].min()),
                        "end": float(self.ego_data['timestamp'].max()),
                        "duration": float(self.ego_data['timestamp'].max() - self.ego_data['timestamp'].min())
                    }
                },
                "usage_instructions": {
                    "model_input": "Use 'past/ego_data.csv' and 'lidar/*.ply' files as model input",
                    "ground_truth": "Use 'ground_truth/ego_data_future.csv' for evaluation",
                    "metadata": "Check 'metadata.json' for sequence-specific information"
                }
            }
            
            summary_path = os.path.join(self.output_dir, "dataset_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Created dataset summary: {summary_path}")
            
        except Exception as e:
            print(f"Error creating summary file: {e}")
    
    def run(self):
        """Run the complete test sequence generation process"""
        print("Starting CARLA test sequence generation...")
        
        # Step 1: Find data files
        if not self.find_data_files():
            return False
        
        # Step 2: Load ego data
        if not self.load_ego_data():
            return False
        
        # Step 3: Match LiDAR to ego data
        if not self.match_lidar_to_ego_data():
            return False
        
        # Step 4: Create test sequences
        if not self.create_test_sequences():
            return False
        
        print(f"\nTest sequence generation completed successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"\nEach sequence contains:")
        print(f"  - past/ego_data.csv: {self.sequence_length} past ego vehicle states")
        print(f"  - lidar/*.ply: {self.sequence_length} corresponding LiDAR point clouds")
        print(f"  - ground_truth/ego_data_future.csv: {self.prediction_length} future positions")
        print(f"  - metadata.json: Sequence information and statistics")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Generate test sequences from CARLA data for trajectory prediction')
    parser.add_argument('--data-dir', required=True, help='Directory containing collected CARLA data')
    parser.add_argument('--output-dir', default='./test_sequences', help='Output directory for test sequences')
    parser.add_argument('--sequence-length', default=10, type=int, help='Number of past data points per sequence')
    parser.add_argument('--prediction-length', default=10, type=int, help='Number of future points to predict')
    parser.add_argument('--step-size', default=5, type=int, help='Step size between sequences (overlap control)')
    parser.add_argument('--min-velocity', default=0.5, type=float, help='Minimum velocity to include (m/s)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    if args.sequence_length <= 0 or args.prediction_length <= 0:
        print("Error: Sequence length and prediction length must be positive")
        return
    
    # Create and run generator
    generator = CARLATestSequenceGenerator(args)
    success = generator.run()
    
    if success:
        print("\n✅ Test sequence generation completed successfully!")
    else:
        print("\n❌ Test sequence generation failed!")


if __name__ == '__main__':
    main()