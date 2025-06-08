#!/usr/bin/env python

"""
Trajectory Prediction Inference Script
-------------------------------------
Standalone inference script for trajectory prediction model.
Loads a trained model and makes predictions on input sequences.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import glob
import pickle
from plyfile import PlyData
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


class TrajectoryPredictor:
    """Main class for trajectory prediction inference"""
    
    def __init__(self, model_path, scalers_path):
        """
        Initialize the predictor with model and scalers
        
        Args:
            model_path (str): Path to the trained model file (.h5)
            scalers_path (str): Path to the scalers pickle file
        """
        self.model_path = model_path
        self.scalers_path = scalers_path
        self.model = None
        self.input_scaler = None
        self.target_scaler = None
        
        self._load_model_and_scalers()
    
    def _weighted_displacement_loss(self, y_true, y_pred):
        """Custom loss function required for model loading"""
        delta_x_true, delta_y_true = y_true[:, :, 0], y_true[:, :, 1]
        delta_x_pred, delta_y_pred = y_pred[:, :, 0], y_pred[:, :, 1]
        distance_error = tf.sqrt(tf.square(delta_x_true - delta_x_pred) + tf.square(delta_y_true - delta_y_pred) + 1e-6)
        time_steps = tf.shape(delta_x_true)[1]
        time_weights = 1.0 / tf.sqrt(tf.cast(tf.range(1, time_steps + 1), tf.float32))
        time_weights = time_weights / tf.reduce_sum(time_weights)
        weighted_error = distance_error * tf.expand_dims(time_weights, axis=0)
        return tf.reduce_mean(weighted_error)
    
    def _load_model_and_scalers(self):
        """Load the trained model and scalers"""
        print(f"Loading model from: {self.model_path}")
        self.model = keras.models.load_model(
            self.model_path, 
            custom_objects={'weighted_displacement_loss': self._weighted_displacement_loss}
        )
        
        print(f"Loading scalers from: {self.scalers_path}")
        with open(self.scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        self.input_scaler = scalers['input_scaler']
        self.target_scaler = scalers['target_scaler']
        
        print("✅ Model and scalers loaded successfully!")
    
    def load_lidar_point_cloud(self, file_path):
        """
        Load and process LiDAR point cloud from PLY file
        
        Args:
            file_path (str): Path to the PLY file
            
        Returns:
            np.ndarray: Processed point cloud data (1024, 4) - [x, y, z, intensity]
        """
        try:
            plydata = PlyData.read(file_path)
            x, y, z = plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']
            intensity = plydata['vertex']['intensity']
            points = np.column_stack((x, y, z, intensity))
            
            # Sample points to fixed size (1024 points)
            if len(points) >= 1024:
                # Distance-based weighted sampling
                distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
                weights = 1.0 / (distances + 0.1)
                weights /= np.sum(weights)
                indices = np.random.choice(len(points), 1024, replace=False, p=weights)
                points = points[indices]
            else:
                # Upsample if insufficient points
                indices = np.random.choice(len(points), 1024, replace=True)
                points = points[indices]
            
            # Process points - center and normalize
            xyz = points[:, :3]
            if len(xyz) > 0:
                center = np.median(xyz, axis=0)
                xyz = xyz - center
                scale = np.percentile(np.sqrt(np.sum(xyz**2, axis=1)), 90)
                if scale > 0:
                    xyz = xyz / scale
            
            # Normalize intensity
            intensity = points[:, 3].reshape(-1, 1)
            if intensity.size > 0:
                intensity_max = np.max(intensity) if np.max(intensity) > 0 else 1.0
                intensity = intensity / intensity_max
            
            return np.hstack((xyz, intensity))
            
        except Exception as e:
            print(f"Error loading LiDAR data from {file_path}: {e}")
            return np.zeros((1024, 4))
    
    def process_ego_data(self, ego_data_path):
        """
        Process ego vehicle data CSV to extract features
        
        Args:
            ego_data_path (str): Path to the ego_data.csv file
            
        Returns:
            tuple: (processed_features, raw_data)
        """
        if not os.path.exists(ego_data_path):
            raise FileNotFoundError(f"Ego data not found: {ego_data_path}")
        
        ego_data = pd.read_csv(ego_data_path)
        
        # Calculate derived features (same as training)
        ego_data['speed'] = np.sqrt(ego_data['velocity_x']**2 + ego_data['velocity_y']**2)
        ego_data['heading'] = np.arctan2(ego_data['velocity_y'], ego_data['velocity_x'])
        ego_data['acceleration_x'] = ego_data['velocity_x'].diff().fillna(0) / ego_data['timestamp'].diff().fillna(1)
        ego_data['acceleration_y'] = ego_data['velocity_y'].diff().fillna(0) / ego_data['timestamp'].diff().fillna(1)
        ego_data['acceleration'] = np.sqrt(ego_data['acceleration_x']**2 + ego_data['acceleration_y']**2)
        ego_data['heading_change'] = ego_data['heading'].diff().fillna(0) / ego_data['timestamp'].diff().fillna(1)
        ego_data['curvature'] = ego_data['heading_change'] / (ego_data['speed'] + 1e-6)
        ego_data['is_moving'] = (ego_data['speed'] > 0.5).astype(float)
        ego_data['delta_x'] = ego_data['x'].diff().fillna(0)
        ego_data['delta_y'] = ego_data['y'].diff().fillna(0)
        
        # Feature columns (must match training features)
        input_features = [
            'delta_x', 'delta_y', 'velocity_x', 'velocity_y', 'speed',
            'heading', 'heading_change', 'curvature',
            'acceleration_x', 'acceleration_y', 'acceleration',
            'steering', 'throttle', 'brake', 'is_moving'
        ]
        
        # Check for missing features
        missing_features = [f for f in input_features if f not in ego_data.columns]
        if missing_features:
            raise ValueError(f"Missing features in ego data: {missing_features}")
        
        # Extract and normalize features
        feature_data = ego_data[input_features].values
        normalized_features = self.input_scaler.transform(feature_data)
        
        return normalized_features, ego_data
    
    def predict_trajectory(self, ego_data_path, lidar_file_path=None):
        """
        Make trajectory prediction for a single sequence
        
        Args:
            ego_data_path (str): Path to ego_data.csv file
            lidar_file_path (str, optional): Path to LiDAR PLY file
            
        Returns:
            np.ndarray: Predicted relative displacements (n_steps, 2)
        """
        # Process ego data
        input_seq, ego_data = self.process_ego_data(ego_data_path)
        
        # Load LiDAR data
        if lidar_file_path and os.path.exists(lidar_file_path):
            lidar_data = self.load_lidar_point_cloud(lidar_file_path)
        else:
            print("Warning: No LiDAR data provided, using zeros")
            lidar_data = np.zeros((1024, 4))
        
        # Prepare inputs for model (add batch dimension)
        input_batch = input_seq[np.newaxis]  # (1, seq_len, features)
        lidar_batch = lidar_data[np.newaxis]   # (1, 1024, 4)
        
        # Make prediction
        print("Making prediction...")
        prediction = self.model.predict([input_batch, lidar_batch], verbose=0)
        
        # Denormalize prediction
        pred_original = self.target_scaler.inverse_transform(
            prediction.reshape(-1, 2)
        ).reshape(prediction.shape)[0]
        
        return pred_original, ego_data
    
    def convert_relative_to_absolute(self, start_x, start_y, relative_displacements):
        """
        Convert relative displacements to absolute trajectory
        
        Args:
            start_x (float): Starting X position
            start_y (float): Starting Y position  
            relative_displacements (np.ndarray): Relative displacements (n_steps, 2)
            
        Returns:
            np.ndarray: Absolute trajectory positions (n_steps+1, 2)
        """
        positions = np.zeros((len(relative_displacements) + 1, 2))
        positions[0] = [start_x, start_y]
        
        for i, (dx, dy) in enumerate(relative_displacements):
            positions[i+1] = positions[i] + [dx, dy]
        
        return positions
    
    def predict_sequence_directory(self, sequence_dir):
        """
        Process a complete sequence directory and make prediction
        
        Args:
            sequence_dir (str): Path to sequence directory containing past/, lidar/, etc.
            
        Returns:
            dict: Results containing prediction, ground truth, and metadata
        """
        # Find required files
        past_ego = os.path.join(sequence_dir, 'past', 'ego_data.csv')
        lidar_dir = os.path.join(sequence_dir, 'lidar')
        gt_future = os.path.join(sequence_dir, 'ground_truth', 'ego_data_future.csv')
        
        # Get latest LiDAR file
        lidar_file = None
        if os.path.exists(lidar_dir):
            lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.ply')))
            if lidar_files:
                lidar_file = lidar_files[-1]  # Use latest frame
        
        # Make prediction
        pred_relative, past_data = self.predict_trajectory(past_ego, lidar_file)
        
        # Convert to absolute coordinates
        start_x, start_y = past_data.iloc[-1]['x'], past_data.iloc[-1]['y']
        pred_absolute = self.convert_relative_to_absolute(start_x, start_y, pred_relative)
        
        # Load ground truth if available
        gt_absolute = None
        final_error = None
        if os.path.exists(gt_future):
            future_data = pd.read_csv(gt_future)
            gt_relative = np.column_stack([
                future_data['x'].diff().fillna(0), 
                future_data['y'].diff().fillna(0)
            ])
            gt_absolute = self.convert_relative_to_absolute(start_x, start_y, gt_relative)
            
            # Calculate final position error
            final_error = np.linalg.norm(pred_absolute[-1] - gt_absolute[-1])
        
        return {
            'predicted_trajectory': pred_absolute,
            'ground_truth_trajectory': gt_absolute,
            'relative_displacements': pred_relative,
            'past_data': past_data,
            'final_error': final_error,
            'start_position': (start_x, start_y)
        }
    
    def plot_trajectory(self, results, save_path=None, show_plot=True):
        """
        Plot trajectory prediction results
        
        Args:
            results (dict): Results from predict_sequence_directory
            save_path (str, optional): Path to save the plot
            show_plot (bool): Whether to display the plot
        """
        plt.figure(figsize=(10, 8))
        
        pred_traj = results['predicted_trajectory']
        gt_traj = results['ground_truth_trajectory']
        
        # Plot trajectories
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, 
                label='Predicted', marker='o', markersize=4)
        
        if gt_traj is not None:
            plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', linewidth=2, 
                    label='Ground Truth', marker='s', markersize=4)
        
        # Mark start and end points
        plt.plot(pred_traj[0, 0], pred_traj[0, 1], 'go', markersize=10, label='Start')
        plt.plot(pred_traj[-1, 0], pred_traj[-1, 1], 'ro', markersize=10, label='Pred End')
        
        if gt_traj is not None:
            plt.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'bo', markersize=10, label='GT End')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Trajectory Prediction Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if results['final_error'] is not None:
            plt.text(0.02, 0.98, f'Final Error: {results["final_error"]:.3f}m', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Trajectory Prediction Inference')
    parser.add_argument('--model', type=str, default='../model/trajectory_prediction_model.h5',
                       help='Path to the trained model file (.h5)')
    parser.add_argument('--scalers', type=str, default='../model/trajectory_scalers.pkl',
                       help='Path to the scalers pickle file')
    parser.add_argument('--sequence', type=str, default='../testing_samples/testing_sequences/sequence_000195',
                       help='Path to test sequence directory')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                       help='Show trajectory plot')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = TrajectoryPredictor(args.model, args.scalers)
    
    # Make prediction
    print(f"Processing sequence: {args.sequence}")
    results = predictor.predict_sequence_directory(args.sequence)
    
    # Print results
    print("\n=== Prediction Results ===")
    print(f"Start Position: ({results['start_position'][0]:.2f}, {results['start_position'][1]:.2f})")
    print(f"Predicted End Position: ({results['predicted_trajectory'][-1, 0]:.2f}, {results['predicted_trajectory'][-1, 1]:.2f})")
    
    if results['final_error'] is not None:
        print(f"Final Position Error: {results['final_error']:.3f} m")
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trajectory data
        np.savetxt(output_dir / 'predicted_trajectory.csv', 
                  results['predicted_trajectory'], delimiter=',', 
                  header='x,y', comments='')
        
        if results['ground_truth_trajectory'] is not None:
            np.savetxt(output_dir / 'ground_truth_trajectory.csv', 
                      results['ground_truth_trajectory'], delimiter=',',
                      header='x,y', comments='')
        
        # Save plot
        if args.plot:
            predictor.plot_trajectory(results, 
                                    save_path=output_dir / 'trajectory_plot.png',
                                    show_plot=True)
        
        print(f"Results saved to: {args.output}")
    
    elif args.plot:
        predictor.plot_trajectory(results, show_plot=True)


# Example usage
if __name__ == "__main__":
    # Command line usage with updated paths for inference/ folder
    main()
    
    # Alternative: Direct usage example with updated paths
    """
    # When running from inference/ folder, use relative paths to parent directory
    
    # Initialize predictor
    predictor = TrajectoryPredictor(
        model_path="../model/trajectory_prediction_model.h5",
        scalers_path="../model/trajectory_scalers.pkl"
    )
    
    # Method 1: Predict from sequence directory
    results = predictor.predict_sequence_directory("../testing_samples/testing_sequences/sequence_001")
    predictor.plot_trajectory(results)
    
    # Method 2: Predict from individual files
    pred, ego_data = predictor.predict_trajectory(
        ego_data_path="../data/ego_data.csv",
        lidar_file_path="../data/lidar_frame.ply"
    )
    
    # Convert to absolute trajectory
    start_x, start_y = ego_data.iloc[-1]['x'], ego_data.iloc[-1]['y']
    absolute_trajectory = predictor.convert_relative_to_absolute(start_x, start_y, pred)
    
    # Expected project structure:
    # project_root/
    # ├── model/
    # │   ├── trajectory_prediction_model.h5
    # │   └── trajectory_scalers.pkl
    # ├── testing_samples/
    # │   └── testing_sequences/
    # │       ├── sequence_001/
    # │       ├── sequence_002/
    # │       └── ...
    # ├── inference/
    # │   ├── trajectory_inference.py  (this file)
    # │   └── results/  (created automatically)
    # └── data/  (optional, for individual files)
    """