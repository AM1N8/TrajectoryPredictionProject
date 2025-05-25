#!/usr/bin/env python

"""
Trajectory Prediction Model Tester
---------------------------------
Tests pre-trained trajectory prediction model on test sequences and visualizes results.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import glob
import json
import pickle
from plyfile import PlyData
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import argparse


def load_lidar_point_cloud(file_path):
    """Load and process LiDAR point cloud"""
    try:
        plydata = PlyData.read(file_path)
        x, y, z = plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']
        intensity = plydata['vertex']['intensity']
        points = np.column_stack((x, y, z, intensity))
        
        # Sample and process points (same as training)
        if len(points) >= 1024:
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            weights = 1.0 / (distances + 0.1)
            weights /= np.sum(weights)
            indices = np.random.choice(len(points), 1024, replace=False, p=weights)
            points = points[indices]
        else:
            indices = np.random.choice(len(points), 1024, replace=True)
            points = points[indices]
        
        # Process points (center and normalize)
        xyz = points[:, :3]
        if len(xyz) > 0:
            center = np.median(xyz, axis=0)
            xyz = xyz - center
            scale = np.percentile(np.sqrt(np.sum(xyz**2, axis=1)), 90)
            if scale > 0:
                xyz = xyz / scale
        
        intensity = points[:, 3].reshape(-1, 1)
        if intensity.size > 0:
            intensity_max = np.max(intensity) if np.max(intensity) > 0 else 1.0
            intensity = intensity / intensity_max
        
        return np.hstack((xyz, intensity))
    except:
        return np.zeros((1024, 4))


def process_sequence(sequence_dir, input_scaler):
    """Process a single test sequence"""
    # Load past ego data
    past_csv = os.path.join(sequence_dir, 'past', 'ego_data.csv')
    ego_data = pd.read_csv(past_csv)
    
    # Calculate dynamics features (same as training)
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
    
    # Input features (must match training)
    input_features = [
        'delta_x', 'delta_y', 'velocity_x', 'velocity_y', 'speed',
        'heading', 'heading_change', 'curvature',
        'acceleration_x', 'acceleration_y', 'acceleration',
        'steering', 'throttle', 'brake', 'is_moving'
    ]
    
    # Normalize input
    input_seq = input_scaler.transform(ego_data[input_features].values)
    
    # Load LiDAR data (use last frame)
    lidar_dir = os.path.join(sequence_dir, 'lidar')
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.ply')))
    if lidar_files:
        lidar_data = load_lidar_point_cloud(lidar_files[-1])
    else:
        lidar_data = np.zeros((1024, 4))
    
    # Load ground truth
    future_csv = os.path.join(sequence_dir, 'ground_truth', 'ego_data_future.csv')
    future_data = pd.read_csv(future_csv)
    
    return input_seq, lidar_data, ego_data, future_data


def weighted_displacement_loss(y_true, y_pred):
    """Custom loss function (needed for model loading)"""
    delta_x_true, delta_y_true = y_true[:, :, 0], y_true[:, :, 1]
    delta_x_pred, delta_y_pred = y_pred[:, :, 0], y_pred[:, :, 1]
    distance_error = tf.sqrt(tf.square(delta_x_true - delta_x_pred) + tf.square(delta_y_true - delta_y_pred) + 1e-6)
    time_steps = tf.shape(delta_x_true)[1]
    time_weights = 1.0 / tf.sqrt(tf.cast(tf.range(1, time_steps + 1), tf.float32))
    time_weights = time_weights / tf.reduce_sum(time_weights)
    weighted_error = distance_error * tf.expand_dims(time_weights, axis=0)
    return tf.reduce_mean(weighted_error)


def convert_relative_to_absolute(start_x, start_y, relative_displacements):
    """Convert relative displacements to absolute positions"""
    positions = np.zeros((len(relative_displacements) + 1, 2))
    positions[0] = [start_x, start_y]
    for i, (dx, dy) in enumerate(relative_displacements):
        positions[i+1] = positions[i] + [dx, dy]
    return positions


def test_model(model_path, scalers_path, test_sequences_dir, max_sequences=20):
    """Test model on sequences and visualize results"""
    
    # Load model and scalers
    print("Loading model and scalers...")
    model = keras.models.load_model(model_path, custom_objects={'weighted_displacement_loss': weighted_displacement_loss})
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
        input_scaler = scalers['input_scaler']
        target_scaler = scalers['target_scaler']
    
    # Find test sequences
    sequence_dirs = sorted(glob.glob(os.path.join(test_sequences_dir, 'sequence_*')))[:max_sequences]
    print(f"Found {len(sequence_dirs)} test sequences")
    
    # Process sequences
    predictions = []
    ground_truths = []
    errors = []
    
    for i, seq_dir in enumerate(sequence_dirs):
        try:
            print(f"Processing sequence {i+1}/{len(sequence_dirs)}")
            
            # Process sequence
            input_seq, lidar_data, past_data, future_data = process_sequence(seq_dir, input_scaler)
            
            # Predict
            pred = model.predict([input_seq[np.newaxis], lidar_data[np.newaxis]], verbose=0)
            pred_original = target_scaler.inverse_transform(pred.reshape(-1, 2)).reshape(pred.shape)[0]
            
            # Ground truth relative displacements
            gt_relative = np.column_stack([future_data['x'].diff().fillna(0), future_data['y'].diff().fillna(0)])
            
            # Convert to absolute paths
            start_x, start_y = past_data.iloc[-1]['x'], past_data.iloc[-1]['y']
            pred_path = convert_relative_to_absolute(start_x, start_y, pred_original)
            gt_path = convert_relative_to_absolute(start_x, start_y, gt_relative)
            
            predictions.append(pred_path)
            ground_truths.append(gt_path)
            
            # Calculate error
            final_error = np.linalg.norm(pred_path[-1] - gt_path[-1])
            errors.append(final_error)
            
        except Exception as e:
            print(f"Error processing sequence {seq_dir}: {e}")
            continue
    
    # Visualize results
    print(f"\nResults Summary:")
    print(f"Mean final position error: {np.mean(errors):.3f}m")
    print(f"Max final position error: {np.max(errors):.3f}m")
    print(f"Std final position error: {np.std(errors):.3f}m")
    
    # Plot trajectories
    n_plots = min(8, len(predictions))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i in range(n_plots):
        ax = axes[i]
        gt = ground_truths[i]
        pred = predictions[i]
        
        ax.plot(gt[:, 0], gt[:, 1], 'b-', linewidth=2, label='Ground Truth')
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='Predicted')
        ax.scatter(gt[0, 0], gt[0, 1], c='g', s=100, marker='o', label='Start')
        ax.scatter(gt[-1, 0], gt[-1, 1], c='b', s=100, marker='s', label='GT End')
        ax.scatter(pred[-1, 0], pred[-1, 1], c='r', s=100, marker='^', label='Pred End')
        
        ax.set_title(f'Seq {i+1} (Error: {errors[i]:.2f}m)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True)
        ax.axis('equal')
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('trajectory_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Error distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Final Position Error (m)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(errors, 'o-', alpha=0.7)
    plt.xlabel('Sequence Index')
    plt.ylabel('Final Position Error (m)')
    plt.title('Error by Sequence')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test trajectory prediction model')
    parser.add_argument('--model', default='trajectory_prediction_model.h5', help='Path to saved model')
    parser.add_argument('--scalers', default='trajectory_scalers.pkl', help='Path to saved scalers')
    parser.add_argument('--test-dir', default='./testing_samples/testing_sequences', help='Directory with test sequences')
    parser.add_argument('--max-sequences', default=20, type=int, help='Maximum sequences to test')
    
    args = parser.parse_args()
    
    if not all(os.path.exists(p) for p in [args.model, args.scalers, args.test_dir]):
        print("Error: Model, scalers, or test directory not found!")
        return
    
    test_model(args.model, args.scalers, args.test_dir, args.max_sequences)
    print("\nTesting completed! Check 'trajectory_test_results.png' and 'error_analysis.png'")


if __name__ == '__main__':
    main()