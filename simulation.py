import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Arrow
from matplotlib.collections import LineCollection
import glob
from plyfile import PlyData
from tqdm import tqdm
import math
import time
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import deque

class TrajectorySimulator:
    def __init__(self, model_path, scalers_path, ego_df_filtered, timestamp_to_lidar, 
                 seq_length=10, prediction_horizon=10):
        """
        Initialize the trajectory prediction simulator
        
        Args:
            model_path: Path to the trained model
            scalers_path: Path to the saved scalers
            ego_df_filtered: Processed ego vehicle data
            timestamp_to_lidar: Mapping of timestamps to LiDAR files
            seq_length: Length of input sequence
            prediction_horizon: Number of timesteps to predict
        """
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.ego_df = ego_df_filtered.copy()
        self.timestamp_to_lidar = timestamp_to_lidar
        
        # Load model and scalers
        self.load_model_and_scalers(model_path, scalers_path)
        
        # Input features (must match training)
        self.input_features = [
            'delta_x', 'delta_y',
            'velocity_x', 'velocity_y', 'speed',
            'heading', 'heading_change', 'curvature',
            'acceleration_x', 'acceleration_y', 'acceleration',
            'steering', 'throttle', 'brake',
            'is_moving'
        ]
        
        # Simulation state
        self.current_index = 0
        self.is_playing = False
        self.speed_multiplier = 1.0
        self.show_lidar = False
        self.show_historical_path = True
        self.historical_path_length = 50
        
        # Visualization elements
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.setup_plot()
        
        # Data storage for visualization
        self.historical_positions = deque(maxlen=self.historical_path_length)
        self.current_prediction = None
        self.prediction_paths = deque(maxlen=5)  # Store last 5 predictions
        
        # Performance tracking
        self.prediction_errors = []
        self.computation_times = []
        
    def load_model_and_scalers(self, model_path, scalers_path):
        """Load the trained model and scalers"""
        # Define custom loss function for loading
        def weighted_displacement_loss(y_true, y_pred):
            delta_x_true = y_true[:, :, 0]
            delta_y_true = y_true[:, :, 1]
            delta_x_pred = y_pred[:, :, 0]
            delta_y_pred = y_pred[:, :, 1]
            
            distance_error = tf.sqrt(tf.square(delta_x_true - delta_x_pred) + 
                                   tf.square(delta_y_true - delta_y_pred) + 1e-6)
            
            time_steps = tf.shape(delta_x_true)[1]
            time_weights = 1.0 / tf.sqrt(tf.cast(tf.range(1, time_steps + 1), tf.float32))
            time_weights = time_weights / tf.reduce_sum(time_weights)
            
            weighted_error = distance_error * tf.expand_dims(time_weights, axis=0)
            return tf.reduce_mean(weighted_error)
        
        self.model = keras.models.load_model(model_path, 
                                           custom_objects={'weighted_displacement_loss': weighted_displacement_loss})
        
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_input = scalers['input_scaler']
            self.scaler_target = scalers['target_scaler']
    
    def setup_plot(self):
        """Setup the matplotlib plot for simulation"""
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Real-Time Trajectory Prediction Simulation', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        
        # Plot the full trajectory as background "road"
        self.ax.plot(self.ego_df['x'], self.ego_df['y'], 'lightgray', 
                    linewidth=3, alpha=0.5, label='Full Route', zorder=1)
        
        # Initialize dynamic elements
        self.ego_vehicle = Circle((0, 0), 1.5, color='blue', zorder=5)
        self.ax.add_patch(self.ego_vehicle)
        
        # Historical path line
        self.historical_line, = self.ax.plot([], [], 'darkblue', linewidth=2, 
                                           alpha=0.7, label='Historical Path', zorder=3)
        
        # Current prediction line
        self.prediction_line, = self.ax.plot([], [], 'red', linewidth=3, 
                                           linestyle='--', label='Current Prediction', zorder=4)
        
        # Previous predictions (fading)
        self.prev_predictions = []
        for i in range(5):
            alpha = 0.3 - i * 0.05
            line, = self.ax.plot([], [], 'red', linewidth=1, alpha=alpha, zorder=2)
            self.prev_predictions.append(line)
        
        # Direction arrow
        self.direction_arrow = None
        
        # Status text
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Legend
        self.ax.legend(loc='upper right')
        
        # Set initial view
        self.update_viewport()
    
    def load_lidar_point_cloud(self, file_path):
        """Load LiDAR point cloud from PLY file"""
        try:
            plydata = PlyData.read(file_path)
            x = plydata['vertex']['x']
            y = plydata['vertex']['y'] 
            z = plydata['vertex']['z']
            intensity = plydata['vertex']['intensity']
            points = np.column_stack((x, y, z, intensity))
            return points
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros((100, 4))
    
    def sample_point_cloud(self, points, n_points=1024):
        """Sample a fixed number of points from point cloud"""
        if len(points) == 0:
            return np.zeros((n_points, points.shape[1]))
        
        if len(points) >= n_points:
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            weights = 1.0 / (distances + 0.1)
            weights /= np.sum(weights)
            indices = np.random.choice(len(points), n_points, replace=False, p=weights)
            return points[indices]
        else:
            indices = np.random.choice(len(points), n_points, replace=True)
            return points[indices]
    
    def process_point_cloud(self, points, max_points=1024):
        """Process point cloud data with normalization"""
        xyz = points[:, :3]
        
        if len(xyz) > 0:
            center = np.median(xyz, axis=0)
            xyz = xyz - center
            
            radii = np.sqrt(np.sum(xyz**2, axis=1))
            if len(radii) > 0:
                scale = np.percentile(radii, 90)
                if scale > 0:
                    xyz = xyz / scale
        
        if points.shape[1] > 3:
            intensity = points[:, 3].reshape(-1, 1)
            if intensity.size > 0:
                intensity_max = np.max(intensity) if np.max(intensity) > 0 else 1.0
                intensity = intensity / intensity_max
            processed_points = np.hstack((xyz, intensity))
        else:
            processed_points = xyz
        
        return processed_points
    
    def convert_relative_to_absolute(self, start_x, start_y, relative_displacements):
        """Convert relative displacements to absolute positions"""
        absolute_positions = np.zeros((len(relative_displacements) + 1, 2))
        absolute_positions[0] = [start_x, start_y]
        
        for i in range(len(relative_displacements)):
            absolute_positions[i+1, 0] = absolute_positions[i, 0] + relative_displacements[i, 0]
            absolute_positions[i+1, 1] = absolute_positions[i, 1] + relative_displacements[i, 1]
        
        return absolute_positions
    
    def predict_trajectory(self, current_index):
        """Predict trajectory at current timestep"""
        if current_index < self.seq_length:
            return None, None  # Return tuple even when None
        
        start_time = time.time()
        
        # Get sequence indices
        seq_indices = list(range(current_index - self.seq_length, current_index))
        
        # Extract input sequence
        input_seq = self.ego_df.iloc[seq_indices][self.input_features].values
        input_seq_norm = self.scaler_input.transform(input_seq)
        
        # Get LiDAR data
        current_ts = self.ego_df.iloc[current_index - 1]['timestamp']
        if current_ts in self.timestamp_to_lidar:
            lidar_file = self.timestamp_to_lidar[current_ts]
            pointcloud = self.load_lidar_point_cloud(lidar_file)
            sampled_points = self.sample_point_cloud(pointcloud)
            processed_points = self.process_point_cloud(sampled_points)
        else:
            processed_points = np.zeros((1024, 4))
        
        # Expand dimensions for batch processing
        sequence_input = np.expand_dims(input_seq_norm, axis=0)
        lidar_input = np.expand_dims(processed_points, axis=0)
        
        # Get model prediction
        prediction = self.model.predict([sequence_input, lidar_input], verbose=0)
        
        # Reshape and inverse transform
        pred_reshaped = prediction.reshape(-1, 2)
        pred_original = self.scaler_target.inverse_transform(pred_reshaped)
        pred_original = pred_original.reshape(prediction.shape)[0]
        
        # Get current position
        start_x = self.ego_df.iloc[current_index - 1]['x']
        start_y = self.ego_df.iloc[current_index - 1]['y']
        
        # Convert to absolute positions
        predicted_path = self.convert_relative_to_absolute(start_x, start_y, pred_original)
        
        # Track computation time
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        return predicted_path, processed_points if self.show_lidar else None
    
    def calculate_prediction_error(self, predicted_path, current_index):
        """Calculate prediction error against ground truth"""
        if predicted_path is None or current_index >= len(self.ego_df) - 1:
            return None
        
        # Get actual next position
        actual_x = self.ego_df.iloc[current_index]['x']
        actual_y = self.ego_df.iloc[current_index]['y']
        
        # Calculate error for first predicted position
        pred_x = predicted_path[1, 0]  # Skip start position
        pred_y = predicted_path[1, 1]
        
        error = np.sqrt((actual_x - pred_x)**2 + (actual_y - pred_y)**2)
        self.prediction_errors.append(error)
        return error
    
    def update_viewport(self):
        """Update the viewport to follow the ego vehicle"""
        if self.current_index >= len(self.ego_df):
            return
        
        current_x = self.ego_df.iloc[self.current_index]['x']
        current_y = self.ego_df.iloc[self.current_index]['y']
        
        # Set viewport size based on speed (faster = wider view)
        current_speed = self.ego_df.iloc[self.current_index]['speed']
        viewport_size = max(50, current_speed * 3)  # Minimum 50m, scale with speed
        
        self.ax.set_xlim(current_x - viewport_size, current_x + viewport_size)
        self.ax.set_ylim(current_y - viewport_size, current_y + viewport_size)
    
    def update_visualization(self):
        """Update all visualization elements"""
        if self.current_index >= len(self.ego_df):
            self.is_playing = False
            return
    
        current_row = self.ego_df.iloc[self.current_index]
        current_x, current_y = current_row['x'], current_row['y']
    
        # Update ego vehicle position
        self.ego_vehicle.center = (current_x, current_y)
    
        # Update historical path
        self.historical_positions.append((current_x, current_y))
        if len(self.historical_positions) > 1:
            hist_x, hist_y = zip(*self.historical_positions)
            self.historical_line.set_data(hist_x, hist_y)
    
        # Update direction arrow - safer removal
        if self.direction_arrow:
            try:
                if self.direction_arrow in self.ax.patches:
                    self.direction_arrow.remove()
            except (ValueError, AttributeError):
                # If removal fails, just continue
                pass
            finally:
                self.direction_arrow = None
    
        if self.current_index > 0:
            prev_row = self.ego_df.iloc[self.current_index - 1]
            dx = current_x - prev_row['x']
            dy = current_y - prev_row['y']
            if abs(dx) > 0.1 or abs(dy) > 0.1:  # Only show arrow if moving
                try:
                    self.direction_arrow = Arrow(current_x, current_y, dx*5, dy*5, 
                                               width=2, color='blue', alpha=0.7)
                    self.ax.add_patch(self.direction_arrow)
                except Exception:
                    # If arrow creation fails, continue without it
                    self.direction_arrow = None
    
        # Get trajectory prediction
        predicted_path, lidar_data = self.predict_trajectory(self.current_index)
    
        # Update current prediction
        if predicted_path is not None:
            self.prediction_line.set_data(predicted_path[:, 0], predicted_path[:, 1])
        
            # Add to previous predictions (for fading effect)
            self.prediction_paths.append(predicted_path)
        
            # Update previous prediction lines (fading effect)
            for i, prev_line in enumerate(self.prev_predictions):
                if i < len(self.prediction_paths) - 1:
                    path_idx = len(self.prediction_paths) - 2 - i
                    if path_idx >= 0:
                        path = self.prediction_paths[path_idx]
                        prev_line.set_data(path[:, 0], path[:, 1])
                    else:
                        prev_line.set_data([], [])
                else:
                    prev_line.set_data([], [])
        
            # Calculate prediction error
            error = self.calculate_prediction_error(predicted_path, self.current_index)
        else:
            self.prediction_line.set_data([], [])
            error = None
    
        # Update status text
        status_info = [
            f"Time: {current_row['timestamp']:.2f}s",
            f"Position: ({current_x:.1f}, {current_y:.1f})",
            f"Speed: {current_row['speed']:.1f} m/s",
            f"Heading: {np.degrees(current_row['heading']):.1f}°",
            f"Frame: {self.current_index}/{len(self.ego_df)-1}",
            f"Speed: {self.speed_multiplier:.1f}x"
        ]

        if error is not None:
            status_info.append(f"Pred Error: {error:.2f}m")

        if self.prediction_errors:
            avg_error = np.mean(self.prediction_errors[-50:])  # Last 50 predictions
            status_info.append(f"Avg Error: {avg_error:.2f}m")

        if self.computation_times:
            avg_time = np.mean(self.computation_times[-50:]) * 1000  # Convert to ms
            status_info.append(f"Compute: {avg_time:.1f}ms")
    
        self.status_text.set_text('\n'.join(status_info))
    
        # Update viewport
        self.update_viewport()
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == ' ':  # Space bar - play/pause
            self.is_playing = not self.is_playing
            print(f"Simulation {'resumed' if self.is_playing else 'paused'}")
        
        elif event.key == 'up':  # Increase speed
            self.speed_multiplier = min(10.0, self.speed_multiplier * 1.5)
            print(f"Speed: {self.speed_multiplier:.1f}x")
        
        elif event.key == 'down':  # Decrease speed
            self.speed_multiplier = max(0.1, self.speed_multiplier / 1.5)
            print(f"Speed: {self.speed_multiplier:.1f}x")
        
        elif event.key == 'right':  # Step forward
            if not self.is_playing:
                self.current_index = min(len(self.ego_df) - 1, self.current_index + 1)
                self.update_visualization()
                plt.draw()
        
        elif event.key == 'left':  # Step backward
            if not self.is_playing:
                self.current_index = max(0, self.current_index - 1)
                self.update_visualization()
                plt.draw()
        
        elif event.key == 'r':  # Reset
            self.current_index = 0
            self.historical_positions.clear()
            self.prediction_paths.clear()
            self.prediction_errors.clear()
            self.computation_times.clear()
            print("Simulation reset")
        
        elif event.key == 'h':  # Toggle historical path
            self.show_historical_path = not self.show_historical_path
            if not self.show_historical_path:
                self.historical_line.set_data([], [])
            print(f"Historical path: {'ON' if self.show_historical_path else 'OFF'}")
        
        elif event.key == 'l':  # Toggle LiDAR visualization
            self.show_lidar = not self.show_lidar
            print(f"LiDAR visualization: {'ON' if self.show_lidar else 'OFF'}")
    
    def animate(self, frame):
        """Animation function for matplotlib"""
        if self.is_playing and self.current_index < len(self.ego_df) - 1:
            self.current_index += 1
            self.update_visualization()
        
        return [self.ego_vehicle, self.historical_line, self.prediction_line] + self.prev_predictions
    
    def run_simulation(self):
        """Run the interactive simulation"""
        print("\n" + "="*60)
        print("TRAJECTORY PREDICTION SIMULATION")
        print("="*60)
        print("Controls:")
        print("  SPACE    - Play/Pause")
        print("  UP/DOWN  - Increase/Decrease speed")
        print("  LEFT/RIGHT - Step backward/forward (when paused)")
        print("  R        - Reset simulation")
        print("  H        - Toggle historical path")
        print("  L        - Toggle LiDAR visualization")
        print("  Close window to exit")
        print("="*60)
        
        # Connect keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Set up animation
        # Calculate interval based on data frequency and speed multiplier
        if len(self.ego_df) > 1:
            avg_dt = (self.ego_df['timestamp'].iloc[-1] - self.ego_df['timestamp'].iloc[0]) / len(self.ego_df)
            base_interval = max(50, avg_dt * 1000)  # Convert to ms, minimum 50ms
        else:
            base_interval = 100
        
        def interval_func():
            return max(50, base_interval / self.speed_multiplier)
        
        # Create animation
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, interval=100, blit=False, repeat=True
        )
        
        # Start simulation
        self.is_playing = True
        plt.tight_layout()
        plt.show()
        
        # Print final statistics
        if self.prediction_errors:
            print(f"\nFinal Statistics:")
            print(f"Total predictions: {len(self.prediction_errors)}")
            print(f"Average prediction error: {np.mean(self.prediction_errors):.3f}m")
            print(f"Max prediction error: {np.max(self.prediction_errors):.3f}m")
            print(f"Min prediction error: {np.min(self.prediction_errors):.3f}m")
        
        if self.computation_times:
            print(f"Average computation time: {np.mean(self.computation_times)*1000:.1f}ms")


# Example usage and helper functions
def load_lidar_point_cloud(file_path):
    """Load LiDAR point cloud from PLY file"""
    try:
        plydata = PlyData.read(file_path)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        intensity = plydata['vertex']['intensity']
        points = np.column_stack((x, y, z, intensity))
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros((100, 4))

def calculate_vehicle_dynamics(ego_df):
    """Calculate enhanced vehicle dynamics features"""
    df = ego_df.copy()
    
    df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    df['heading'] = np.arctan2(df['velocity_y'], df['velocity_x'])
    
    df['acceleration_x'] = df['velocity_x'].diff() / df['timestamp'].diff()
    df['acceleration_y'] = df['velocity_y'].diff() / df['timestamp'].diff()
    df['acceleration'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2)
    
    df['jerk_x'] = df['acceleration_x'].diff() / df['timestamp'].diff()
    df['jerk_y'] = df['acceleration_y'].diff() / df['timestamp'].diff()
    df['jerk'] = np.sqrt(df['jerk_x']**2 + df['jerk_y']**2)
    
    df['heading_change'] = df['heading'].diff() / df['timestamp'].diff()
    df['curvature'] = df['heading_change'] / (df['speed'] + 1e-6)
    df['is_moving'] = (df['speed'] > 0.5).astype(float)
    
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].fillna(0)
    
    return df

def calculate_relative_displacement(ego_df):
    """Calculate relative displacements instead of absolute positions"""
    df = ego_df.copy()
    df['delta_x'] = df['x'].diff().fillna(0)
    df['delta_y'] = df['y'].diff().fillna(0)
    return df

def filter_and_segment_data(ego_df, min_speed=0.5, min_segment_length=5):
    """Filter stationary data and segment into meaningful driving sequences"""
    df = ego_df.copy()
    
    df['is_moving'] = (df['speed'] > min_speed).astype(int)
    df['movement_change'] = df['is_moving'].diff().abs()
    df.loc[0, 'movement_change'] = 0
    
    df['segment_id'] = df['movement_change'].cumsum()
    
    segment_lengths = df.groupby('segment_id').size()
    valid_segments = segment_lengths[segment_lengths >= min_segment_length].index
    
    df_filtered = df[df['segment_id'].isin(valid_segments)].copy()
    
    segment_mapping = {old_id: new_id for new_id, old_id in enumerate(df_filtered['segment_id'].unique())}
    df_filtered['segment_id'] = df_filtered['segment_id'].map(segment_mapping)
    
    df_filtered['segment_start'] = df_filtered['segment_id'].diff().ne(0).astype(int)
    df_filtered.loc[df_filtered.index[0], 'segment_start'] = 1
    
    return df_filtered

def map_timestamps_to_lidar(ego_df, lidar_path):
    """Map timestamps from ego data to LiDAR files"""
    lidar_files = glob.glob(os.path.join(lidar_path, 'lidar_*.ply'))
    lidar_timestamps = [float(os.path.splitext(os.path.basename(f))[0].split('_')[1]) for f in lidar_files]
    
    lidar_timestamps = np.array(lidar_timestamps)
    sorted_indices = np.argsort(lidar_timestamps)
    lidar_timestamps = lidar_timestamps[sorted_indices]
    lidar_files = [lidar_files[i] for i in sorted_indices]
    
    timestamp_to_lidar = {}
    
    for idx, ts in enumerate(ego_df['timestamp']):
        if len(lidar_timestamps) > 0:
            closest_idx = np.searchsorted(lidar_timestamps, ts)
            
            if closest_idx == 0:
                closest_match = closest_idx
            elif closest_idx == len(lidar_timestamps):
                closest_match = closest_idx - 1
            else:
                if abs(lidar_timestamps[closest_idx] - ts) < abs(lidar_timestamps[closest_idx-1] - ts):
                    closest_match = closest_idx
                else:
                    closest_match = closest_idx - 1
                
            timestamp_to_lidar[ts] = lidar_files[closest_match]
    
    return timestamp_to_lidar

# Example usage
if __name__ == "__main__":
    # Example data paths - adjust these to match your setup
    data_path = './carla_data'
    ego_data_file = glob.glob(os.path.join(data_path, 'ego_data_*.csv'))[0]
    lidar_path = os.path.join(data_path, 'lidar')
    model_path = './model/trajectory_prediction_model.h5'
    scalers_path = './model/trajectory_scalers.pkl'
    
    print("Loading and processing data...")
    
    # Load and process data
    ego_df = pd.read_csv(ego_data_file)
    ego_df_dynamics = calculate_vehicle_dynamics(ego_df)
    ego_df_relative = calculate_relative_displacement(ego_df_dynamics)
    ego_df_filtered = filter_and_segment_data(ego_df_relative)
    timestamp_to_lidar = map_timestamps_to_lidar(ego_df_filtered, lidar_path)
    
    print(f"Processed {len(ego_df_filtered)} timesteps")
    
    # Create and run simulation
    simulator = TrajectorySimulator(
        model_path=model_path,
        scalers_path=scalers_path,
        ego_df_filtered=ego_df_filtered,
        timestamp_to_lidar=timestamp_to_lidar,
        seq_length=10,
        prediction_horizon=10
    )
    
    simulator.run_simulation()