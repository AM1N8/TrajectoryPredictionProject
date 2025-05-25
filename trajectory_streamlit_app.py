#!/usr/bin/env python

"""
Trajectory Prediction Streamlit App
----------------------------------
Interactive web application for testing trajectory prediction model on sequences.
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import glob
import pickle
from plyfile import PlyData
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64


@st.cache_resource
def load_model_and_scalers(model_path, scalers_path):
    """Load model and scalers with caching"""
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
    
    model = keras.models.load_model(model_path, custom_objects={'weighted_displacement_loss': weighted_displacement_loss})
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    return model, scalers['input_scaler'], scalers['target_scaler']


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
    except Exception as e:
        st.error(f"Error loading LiDAR data: {e}")
        return np.zeros((1024, 4))


def process_sequence(sequence_dir, input_scaler):
    """Process a single test sequence"""
    # Load past ego data
    past_csv = os.path.join(sequence_dir, 'past', 'ego_data.csv')
    if not os.path.exists(past_csv):
        raise FileNotFoundError(f"Past ego data not found: {past_csv}")
    
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
    
    # Check if all features exist
    missing_features = [f for f in input_features if f not in ego_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in ego data: {missing_features}")
    
    # Normalize input
    input_seq = input_scaler.transform(ego_data[input_features].values)
    
    # Load LiDAR data (use last frame)
    lidar_dir = os.path.join(sequence_dir, 'lidar')
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.ply')))
    if lidar_files:
        lidar_data = load_lidar_point_cloud(lidar_files[-1])
    else:
        st.warning("No LiDAR files found, using zeros")
        lidar_data = np.zeros((1024, 4))
    
    # Load ground truth
    future_csv = os.path.join(sequence_dir, 'ground_truth', 'ego_data_future.csv')
    if not os.path.exists(future_csv):
        raise FileNotFoundError(f"Future ego data not found: {future_csv}")
    
    future_data = pd.read_csv(future_csv)
    
    return input_seq, lidar_data, ego_data, future_data


def convert_relative_to_absolute(start_x, start_y, relative_displacements):
    """Convert relative displacements to absolute positions"""
    positions = np.zeros((len(relative_displacements) + 1, 2))
    positions[0] = [start_x, start_y]
    for i, (dx, dy) in enumerate(relative_displacements):
        positions[i+1] = positions[i] + [dx, dy]
    return positions


def create_trajectory_plot(pred_path, gt_path, sequence_name):
    """Create interactive trajectory plot using Plotly"""
    fig = go.Figure()
    
    # Ground truth trajectory
    fig.add_trace(go.Scatter(
        x=gt_path[:, 0], y=gt_path[:, 1],
        mode='lines+markers',
        name='Ground Truth',
        line=dict(color='blue', width=3),
        marker=dict(size=4)
    ))
    
    # Predicted trajectory
    fig.add_trace(go.Scatter(
        x=pred_path[:, 0], y=pred_path[:, 1],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=4)
    ))
    
    # Start point
    fig.add_trace(go.Scatter(
        x=[gt_path[0, 0]], y=[gt_path[0, 1]],
        mode='markers',
        name='Start',
        marker=dict(color='green', size=12, symbol='circle')
    ))
    
    # End points
    fig.add_trace(go.Scatter(
        x=[gt_path[-1, 0]], y=[gt_path[-1, 1]],
        mode='markers',
        name='GT End',
        marker=dict(color='blue', size=12, symbol='square')
    ))
    
    fig.add_trace(go.Scatter(
        x=[pred_path[-1, 0]], y=[pred_path[-1, 1]],
        mode='markers',
        name='Pred End',
        marker=dict(color='red', size=12, symbol='triangle-up')
    ))
    
    # Calculate error
    final_error = np.linalg.norm(pred_path[-1] - gt_path[-1])
    
    fig.update_layout(
        title=f'{sequence_name} - Final Position Error: {final_error:.3f}m',
        xaxis_title='X Position (m)',
        yaxis_title='Y Position (m)',
        hovermode='closest',
        showlegend=True,
        width=800,
        height=600
    )
    
    # Equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig, final_error


def create_lidar_visualization(lidar_data):
    """Create 3D LiDAR point cloud visualization"""
    fig = go.Figure(data=[go.Scatter3d(
        x=lidar_data[:, 0],
        y=lidar_data[:, 1],
        z=lidar_data[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=lidar_data[:, 3],  # intensity
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Intensity")
        )
    )])
    
    fig.update_layout(
        title='LiDAR Point Cloud',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='cube'
        ),
        width=800,
        height=600
    )
    
    return fig


def create_metrics_plot(past_data, future_data):
    """Create time series plots of vehicle metrics"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Speed', 'Steering', 'Throttle & Brake', 'Acceleration'),
        vertical_spacing=0.12
    )
    
    # Combine past and future data
    all_times = list(past_data['timestamp']) + list(future_data['timestamp'])
    all_speeds = list(past_data['speed']) + [np.sqrt(future_data.iloc[i]['velocity_x']**2 + future_data.iloc[i]['velocity_y']**2) 
                                           for i in range(len(future_data))]
    
    # Speed
    fig.add_trace(go.Scatter(x=past_data['timestamp'], y=past_data['speed'], 
                            name='Past Speed', line=dict(color='blue')), row=1, col=1)
    
    # Steering
    fig.add_trace(go.Scatter(x=past_data['timestamp'], y=past_data['steering'], 
                            name='Past Steering', line=dict(color='green')), row=1, col=2)
    
    # Throttle and Brake
    fig.add_trace(go.Scatter(x=past_data['timestamp'], y=past_data['throttle'], 
                            name='Throttle', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=past_data['timestamp'], y=past_data['brake'], 
                            name='Brake', line=dict(color='orange')), row=2, col=1)
    
    # Acceleration
    fig.add_trace(go.Scatter(x=past_data['timestamp'], y=past_data['acceleration'], 
                            name='Acceleration', line=dict(color='purple')), row=2, col=2)
    
    fig.update_layout(height=500, showlegend=True, title_text="Vehicle Dynamics")
    
    return fig


def main():
    st.set_page_config(page_title="Trajectory Prediction Tester", layout="wide")
    
    st.title("🚗 Trajectory Prediction Model Tester")
    st.markdown("Interactive testing interface for trajectory prediction using time series data and LiDAR point clouds")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model and data paths
    model_path = st.sidebar.text_input("Model Path", value="./model/trajectory_prediction_model.h5")
    scalers_path = st.sidebar.text_input("Scalers Path", value="./model/trajectory_scalers.pkl")
    test_dir = st.sidebar.text_input("Test Sequences Directory", value="./testing_samples/testing_sequences")
    
    # Check if files exist
    files_exist = all(os.path.exists(p) for p in [model_path, scalers_path, test_dir])
    
    if not files_exist:
        st.error("❌ Model, scalers, or test directory not found! Please check the paths.")
        st.stop()
    
    # Load model and scalers
    try:
        with st.spinner("Loading model and scalers..."):
            model, input_scaler, target_scaler = load_model_and_scalers(model_path, scalers_path)
        st.sidebar.success("✅ Model and scalers loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model or scalers: {e}")
        st.stop()
    
    # Find available sequences
    sequence_dirs = sorted(glob.glob(os.path.join(test_dir, 'sequence_*')))
    
    if not sequence_dirs:
        st.error("❌ No test sequences found in the specified directory!")
        st.stop()
    
    st.sidebar.write(f"Found {len(sequence_dirs)} test sequences")
    
    # Sequence selection
    sequence_names = [os.path.basename(seq_dir) for seq_dir in sequence_dirs]
    selected_sequence = st.sidebar.selectbox("Select Test Sequence", sequence_names)
    
    if selected_sequence:
        sequence_path = os.path.join(test_dir, selected_sequence)
        
        # Process sequence button
        if st.sidebar.button("🔍 Analyze Sequence", type="primary"):
            try:
                with st.spinner(f"Processing {selected_sequence}..."):
                    # Process the sequence
                    input_seq, lidar_data, past_data, future_data = process_sequence(sequence_path, input_scaler)
                    
                    # Make prediction
                    pred = model.predict([input_seq[np.newaxis], lidar_data[np.newaxis]], verbose=0)
                    pred_original = target_scaler.inverse_transform(pred.reshape(-1, 2)).reshape(pred.shape)[0]
                    
                    # Ground truth relative displacements
                    gt_relative = np.column_stack([future_data['x'].diff().fillna(0), future_data['y'].diff().fillna(0)])
                    
                    # Convert to absolute paths
                    start_x, start_y = past_data.iloc[-1]['x'], past_data.iloc[-1]['y']
                    pred_path = convert_relative_to_absolute(start_x, start_y, pred_original)
                    gt_path = convert_relative_to_absolute(start_x, start_y, gt_relative)
                    
                    # Store results in session state
                    st.session_state.results = {
                        'pred_path': pred_path,
                        'gt_path': gt_path,
                        'lidar_data': lidar_data,
                        'past_data': past_data,
                        'future_data': future_data,
                        'sequence_name': selected_sequence
                    }
                
                st.success(f"✅ Analysis completed for {selected_sequence}!")
                
            except Exception as e:
                st.error(f"❌ Error processing sequence: {e}")
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Main trajectory plot
        st.header("🎯 Trajectory Prediction Results")
        
        trajectory_fig, final_error = create_trajectory_plot(
            results['pred_path'], 
            results['gt_path'], 
            results['sequence_name']
        )
        st.plotly_chart(trajectory_fig, use_container_width=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Position Error", f"{final_error:.3f} m")
        with col2:
            path_length = np.sum(np.linalg.norm(np.diff(results['gt_path'], axis=0), axis=1))
            st.metric("Path Length", f"{path_length:.2f} m")
        with col3:
            avg_speed = np.mean(results['past_data']['speed'])
            st.metric("Avg Speed", f"{avg_speed:.2f} m/s")
        with col4:
            sequence_duration = results['past_data']['timestamp'].iloc[-1] - results['past_data']['timestamp'].iloc[0]
            st.metric("Sequence Duration", f"{sequence_duration:.1f} s")
        
        # Additional visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Vehicle Dynamics", "🌟 LiDAR Point Cloud", "📋 Data Summary"])
        
        with tab1:
            metrics_fig = create_metrics_plot(results['past_data'], results['future_data'])
            st.plotly_chart(metrics_fig, use_container_width=True)
        
        with tab2:
            lidar_fig = create_lidar_visualization(results['lidar_data'])
            st.plotly_chart(lidar_fig, use_container_width=True)
            
            st.write(f"**LiDAR Points**: {len(results['lidar_data'])} points")
            st.write(f"**Intensity Range**: {results['lidar_data'][:, 3].min():.3f} - {results['lidar_data'][:, 3].max():.3f}")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Past Data Summary")
                st.write(f"**Time Steps**: {len(results['past_data'])}")
                st.write(f"**Max Speed**: {results['past_data']['speed'].max():.2f} m/s")
                st.write(f"**Max Steering**: {results['past_data']['steering'].max():.3f}")
                st.write(f"**Distance Traveled**: {np.sum(np.sqrt(results['past_data']['delta_x']**2 + results['past_data']['delta_y']**2)):.2f} m")
            
            with col2:
                st.subheader("Future Data Summary")
                st.write(f"**Prediction Steps**: {len(results['future_data'])}")
                future_speeds = [np.sqrt(results['future_data'].iloc[i]['velocity_x']**2 + results['future_data'].iloc[i]['velocity_y']**2) 
                               for i in range(len(results['future_data']))]
                st.write(f"**Max Future Speed**: {max(future_speeds):.2f} m/s")
                future_distance = np.sum(np.linalg.norm(np.diff(results['gt_path'], axis=0), axis=1))
                st.write(f"**Future Path Length**: {future_distance:.2f} m")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Configure Paths**: Set the paths to your model file (.h5), scalers file (.pkl), and test sequences directory
        2. **Select Sequence**: Choose a test sequence from the dropdown menu
        3. **Analyze**: Click "Analyze Sequence" to run the prediction
        4. **Explore Results**: 
           - View the trajectory comparison plot
           - Check prediction metrics
           - Explore vehicle dynamics, LiDAR data, and detailed summaries in the tabs
        
        **Requirements**:
        - Model file trained with the trajectory prediction pipeline
        - Scalers file containing input/target scalers
        - Test sequences with the expected directory structure:
          ```
          sequence_X/
          ├── past/ego_data.csv
          ├── lidar/*.ply
          └── ground_truth/ego_data_future.csv
          ```
        """)


if __name__ == "__main__":
    main()