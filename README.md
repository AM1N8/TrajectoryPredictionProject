Vehicle Trajectory Prediction

## Project Overview

This project implements an advanced trajectory prediction system for autonomous vehicles by combining vehicle telemetry data with LiDAR point cloud information. The system predicts future vehicle paths while accounting for both the vehicle's dynamics and its surrounding environment.

## Project Description

### Core Concept
The project demonstrates how to create a neural network model that can predict a vehicle's future trajectory based on:
1. Historical driving data (position, velocity, steering, etc.)
2. 3D LiDAR point cloud data representing the surrounding environment

### Data Sources
The system uses CARLA simulator data consisting of:
- **Ego vehicle telemetry**: Speed, position, orientation, steering, throttle, and brake
- **LiDAR point clouds**: 3D representations of the vehicle's surroundings

### Key Components

#### 1. Data Processing and Feature Engineering
- **Vehicle Dynamics Calculation**: Transforms raw telemetry into meaningful features (acceleration, jerk, curvature)
- **Relative Displacement Representation**: Focuses on changes in position rather than absolute coordinates
- **LiDAR Processing**: Converts raw point clouds into normalized, consistent representations
- **Sequence Creation**: Segments driving data into meaningful sequences for training

#### 2. Neural Network Architecture
- **Dual-Input Design**: Processes vehicle telemetry and LiDAR data in parallel branches
- **Point Cloud Processing**: Uses a PointNet-inspired architecture to handle 3D point clouds
- **Sequence Processing**: Employs bidirectional LSTMs with attention mechanisms
- **Feature Fusion**: Combines features from both branches for trajectory prediction

#### 3. Trajectory Prediction
- **Multi-Step Prediction**: Forecasts vehicle positions across multiple future timesteps
- **Weighted Loss Function**: Prioritizes accuracy in near-future predictions
- **Steering Calculation**: Converts predicted paths into steering commands using a bicycle model

#### 4. Evaluation System
- **Speed-Stratified Analysis**: Evaluates performance at different vehicle speeds
- **Smoothness Metrics**: Assesses the realism and comfort of predicted trajectories
- **Visualization Tools**: Provides 2D plots and interactive visualizations of predicted vs. actual paths

### Practical Applications
This project demonstrates techniques applicable to:
- Autonomous vehicle planning and control
- Advanced driver assistance systems (ADAS)
- Traffic simulation and modeling
- Robotics motion planning

### Technical Approach
The implementation uses:
- TensorFlow/Keras for deep learning
- Custom architectures for point cloud and sequence processing
- Advanced normalization and sampling techniques
- Comprehensive evaluation metrics

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib
- scikit-learn
- PlyFile (for LiDAR data processing)
- tqdm (for progress tracking)

## Results

The system demonstrates:
- Accurate trajectory prediction across various driving scenarios
- Robust performance at different vehicle speeds
- Smooth trajectory generation suitable for comfortable vehicle control
- Effective integration of environmental context from LiDAR data

The visualizations produced by the system illustrate the comparison between predicted and actual vehicle paths, along with corresponding steering commands.
