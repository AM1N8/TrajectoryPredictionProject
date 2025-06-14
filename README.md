# Trajectory Prediction using Point Clouds and Past Sensor Data

## Table of Contents
- [Project Overview](#project-overview)
- [Introduction](#introduction)
- [Problem Formulation](#problem-formulation)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation)
  - [Real-Time Simulation](#real-time-simulation)
  - [Streamlit Application](#streamlit-application)
- [Demonstrations and Results](#demonstrations-and-results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project focuses on time series analysis, specifically for autonomous vehicle trajectory prediction and simulation using multimodal sensor fusion. It combines sequential vehicle dynamics data with LiDAR point cloud information to predict future vehicle trajectories. The methodology employs deep learning architectures, drawing inspiration from PointNet for point cloud processing and LSTMs with attention for sequence modeling, specifically designed for spatiotemporal data fusion. This addresses the critical challenge of accurate motion prediction in autonomous driving scenarios, potentially integrating with the CARLA simulator (indicated by carla_data).

## Introduction

This project presents a comprehensive approach to autonomous vehicle trajectory prediction using multimodal sensor fusion. The system combines sequential vehicle dynamics data with LiDAR point cloud information to predict future vehicle trajectories. The methodology employs deep learning architectures specifically designed for spatiotemporal data fusion, addressing the critical challenge of accurate motion prediction in autonomous driving scenarios.

## Problem Formulation

The trajectory prediction problem is formulated as a sequence-to-sequence learning task where we predict future vehicle displacements based on historical vehicle states and environmental perception data.

**Mathematical Formulation:**

Given a sequence of historical vehicle states:
```
X_seq = {x₁, x₂, ..., x_T}
```

And corresponding LiDAR point cloud data:
```
P = {p₁, p₂, ..., p_N} where pᵢ ∈ R⁴ (x,y,z,intensity)
```

The objective is to predict future relative displacements:
```
Y = {Δx₁, Δx₂, ..., Δx_H} where Δxᵢ = (δxᵢ, δyᵢ)
```

Where T is the sequence length, N is the number of LiDAR points, and H is the prediction horizon.

## Features

- **Multimodal Sensor Fusion**: Integrates historical vehicle dynamics (kinematics, control inputs) and static/dynamic environmental context from LiDAR point clouds.

- **Enhanced Data Preprocessing**: Includes robust methods for LiDAR point cloud sampling (weighted by distance), normalization, and comprehensive vehicle dynamics feature calculation (speed, acceleration, jerk, heading, curvature, relative displacements).

- **Sequence Segmentation**: Filters out stationary data and segments continuous driving sequences for more meaningful training.

- **Deep Learning Architecture**: Utilizes a hybrid neural network featuring:
  - A PointNet-inspired encoder for processing irregular LiDAR point cloud data, extracting global features.
  - A Bidirectional LSTM encoder with a self-attention mechanism for capturing temporal dependencies and important historical states in vehicle dynamics.

- **Custom Loss Function**: Employs a weighted_displacement_loss that prioritizes accuracy for earlier predictions within the horizon.

- **Comprehensive Evaluation Metrics**: Beyond standard errors, it evaluates path smoothness and performance at different vehicle speeds.

- **Real-time Simulation**: An interactive matplotlib-based simulation to visualize ego-vehicle movement, historical paths, predicted trajectories, and key performance metrics in real-time.

- **Interactive Streamlit Application**: A user-friendly web interface for demonstrating trajectory predictions.

## Project Structure

```
.
├── carla_data/                  # Storage for CARLA simulation data or related datasets
│   ├── ego_data_*.csv          # Ego vehicle data files
│   └── lidar/                  # LiDAR point cloud files (*.ply)
├── data_collection/             # Scripts and utilities for collecting time series data
├── inference/                   # Code for running predictions with trained models
├── model/                       # Contains trained models and architectures
│   ├── trajectory_prediction_model.h5
│   └── trajectory_scalers.pkl
├── testing_samples/             # Sample datasets for testing
├── visualizations/              # Scripts and notebooks for plots and animations
├── data_EDA.ipynb               # Exploratory Data Analysis notebook
├── pipeline.ipynb               # Data preprocessing and model training pipeline
├── pipeline.md                  # Pipeline documentation
├── pipeline_doc.html            # HTML export of pipeline documentation
├── real_time_simulation.mp4     # Real-time simulation demonstration
├── requirements.txt             # Python dependencies
├── simulation.py                # Interactive real-time simulation script
├── trajectory_comparison_animation.mp4
├── trajectory_prediction_animation.mp4
├── trajectory_streamlit_app.py  # Streamlit web application
└── README.md                    # This README file
```

## Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

- Python 3.8+ (recommended)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AM1N8/TrajectoryPredictionProject.git
   cd TrajectoryPredictionProject
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


4. **Ensure you have the required data and model files:**
   - `carla_data` folder with `ego_data_*.csv` files
   - `lidar` subfolder containing `.ply` files
   - Trained model files (`trajectory_prediction_model.h5`, `trajectory_scalers.pkl`) in the `model` directory

## Usage

This section details how to run different parts of the project, covering data handling, model training, and real-time prediction.

### Data Loading and Preprocessing

The `data_EDA.ipynb` and `pipeline.ipynb` notebooks are central to understanding and preparing the data.

**Key Steps:**

1. **Load Ego Vehicle Data:**
   - The `ego_df` DataFrame is loaded from `carla_data/ego_data_*.csv`
   - Contains timestamped ego vehicle state information (position, velocity, control inputs)

2. **Load LiDAR Point Clouds:**
   - LiDAR data stored as PLY files (`.ply`) in `carla_data/lidar/`
   - Loaded using `load_lidar_point_cloud`
   - `sample_point_cloud` implements weighted sampling strategy
   - `process_point_cloud` normalizes and centers point cloud data

3. **Synchronize Data:**
   - `map_timestamps_to_lidar` finds closest LiDAR scan for each ego vehicle timestamp

4. **Calculate Advanced Vehicle Dynamics:**
   - `calculate_vehicle_dynamics` derives features: speed, heading, acceleration, jerk, curvature
   - `calculate_relative_displacement` computes delta_x and delta_y

5. **Filter and Segment Data:**
   - `filter_and_segment_data` removes stationary periods and segments continuous driving sequences

6. **Prepare Sequences:**
   - `prepare_relative_sequences` constructs input and target sequences
   - Features normalized using `MinMaxScaler` and `StandardScaler`

**To run preprocessing:**
```bash
jupyter notebook pipeline.ipynb
```

### Model Architecture

The trajectory prediction model combines sequential and point cloud data processing:

**Components:**

1. **Point Cloud Encoder (`build_point_cloud_encoder`):**
   - PointNet-inspired architecture for LiDAR processing
   - Uses Conv1D layers for point-wise feature extraction
   - Combines GlobalMaxPooling1D and GlobalAveragePooling1D
   - Includes BatchNormalization and Dropout for regularization

2. **Sequence Encoder (`build_sequence_encoder`):**
   - Bidirectional LSTM layers for temporal dependencies
   - Self-attention mechanism for dynamic time step weighting
   - Processes historical vehicle dynamics sequence

3. **Complete Model (`build_trajectory_prediction_model`):**
   - Fuses outputs from both encoders
   - Dense layers for final displacement predictions

![Trajectory Prediction Model Architecture](./visualizations/pipeline_graph.svg)


### Training and Evaluation

**Training Setup:**
- Custom `weighted_displacement_loss` function
- Adam optimizer with callbacks:
  - `EarlyStopping` (patience=10)
  - `ReduceLROnPlateau`
  - `ModelCheckpoint`

**Evaluation Metrics:**
- Mean Displacement Error
- Final Displacement Error
- Speed-based Performance Analysis
- Path Smoothness Quantification

**Model outputs saved:**
- `trajectory_prediction_model.h5`
- `trajectory_scalers.pkl`
- `training_history.png`
- `trajectory_predictions.png`

![Trajectory Prediction](./visualizations/streamlit.png)

### Real-Time Simulation

Run the interactive simulation:
```bash
python simulation.py
```

**Controls:**
- **SPACE**: Play/Pause simulation
- **UP/DOWN**: Increase/Decrease speed
- **LEFT/RIGHT**: Step backward/forward (when paused)
- **R**: Reset simulation
- **H**: Toggle historical path display
- **L**: Toggle LiDAR visualization
- **Close window**: Exit simulation

**Display Features:**
- Real-time trajectory prediction
- Ground truth comparison
- Performance metrics
- LiDAR point cloud visualization

### Streamlit Application

Launch the web interface:
```bash
streamlit run trajectory_streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

## Demonstrations and Results

The project includes several demonstration files:

- **`real_time_simulation.mp4`**: Real-time simulation demonstration
- **`trajectory_comparison_animation.mp4`**: Predicted vs. ground truth comparison
- **`trajectory_prediction_animation.mp4`**: Model prediction showcase
- **`training_history.png`**: Training progress visualization
- **`trajectory_predictions.png`**: Sample prediction comparisons

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or inquiries, please contact 
* [Gmail](mailto:mohamedaminedarraj@gmail.com)
* [linkedIn](https://www.linkedin.com/in/mohamed-amine-darraj-b4754631a/).

