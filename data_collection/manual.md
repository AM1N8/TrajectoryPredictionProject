# CARLA Data Collection Script for Time Series Vehicle Trajectory Prediction

## Overview

This Python script provides a comprehensive data collection framework for CARLA simulator, specifically designed for time series vehicle trajectory prediction research. The script collects multi-modal data including vehicle dynamics, environmental context, and sensor information that are essential for training trajectory prediction models.

## Features

- **Ego Vehicle Data Collection**: Position, velocity, acceleration, control inputs, and orientation
- **Environmental Context**: Lane geometry, road topology, and traffic information  
- **LiDAR Point Cloud**: 360-degree environmental perception data
- **Traffic Simulation**: Configurable traffic density and behavior
- **Flexible Output**: Time-synchronized CSV files and PLY point clouds

## Requirements

### Software Dependencies
- CARLA Simulator 0.9.14+
- Python 3.7+
- Required Python packages:
  ```bash
  pip install numpy carla
  ```

### Hardware Recommendations
- GPU with 4GB+ VRAM for CARLA rendering
- 8GB+ RAM for data processing
- SSD storage for high-frequency data collection

## Installation

1. **Install CARLA Simulator**
   ```bash
   # Download CARLA 0.9.14 from official repository
   wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.14.tar.gz
   tar -xvf CARLA_0.9.14.tar.gz
   ```

2. **Start CARLA Server**
   ```bash
   cd CARLA_0.9.14
   ./CarlaUE4.sh -quality-level=Low -world-port=2000
   ```

3. **Run Data Collection Script**
   ```bash
   python carla_data_collector.py [OPTIONS]
   ```

## Command Line Arguments

### Basic Configuration
- `--host` (default: localhost): CARLA server hostname
- `--port` (default: 2000): CARLA server port
- `--output-dir` (default: ./carla_data): Output directory for collected data

### Simulation Parameters
- `--sync`: Enable synchronous mode for deterministic simulation
- `--fps` (default: 10): Simulation frequency in frames per second
- `--max-frames` (default: 0): Maximum frames to collect (0 = unlimited)
- `--seed` (default: 42): Random seed for reproducible experiments

### Traffic and Environment
- `--num-vehicles` (default: 0): Number of AI-controlled vehicles to spawn
- `--keep-traffic-lights`: Keep traffic lights active (disabled by default)
- `--tm-port` (default: 8000): Traffic Manager port

### Sensor Configuration
- `--radius` (default: 100.0): LiDAR detection range in meters

## Data Output Structure

The script generates time-synchronized datasets in the following structure:

```
carla_data/
├── ego_data_YYYYMMDD_HHMMSS.csv     # Vehicle trajectory data
├── lane_data_YYYYMMDD_HHMMSS.csv    # Road geometry data
└── lidar/                           # LiDAR point clouds
    ├── lidar_0.000000.ply
    ├── lidar_0.100000.ply
    └── ...
```

### Ego Vehicle Data (`ego_data_*.csv`)

| Column | Description | Unit | Usage for Trajectory Prediction |
|--------|-------------|------|--------------------------------|
| `timestamp` | Simulation time | seconds | Temporal indexing |
| `x`, `y`, `z` | Vehicle position | meters | Target trajectory coordinates |
| `steering` | Steering input | [-1, 1] | Control input features |
| `throttle` | Throttle input | [0, 1] | Control input features |
| `brake` | Brake input | [0, 1] | Control input features |
| `yaw` | Vehicle yaw angle | degrees | Orientation state |
| `heading` | Heading direction | degrees | Motion direction |
| `velocity_x`, `velocity_y`, `velocity_z` | Velocity components | m/s | Motion state features |
| `accel_x`, `accel_y`, `accel_z` | Acceleration components | m/s² | Dynamic features |

### Lane Data (`lane_data_*.csv`)

| Column | Description | Usage for Trajectory Prediction |
|--------|-------------|--------------------------------|
| `timestamp` | Simulation time | Temporal alignment with ego data |
| `lane_id` | Unique lane identifier | Spatial context encoding |
| `lane_width` | Width of the lane | Constraint boundaries |
| `waypoint_x`, `waypoint_y`, `waypoint_z` | Lane centerline points | Reference path features |
| `is_junction` | Junction indicator | Behavioral context |
| `lane_type` | Lane type (driving, shoulder, etc.) | Environmental constraints |
| `is_drivable` | Drivability flag | Navigation constraints |

### LiDAR Data (`lidar/*.ply`)

Point cloud files containing:
- 3D coordinates (x, y, z) of detected obstacles
- Intensity values for material classification
- Environmental perception for obstacle avoidance

## Usage Examples for Trajectory Prediction

### 1. Basic Data Collection

```bash
# Collect basic trajectory data with minimal traffic
python carla_data_collector.py \
    --sync \
    --fps 20 \
    --max-frames 10000 \
    --output-dir ./trajectory_dataset
```

### 2. Dense Traffic Scenario

```bash
# Collect data with heavy traffic for interaction modeling
python carla_data_collector.py \
    --sync \
    --fps 10 \
    --num-vehicles 50 \
    --keep-traffic-lights \
    --max-frames 20000 \
    --output-dir ./dense_traffic_dataset
```

### 3. Highway Scenario

```bash
# High-speed trajectory collection with extended perception range
python carla_data_collector.py \
    --sync \
    --fps 15 \
    --radius 150.0 \
    --num-vehicles 30 \
    --max-frames 15000 \
    --output-dir ./highway_dataset
```

### 4. Urban Intersection Dataset

```bash
# Complex intersection scenarios with traffic lights
python carla_data_collector.py \
    --sync \
    --fps 10 \
    --num-vehicles 25 \
    --keep-traffic-lights \
    --max-frames 30000 \
    --seed 123 \
    --output-dir ./intersection_dataset
```

## Recommended Parameters for Different Trajectory Prediction Tasks

### Short-term Prediction (1-3 seconds)
```bash
--fps 20 --max-frames 12000  # 10 minutes at 20 Hz
--radius 50.0                # Local environment
--num-vehicles 15            # Moderate traffic
```

### Medium-term Prediction (3-8 seconds)
```bash
--fps 10 --max-frames 18000  # 30 minutes at 10 Hz
--radius 100.0               # Extended perception
--num-vehicles 25            # Dense traffic
```

### Long-term Prediction (8+ seconds)
```bash
--fps 5 --max-frames 15000   # 50 minutes at 5 Hz
--radius 150.0               # Wide-area context
--num-vehicles 40            # Complex interactions
```

## Data Processing for Machine Learning

### Time Series Preparation

1. **Temporal Alignment**: Synchronize ego data with lane data using timestamps
2. **Sequence Generation**: Create sliding windows for input-output pairs
3. **Feature Engineering**: Compute relative positions, velocities, and accelerations
4. **Normalization**: Scale coordinates to local reference frames

### Recommended Sequence Lengths

| Prediction Horizon | Input Sequence | Output Sequence | Sampling Rate |
|--------------------|----------------|-----------------|---------------|
| 1-3 seconds | 20-30 frames | 10-60 frames | 10-20 Hz |
| 3-8 seconds | 30-50 frames | 30-160 frames | 5-10 Hz |
| 8+ seconds | 50-100 frames | 80-400 frames | 2-5 Hz |

### Feature Engineering Examples

```python
import pandas as pd
import numpy as np

# Load collected data
ego_data = pd.read_csv('ego_data_timestamp.csv')
lane_data = pd.read_csv('lane_data_timestamp.csv')

# Compute relative features
ego_data['speed'] = np.sqrt(ego_data['velocity_x']**2 + ego_data['velocity_y']**2)
ego_data['acceleration'] = np.sqrt(ego_data['accel_x']**2 + ego_data['accel_y']**2)

# Create trajectory sequences
def create_sequences(data, seq_len, pred_len):
    sequences = []
    for i in range(len(data) - seq_len - pred_len + 1):
        input_seq = data[i:i+seq_len]
        target_seq = data[i+seq_len:i+seq_len+pred_len]
        sequences.append((input_seq, target_seq))
    return sequences
```

## Performance Optimization

### High-Frequency Collection
- Use `--sync` mode for deterministic timing
- Increase `--fps` up to 30 for fine-grained temporal resolution
- Monitor disk I/O for bottlenecks

### Large-Scale Datasets
- Use multiple CARLA instances with different ports
- Implement data compression for storage efficiency
- Consider distributed collection across multiple machines

### Quality Assurance
- Validate timestamp consistency across data streams
- Check for missing frames or data gaps
- Verify spatial coordinate consistency

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Ensure CARLA server is running
   - Check port availability
   - Verify firewall settings

2. **Performance Issues**
   - Reduce rendering quality in CARLA
   - Lower simulation frequency
   - Close unnecessary applications

3. **Data Quality Problems**
   - Enable synchronous mode for timing consistency
   - Increase LiDAR range for better environment coverage
   - Add more traffic vehicles for realistic interactions

### Memory Management

- LiDAR data can be memory-intensive
- Consider periodic data flushing for long collections
- Monitor available disk space during collection

## Integration with ML Frameworks

### PyTorch Dataset Example

```python
import torch
from torch.utils.data import Dataset

class CARLATrajectoryDataset(Dataset):
    def __init__(self, ego_file, lane_file, seq_len=30, pred_len=10):
        self.ego_data = pd.read_csv(ego_file)
        self.lane_data = pd.read_csv(lane_file)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.ego_data) - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, idx):
        # Extract input sequence
        input_seq = self.ego_data.iloc[idx:idx+self.seq_len]
        target_seq = self.ego_data.iloc[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        
        # Convert to tensors
        features = torch.tensor(input_seq[['x', 'y', 'velocity_x', 'velocity_y']].values)
        targets = torch.tensor(target_seq[['x', 'y']].values)
        
        return features, targets
```

## Citation

If you use this data collection framework in your research, please cite:

```bibtex
@misc{carla_trajectory_collector,
    title={CARLA Data Collection Framework for Vehicle Trajectory Prediction},
    author={Mohamed Amine Darraj},
    year={2025},
    howpublished={https://github.com/AM1N8/TrajectoryPredictionProject}
}
```

## License

This script is provided under the MIT License. CARLA simulator has its own licensing terms which should be reviewed separately.