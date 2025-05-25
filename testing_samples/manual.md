# CARLA Test Sequence Generator - Manual

## Overview

The CARLA Test Sequence Generator is a Python script designed to process collected CARLA simulation data and organize it into structured test sequences for trajectory prediction models. The script takes raw CARLA data (ego vehicle CSV files and LiDAR point clouds) and creates organized folders containing sequential data chunks suitable for machine learning model training and evaluation.

## Table of Contents

1. [Purpose](#purpose)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Input Data Format](#input-data-format)
5. [Usage](#usage)
6. [Command Line Arguments](#command-line-arguments)
7. [Output Structure](#output-structure)
8. [Data Processing Pipeline](#data-processing-pipeline)
9. [Configuration Options](#configuration-options)
10. [Troubleshooting](#troubleshooting)
11. [Examples](#examples)
12. [Best Practices](#best-practices)
13. [API Reference](#api-reference)

## Purpose

This tool is specifically designed for researchers and developers working on autonomous vehicle trajectory prediction models. It bridges the gap between raw CARLA simulation data and structured datasets ready for machine learning workflows.

**Primary Use Cases:**
- Creating test datasets for trajectory prediction models
- Organizing time-series data for sequence-to-sequence learning
- Preparing data for model evaluation and benchmarking
- Generating consistent train/validation/test splits

## Requirements

### System Requirements
- Python 3.7 or higher
- Minimum 4GB RAM (8GB+ recommended for large datasets)
- Sufficient disk space (approximately 2x the size of input data)

### Python Dependencies
```bash
pandas>=1.3.0
numpy>=1.20.0
```

### Optional Dependencies
```bash
matplotlib>=3.3.0  # For data visualization
seaborn>=0.11.0    # For enhanced plotting
```

## Installation

1. **Clone or download the script:**
   ```bash
   wget https://your-repository/carla_test_generator.py
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas numpy
   ```

3. **Make the script executable (Linux/macOS):**
   ```bash
   chmod +x carla_test_generator.py
   ```

## Input Data Format

The script expects data collected using the CARLA Data Collection Script with the following structure:

### Required Input Structure
```
carla_data/
├── ego_data_YYYYMMDD_HHMMSS.csv    # Ego vehicle data
├── lane_data_YYYYMMDD_HHMMSS.csv   # Lane data (optional)
└── lidar/                          # LiDAR directory
    ├── lidar_123.456789.ply
    ├── lidar_123.556789.ply
    └── ...
```

### Ego Data CSV Format
The ego data CSV must contain the following columns:
- `timestamp`: Simulation timestamp (float)
- `x`, `y`, `z`: Vehicle position coordinates
- `steering`: Steering input (-1 to 1)
- `throttle`: Throttle input (0 to 1)
- `brake`: Brake input (0 to 1)
- `yaw`: Vehicle yaw angle (degrees)
- `heading`: Vehicle heading (degrees)
- `velocity_x`, `velocity_y`, `velocity_z`: Velocity components
- `accel_x`, `accel_y`, `accel_z`: Acceleration components

### LiDAR Data Format
- **File Format**: PLY (Polygon File Format)
- **Naming Convention**: `lidar_<timestamp>.ply`
- **Content**: Point cloud data with X, Y, Z coordinates and intensity values

## Usage

### Basic Usage
```bash
python carla_test_generator.py --data-dir /path/to/carla_data
```

### Advanced Usage
```bash
python carla_test_generator.py \
    --data-dir ./carla_data \
    --output-dir ./test_sequences \
    --sequence-length 15 \
    --prediction-length 8 \
    --step-size 3 \
    --min-velocity 2.0
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | string | **Required** | Path to directory containing CARLA data |
| `--output-dir` | string | `./test_sequences` | Output directory for generated sequences |
| `--sequence-length` | int | `10` | Number of past data points per sequence |
| `--prediction-length` | int | `10` | Number of future points to predict |
| `--step-size` | int | `5` | Step size between sequences (controls overlap) |
| `--min-velocity` | float | `0.5` | Minimum velocity threshold (m/s) |

### Argument Details

#### `--data-dir` (Required)
- Specifies the input directory containing CARLA data
- Must contain at least one `ego_data_*.csv` file and a `lidar/` subdirectory
- Example: `--data-dir /home/user/carla_experiments/session_001`

#### `--output-dir`
- Destination directory for generated test sequences
- Will be created if it doesn't exist
- Example: `--output-dir ./my_test_data`

#### `--sequence-length`
- Number of historical data points to include in each sequence
- These serve as input to your prediction model
- Range: 1-1000 (practical range: 5-50)
- Example: `--sequence-length 20` for 20 past states

#### `--prediction-length`
- Number of future data points to predict
- These serve as ground truth for model evaluation
- Range: 1-100 (practical range: 5-30)
- Example: `--prediction-length 15` to predict 15 future positions

#### `--step-size`
- Controls overlap between consecutive sequences
- Smaller values create more sequences with more overlap
- Larger values create fewer sequences with less overlap
- Example: `--step-size 1` for maximum overlap, `--step-size 10` for minimal overlap

#### `--min-velocity`
- Filters out stationary or slow-moving periods
- Helps focus on dynamic driving scenarios
- Set to `0.0` to include all data
- Example: `--min-velocity 1.0` to exclude data below 1 m/s

## Output Structure

The script generates a hierarchical directory structure optimized for machine learning workflows:

```
test_sequences/
├── dataset_summary.json           # Overall dataset statistics
├── sequence_000000/              # First test sequence
│   ├── past/
│   │   └── ego_data.csv          # Past ego vehicle states (input)
│   ├── lidar/
│   │   ├── lidar_000.ply        # Corresponding LiDAR frames
│   │   ├── lidar_001.ply
│   │   └── ... (sequence_length files)
│   ├── ground_truth/
│   │   └── ego_data_future.csv   # Future states (ground truth)
│   └── metadata.json            # Sequence-specific information
├── sequence_000005/              # Second test sequence (step_size=5)
│   └── ... (same structure)
└── ...
```

### File Descriptions

#### `dataset_summary.json`
Contains overall dataset statistics and metadata:
```json
{
  "dataset_info": {
    "source_data_dir": "/path/to/carla_data",
    "output_dir": "/path/to/test_sequences",
    "generation_time": "2024-01-15T10:30:45",
    "total_sequences": 245
  },
  "sequence_parameters": {
    "past_length": 10,
    "prediction_length": 10,
    "step_size": 5,
    "min_velocity_filter": 0.5
  },
  "data_statistics": {
    "total_ego_records": 2500,
    "total_lidar_files": 2500,
    "avg_velocity": 8.7,
    "max_velocity": 25.3,
    "time_span": {
      "start": 123.456,
      "end": 373.456,
      "duration": 250.0
    }
  }
}
```

#### `sequence_XXXXXX/metadata.json`
Contains sequence-specific information:
```json
{
  "sequence_id": "sequence_000000",
  "past_length": 10,
  "prediction_length": 10,
  "past_timestamps": [123.456, 123.556, ...],
  "future_timestamps": [124.556, 124.656, ...],
  "start_position": {"x": 100.5, "y": 200.3, "z": 0.5},
  "end_position": {"x": 150.2, "y": 180.7, "z": 0.5},
  "avg_velocity": 12.4,
  "total_distance": 65.8
}
```

#### `past/ego_data.csv`
Input data for your prediction model, containing the same columns as the original ego data but limited to the sequence length.

#### `ground_truth/ego_data_future.csv`
Ground truth data for model evaluation, containing future vehicle states.

#### `lidar/lidar_XXX.ply`
Numbered LiDAR point cloud files corresponding to each past timestamp.

## Data Processing Pipeline

The script follows a systematic data processing pipeline:

### 1. Data Discovery Phase
- Scans input directory for ego data CSV files
- Locates LiDAR directory and enumerates PLY files
- Validates data format and completeness

### 2. Data Loading Phase
- Loads ego vehicle CSV data into memory
- Calculates derived metrics (velocity magnitude)
- Applies velocity filtering if specified

### 3. Timestamp Synchronization
- Extracts timestamps from LiDAR filenames
- Matches LiDAR data to closest ego data timestamps
- Filters out unmatched or poorly synchronized data

### 4. Sequence Generation
- Creates sliding windows of specified length
- Generates overlapping sequences based on step size
- Validates sequence completeness

### 5. File Organization
- Creates directory structure for each sequence
- Copies and organizes relevant data files
- Generates metadata and summary files

## Configuration Options

### Sequence Length Optimization

**Short Sequences (5-10 points):**
- Faster processing and training
- Less memory usage
- May miss long-term patterns
- Good for real-time applications

**Medium Sequences (10-20 points):**
- Balanced approach (recommended)
- Captures most driving patterns
- Reasonable computational requirements

**Long Sequences (20+ points):**
- Captures complex maneuvers
- Higher memory and processing requirements
- May include irrelevant historical data

### Step Size Strategy

**Small Step Size (1-3):**
- Maximum data utilization
- High sequence overlap
- Larger datasets
- Risk of overfitting due to similarity

**Medium Step Size (3-7):**
- Good balance (recommended)
- Moderate overlap
- Diverse sequences

**Large Step Size (10+):**
- Independent sequences
- Smaller datasets
- May miss important transitions

### Velocity Filtering

**Conservative Filtering (0.1-0.5 m/s):**
- Includes most driving scenarios
- May include parking/stationary periods

**Standard Filtering (0.5-2.0 m/s):**
- Focuses on active driving
- Excludes most stationary periods
- Recommended for most applications

**Aggressive Filtering (2.0+ m/s):**
- Highway and dynamic scenarios only
- Excludes urban low-speed driving

## Troubleshooting

### Common Issues

#### "No ego data CSV files found"
**Cause:** Incorrect data directory or missing files
**Solution:** 
- Verify the data directory path
- Ensure ego_data_*.csv files exist
- Check file permissions

#### "LiDAR directory not found"
**Cause:** Missing lidar subdirectory
**Solution:**
- Verify lidar/ directory exists in data directory
- Check directory name spelling (case-sensitive)

#### "Not enough data points"
**Cause:** Insufficient data for requested sequence parameters
**Solution:**
- Reduce sequence_length or prediction_length
- Check if data collection was complete
- Verify min_velocity isn't too restrictive

#### "Error matching LiDAR to ego data"
**Cause:** Timestamp synchronization issues
**Solution:**
- Check LiDAR filename format
- Verify timestamp consistency
- Reduce timestamp tolerance in code if needed

#### Memory errors
**Cause:** Large datasets exceeding available RAM
**Solution:**
- Process data in smaller batches
- Increase system memory
- Use step_size to reduce sequence count

### Performance Optimization

#### For Large Datasets
1. Use larger step sizes to reduce sequence count
2. Apply stricter velocity filtering
3. Process data in chunks
4. Use SSD storage for faster I/O

#### For Memory-Constrained Systems
1. Reduce sequence length
2. Process sequences individually
3. Use data streaming instead of loading all data

## Examples

### Example 1: Basic Test Set Generation
```bash
# Generate standard test sequences
python carla_test_generator.py --data-dir ./carla_data
```
- Creates sequences with 10 past + 10 future points
- 5-point step size (50% overlap)
- Filters velocities below 0.5 m/s

### Example 2: High-Density Sequence Generation
```bash
# Maximum overlap for comprehensive coverage
python carla_test_generator.py \
    --data-dir ./carla_data \
    --step-size 1 \
    --min-velocity 0.1
```
- Creates maximum number of sequences
- Minimal filtering
- Suitable for data augmentation

### Example 3: Highway Scenario Focus
```bash
# Focus on high-speed scenarios
python carla_test_generator.py \
    --data-dir ./highway_data \
    --min-velocity 15.0 \
    --sequence-length 15 \
    --prediction-length 20
```
- Filters for highway speeds (>15 m/s ≈ 54 km/h)
- Longer sequences for complex maneuvers
- Extended prediction horizon

### Example 4: Real-Time Model Testing
```bash
# Optimize for real-time applications
python carla_test_generator.py \
    --data-dir ./carla_data \
    --sequence-length 5 \
    --prediction-length 5 \
    --step-size 10
```
- Short sequences for fast inference
- Non-overlapping sequences
- Suitable for embedded systems

## Best Practices

### Data Quality
1. **Validate Input Data:** Always check data completeness before processing
2. **Timestamp Consistency:** Ensure LiDAR and ego data timestamps align
3. **Velocity Filtering:** Use appropriate thresholds for the intended scenario
4. **Data Balance:** Consider temporal and spatial diversity in sequences

### Model Development
1. **Train/Validation Split:** Use different CARLA sessions for training and testing
2. **Sequence Independence:** Avoid overlap between train and test sequences
3. **Scenario Diversity:** Include various weather, lighting, and traffic conditions
4. **Ground Truth Quality:** Verify future trajectory accuracy

### Performance
1. **Batch Processing:** Process multiple datasets together for efficiency
2. **Parallel Processing:** Consider parallelizing sequence generation for large datasets
3. **Storage Optimization:** Use appropriate file formats and compression
4. **Memory Management:** Monitor memory usage with large datasets

### Reproducibility
1. **Document Parameters:** Save configuration parameters with datasets
2. **Version Control:** Track script versions and modifications
3. **Random Seeds:** Set consistent seeds for reproducible results
4. **Environment Recording:** Document Python and dependency versions

## API Reference

### CARLATestSequenceGenerator Class

#### Constructor
```python
CARLATestSequenceGenerator(args)
```
**Parameters:**
- `args`: Namespace object containing configuration parameters

#### Key Methods

##### `find_data_files()`
Locates and validates input data files.
**Returns:** `bool` - Success status

##### `load_ego_data()`
Loads ego vehicle data from CSV and applies filtering.
**Returns:** `bool` - Success status

##### `match_lidar_to_ego_data()`
Synchronizes LiDAR files with ego data timestamps.
**Returns:** `bool` - Success status

##### `create_test_sequences()`
Generates test sequences and organizes output files.
**Returns:** `bool` - Success status

##### `run()`
Executes the complete processing pipeline.
**Returns:** `bool` - Overall success status

### Utility Functions

#### `create_summary_file(num_sequences)`
Generates dataset summary with statistics and metadata.

#### `create_test_sequences()`
Core sequence generation logic with file organization.

### Error Handling

The script implements comprehensive error handling:
- **File I/O Errors:** Handles missing files and permission issues
- **Data Format Errors:** Validates CSV structure and content
- **Memory Errors:** Provides guidance for large dataset processing
- **Timestamp Errors:** Manages synchronization issues

## Changelog

### Version 1.0.0
- Initial release
- Basic sequence generation functionality
- CSV and LiDAR data processing
- Metadata generation

### Future Enhancements
- Multi-threading support for faster processing
- Advanced filtering options (lane changes, intersections)
- Integration with other CARLA sensors (cameras, radar)
- Automatic train/validation/test splitting
- Data visualization tools
- Support for other simulation platforms

## Support and Contributing

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Verify your data format matches requirements
3. Test with provided examples
4. Document any modifications for reproducibility

Remember to backup your original data before processing and validate generated sequences before using them for model training.