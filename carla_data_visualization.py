#!/usr/bin/env python

"""
CARLA Data Visualization Script
------------------------------
This script loads and visualizes data collected from the CARLA simulation, including:
- Ego vehicle trajectory with color-coded speed
- Lane geometry and boundaries
- Spatial points with drivability visualization
- Time-series plots of vehicle dynamics

Requirements:
- Python 3.7+
- numpy, pandas, matplotlib, seaborn
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import seaborn as sns
import argparse
from datetime import datetime
import math
from matplotlib.colors import LinearSegmentedColormap


class CARLADataVisualizer:
    def __init__(self, data_dir, output_dir=None):
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(data_dir, 'visualizations')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Data frames
        self.ego_data = None
        self.lane_data = None
        self.spatial_points = None
        
        # Initialize matplotlib style
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Set up matplotlib style for visualizations"""
        plt.style.use('seaborn')
        sns.set_context("notebook", font_scale=1.2)
    
    def load_data(self, timestamp=None):
        """
        Load data from CSV files
        
        Args:
            timestamp: Optional specific timestamp in data filenames to load.
                      If None, load the most recent dataset.
        """
        try:
            if timestamp:
                ego_file = glob.glob(os.path.join(self.data_dir, f"ego_data_{timestamp}*.csv"))
                lane_file = glob.glob(os.path.join(self.data_dir, f"lane_data_{timestamp}*.csv"))
                spatial_file = glob.glob(os.path.join(self.data_dir, f"spatial_points_{timestamp}*.csv"))
            else:
                # Find the most recent files
                ego_file = sorted(glob.glob(os.path.join(self.data_dir, "ego_data_*.csv")))
                lane_file = sorted(glob.glob(os.path.join(self.data_dir, "lane_data_*.csv")))
                spatial_file = sorted(glob.glob(os.path.join(self.data_dir, "spatial_points_*.csv")))
            
            # Load ego vehicle data
            if ego_file:
                ego_path = ego_file[-1] if isinstance(ego_file, list) else ego_file
                self.ego_data = pd.read_csv(ego_path)
                print(f"Loaded ego data: {os.path.basename(ego_path)}")
                print(f" - {len(self.ego_data)} frames")
                print(f" - Time range: {self.ego_data['timestamp'].min():.2f}s - {self.ego_data['timestamp'].max():.2f}s")
            else:
                print("No ego data files found")
            
            # Load lane data
            if lane_file:
                lane_path = lane_file[-1] if isinstance(lane_file, list) else lane_file
                self.lane_data = pd.read_csv(lane_path)
                print(f"Loaded lane data: {os.path.basename(lane_path)}")
                print(f" - {len(self.lane_data)} lane waypoints")
                print(f" - Unique lanes: {self.lane_data['lane_id'].nunique()}")
            else:
                print("No lane data files found")
            
            # Load spatial points data
            if spatial_file:
                spatial_path = spatial_file[-1] if isinstance(spatial_file, list) else spatial_file
                self.spatial_points = pd.read_csv(spatial_path)
                print(f"Loaded spatial points: {os.path.basename(spatial_path)}")
                print(f" - {len(self.spatial_points)} spatial points")
                print(f" - Drivable points: {sum(self.spatial_points['is_drivable'])}")
            else:
                print("No spatial points files found")
                
            return bool(self.ego_data is not None)
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def plot_trajectory_2d(self, show_lanes=True, show_points=True, show_colorbar=True, 
                           save=True, show=True):
        """
        Create a 2D plot of the vehicle trajectory with color-coded speed
        
        Args:
            show_lanes: Whether to show lane geometry
            show_points: Whether to show spatial points
            show_colorbar: Whether to show the speed colorbar
            save: Whether to save the plot to file
            show: Whether to display the plot
        """
        if self.ego_data is None:
            print("No ego data available. Call load_data() first.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate speed
        self.ego_data['speed'] = np.sqrt(
            self.ego_data['velocity_x']**2 + 
            self.ego_data['velocity_y']**2 + 
            self.ego_data['velocity_z']**2
        )
        
        # Plot spatial points if available and requested
        if show_points and self.spatial_points is not None:
            # Get points from the first timestamp to show the road structure
            first_timestamp = self.spatial_points['timestamp'].min()
            points_first = self.spatial_points[self.spatial_points['timestamp'] == first_timestamp]
            
            # Plot drivable and non-drivable points differently
            drivable = points_first[points_first['is_drivable'] == True]
            non_drivable = points_first[points_first['is_drivable'] == False]
            
            ax.scatter(drivable['x'], drivable['y'], color='lightgreen', 
                       alpha=0.3, s=10, label='Drivable Area')
            ax.scatter(non_drivable['x'], non_drivable['y'], color='lightgrey', 
                       alpha=0.2, s=5, label='Non-Drivable Area')
        
        # Plot lane geometry if available and requested
        if show_lanes and self.lane_data is not None:
            # Get lanes from one timestamp to avoid duplicates
            first_timestamp = self.lane_data['timestamp'].min()
            lanes_first = self.lane_data[self.lane_data['timestamp'] == first_timestamp]
            
            # Group by lane_id and plot each lane
            for lane_id, lane_group in lanes_first.groupby('lane_id'):
                # Sort waypoints to ensure proper line connection
                lane_group = lane_group.sort_values(by=['waypoint_x', 'waypoint_y'])
                
                # Different colors for different lane types
                color = 'blue' if lane_group['is_drivable'].iloc[0] else 'red'
                alpha = 0.8 if lane_group['is_drivable'].iloc[0] else 0.4
                
                ax.plot(lane_group['waypoint_x'], lane_group['waypoint_y'], 
                        color=color, alpha=alpha, linewidth=2)
        
        # Plot vehicle trajectory with speed-based coloring
        points = np.array([self.ego_data['x'], self.ego_data['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a line collection for the trajectory
        norm = plt.Normalize(0, self.ego_data['speed'].max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(self.ego_data['speed'])
        lc.set_linewidth(3)
        line = ax.add_collection(lc)
        
        # Show colorbar if requested
        if show_colorbar:
            cbar = fig.colorbar(line, ax=ax, label='Speed (m/s)')
        
        # Plot start and end points of trajectory
        ax.scatter(self.ego_data['x'].iloc[0], self.ego_data['y'].iloc[0], 
                   color='green', s=100, zorder=5, label='Start')
        ax.scatter(self.ego_data['x'].iloc[-1], self.ego_data['y'].iloc[-1], 
                   color='red', s=100, zorder=5, label='End')
        
        # Set plot labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Vehicle Trajectory with Speed Coloring')
        ax.axis('equal')
        ax.legend()
        
        # Add a text box with some statistics
        max_speed = self.ego_data['speed'].max()
        avg_speed = self.ego_data['speed'].mean()
        total_time = self.ego_data['timestamp'].max() - self.ego_data['timestamp'].min()
        
        stats_text = (f"Max Speed: {max_speed:.2f} m/s ({max_speed*3.6:.2f} km/h)\n"
                     f"Avg Speed: {avg_speed:.2f} m/s ({avg_speed*3.6:.2f} km/h)\n"
                     f"Total Time: {total_time:.2f} s")
        
        plt.figtext(0.02, 0.02, stats_text, backgroundcolor='white', 
                   bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"trajectory_2d_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved trajectory plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_dynamics(self, save=True, show=True):
        """
        Plot vehicle dynamics time series data
        
        Args:
            save: Whether to save the plot to file
            show: Whether to display the plot
        """
        if self.ego_data is None:
            print("No ego data available. Call load_data() first.")
            return
        
        # Create subplots for different data
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot 1: Steering and Heading
        ax1 = axs[0]
        ax1.plot(self.ego_data['timestamp'], self.ego_data['steering'], 
                color='blue', label='Steering')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(self.ego_data['timestamp'], self.ego_data['yaw'], 
                     color='orange', label='Yaw')
        
        ax1.set_ylabel('Steering')
        ax1_twin.set_ylabel('Yaw (degrees)')
        ax1.set_title('Steering and Orientation')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 2: Velocity
        ax2 = axs[1]
        ax2.plot(self.ego_data['timestamp'], self.ego_data['velocity_x'], 
                label='Velocity X', color='red')
        ax2.plot(self.ego_data['timestamp'], self.ego_data['velocity_y'], 
                label='Velocity Y', color='green')
        
        # Calculate and plot speed
        self.ego_data['speed'] = np.sqrt(
            self.ego_data['velocity_x']**2 + 
            self.ego_data['velocity_y']**2 + 
            self.ego_data['velocity_z']**2
        )
        ax2.plot(self.ego_data['timestamp'], self.ego_data['speed'], 
                label='Speed', color='blue', linewidth=2)
        
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Vehicle Velocity')
        ax2.legend()
        
        # Plot 3: Acceleration
        ax3 = axs[2]
        ax3.plot(self.ego_data['timestamp'], self.ego_data['accel_x'], 
                label='Accel X', color='red')
        ax3.plot(self.ego_data['timestamp'], self.ego_data['accel_y'], 
                label='Accel Y', color='green')
        
        # Calculate total acceleration
        self.ego_data['accel_total'] = np.sqrt(
            self.ego_data['accel_x']**2 + 
            self.ego_data['accel_y']**2 + 
            self.ego_data['accel_z']**2
        )
        ax3.plot(self.ego_data['timestamp'], self.ego_data['accel_total'], 
                label='Total Accel', color='blue', linewidth=2)
        
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Vehicle Acceleration')
        ax3.legend()
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"dynamics_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved dynamics plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_trajectory_animation(self, radius=30, save=True, show=True):
        """
        Create an animation of the vehicle trajectory with surrounding points
        
        Args:
            radius: Radius around vehicle to show (meters)
            save: Whether to save the animation to file
            show: Whether to display the animation
        """
        if self.ego_data is None or self.spatial_points is None:
            print("Both ego data and spatial points are required. Call load_data() first.")
            return
        
        # Get unique timestamps where we have both ego and spatial data
        ego_timestamps = set(self.ego_data['timestamp'])
        spatial_timestamps = set(self.spatial_points['timestamp'])
        common_timestamps = sorted(list(ego_timestamps.intersection(spatial_timestamps)))
        
        if not common_timestamps:
            print("No common timestamps found between ego and spatial data")
            return
        
        # Subsample timestamps for smoother animation (every 2 frames)
        common_timestamps = common_timestamps[::2]
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Function to update the animation
        def update(frame):
            ax.clear()
            timestamp = common_timestamps[frame]
            
            # Get data for this timestamp
            ego_frame = self.ego_data[self.ego_data['timestamp'] == timestamp].iloc[0]
            spatial_frame = self.spatial_points[self.spatial_points['timestamp'] == timestamp]
            
            # Plot spatial points
            drivable = spatial_frame[spatial_frame['is_drivable'] == True]
            non_drivable = spatial_frame[spatial_frame['is_drivable'] == False]
            
            ax.scatter(drivable['x'], drivable['y'], color='lightgreen', 
                      alpha=0.5, s=10, label='Drivable Area')
            ax.scatter(non_drivable['x'], non_drivable['y'], color='lightgrey', 
                      alpha=0.2, s=5, label='Non-Drivable Area')
            
            # Plot vehicle position with a marker showing heading
            vehicle_x, vehicle_y = ego_frame['x'], ego_frame['y']
            heading = ego_frame['heading']
            
            # Draw the vehicle as a triangle pointing in the heading direction
            vehicle_length = 4.5  # Typical car length in meters
            heading_rad = math.radians(heading)
            dx = vehicle_length * math.cos(heading_rad)
            dy = vehicle_length * math.sin(heading_rad)
            
            # Plot vehicle as a triangle
            ax.plot([vehicle_x, vehicle_x + dx], [vehicle_y, vehicle_y + dy], 
                   color='red', linewidth=3)
            ax.scatter(vehicle_x, vehicle_y, color='red', s=100, zorder=5)
            
            # Plot trajectory up to this point
            ego_subset = self.ego_data[self.ego_data['timestamp'] <= timestamp]
            ax.plot(ego_subset['x'], ego_subset['y'], color='blue', alpha=0.7, linewidth=2)
            
            # Set plot labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Vehicle Trajectory Animation (Time: {timestamp:.2f}s)')
            
            # Set fixed axis limits around the vehicle
            ax.set_xlim(vehicle_x - radius, vehicle_x + radius)
            ax.set_ylim(vehicle_y - radius, vehicle_y + radius)
            ax.axis('equal')
            
            # Add speed, steering info
            speed = np.sqrt(ego_frame['velocity_x']**2 + ego_frame['velocity_y']**2)
            stats_text = (f"Speed: {speed:.2f} m/s ({speed*3.6:.2f} km/h)\n"
                         f"Steering: {ego_frame['steering']:.2f}\n"
                         f"Heading: {heading:.1f}°")
            
            ax.text(vehicle_x - radius + 2, vehicle_y - radius + 2, stats_text,
                   backgroundcolor='white', bbox=dict(facecolor='white', alpha=0.8),
                   fontsize=10)
            
            return ax,
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(common_timestamps),
            interval=100, blit=False)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"trajectory_animation_{timestamp}.mp4")
            writer = animation.FFMpegWriter(fps=10, bitrate=1800)
            ani.save(filepath, writer=writer)
            print(f"Saved animation to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return ani
    
    def plot_lane_heatmap(self, save=True, show=True):
        """
        Create a heatmap showing lane occupancy over time
        
        Args:
            save: Whether to save the plot to file
            show: Whether to display the plot
        """
        if self.ego_data is None or self.lane_data is None:
            print("Both ego and lane data are required. Call load_data() first.")
            return
        
        # Get vehicle position at each timestamp
        vehicle_positions = {}
        for _, row in self.ego_data.iterrows():
            vehicle_positions[row['timestamp']] = (row['x'], row['y'])
        
        # Find closest lane ID for each vehicle position
        lane_occupancy = {}
        
        # Group lane data by timestamp
        for timestamp, group in self.lane_data.groupby('timestamp'):
            if timestamp not in vehicle_positions:
                continue
                
            veh_x, veh_y = vehicle_positions[timestamp]
            
            # Find the closest waypoint
            min_dist = float('inf')
            closest_lane = None
            
            for _, lane_row in group.iterrows():
                wp_x, wp_y = lane_row['waypoint_x'], lane_row['waypoint_y']
                dist = np.sqrt((veh_x - wp_x)**2 + (veh_y - wp_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_lane = lane_row['lane_id']
            
            if closest_lane:
                if closest_lane not in lane_occupancy:
                    lane_occupancy[closest_lane] = 0
                lane_occupancy[closest_lane] += 1
        
        # Create a DataFrame for the heatmap
        lane_ids = sorted(lane_occupancy.keys())
        occupancy_data = [lane_occupancy.get(lane_id, 0) for lane_id in lane_ids]
        
        lane_df = pd.DataFrame({
            'lane_id': lane_ids,
            'occupancy': occupancy_data
        })
        
        # Sort by occupancy for better visualization
        lane_df = lane_df.sort_values('occupancy', ascending=False)
        
        # Create the heatmap plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='lane_id', y='occupancy', data=lane_df, palette='viridis')
        
        plt.title('Lane Occupancy')
        plt.xlabel('Lane ID')
        plt.ylabel('Frames with Vehicle Present')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"lane_occupancy_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved lane occupancy plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_all(self, show=True):
        """Generate all visualizations"""
        self.plot_trajectory_2d(save=True, show=show)
        self.plot_dynamics(save=True, show=show)
        self.plot_lane_heatmap(save=True, show=show)
        self.create_trajectory_animation(save=True, show=show)
        
        print("All visualizations complete!")


def main():
    parser = argparse.ArgumentParser(description='CARLA Data Visualization')
    parser.add_argument('--data-dir', default='./carla_data', 
                        help='Directory containing the CARLA data CSV files')
    parser.add_argument('--output-dir', default=None,
                        help='Directory to save visualizations (default: data_dir/visualizations)')
    parser.add_argument('--timestamp', default=None,
                        help='Specific timestamp to load data for (format: YYYYMMDD_HHMMSS)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (just save them)')
    parser.add_argument('--plot-type', choices=['all', 'trajectory', 'dynamics', 'lane', 'animation'],
                        default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    visualizer = CARLADataVisualizer(args.data_dir, args.output_dir)
    
    if visualizer.load_data(args.timestamp):
        show = not args.no_show
        
        if args.plot_type == 'all':
            visualizer.visualize_all(show=show)
        elif args.plot_type == 'trajectory':
            visualizer.plot_trajectory_2d(save=True, show=show)
        elif args.plot_type == 'dynamics':
            visualizer.plot_dynamics(save=True, show=show)
        elif args.plot_type == 'lane':
            visualizer.plot_lane_heatmap(save=True, show=show)
        elif args.plot_type == 'animation':
            visualizer.create_trajectory_animation(save=True, show=show)
    else:
        print("Failed to load data. Exiting.")


if __name__ == '__main__':
    main()