"""
================================================================================
UNITREE G1 ROBOT - COMPREHENSIVE IMU SENSOR ANALYSIS PIPELINE
================================================================================

This pipeline provides end-to-end sensor data analysis including:
- Automatic data loading and processing
- Statistical analysis (mean, std, RMSE, min/max)
- Research-level graph generation
- Organized output structure with auto-incrementing run folders
- Detailed reports with interpretations

Author: Sensor Analysis System
Date: 2024
================================================================================
"""

import os
import sys
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'OUTPUT_BASE': 'project_outputs',
    'DATA_FILE': 'upper_body_sensor_data.csv',
    'PLOT_DPI': 300,
    'PLOT_FIGSIZE': (20, 14),
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_next_run_number(base_dir):
    """
    Get the next available run number by checking existing folders.
    
    Args:
        base_dir (str): Base output directory
        
    Returns:
        int: Next run number
    """
    if not os.path.exists(base_dir):
        return 1
    
    run_folders = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    if not run_folders:
        return 1
    
    numbers = [int(d.split('_')[1]) for d in run_folders if d.split('_')[1].isdigit()]
    return max(numbers) + 1 if numbers else 1


def create_output_directories(base_dir, run_num):
    """
    Create organized output directory structure.
    
    Args:
        base_dir (str): Base output directory
        run_num (int): Run number
        
    Returns:
        dict: Paths to all output directories
    """
    run_dir = f"{base_dir}/run_{run_num:03d}"
    
    dirs = {
        'run': run_dir,
        'graphs': f"{run_dir}/graphs",
        'data': f"{run_dir}/data",
        'report': f"{run_dir}/report"
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    return dirs


def log_message(message, level="INFO"):
    """Print formatted console messages."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ============================================================================
# DATA LOADING
# ============================================================================

class SensorDataLoader:
    """Load and validate sensor data from CSV file."""
    
    def __init__(self, filepath):
        """
        Initialize loader with data file.
        
        Args:
            filepath (str): Path to CSV file
        """
        self.filepath = filepath
        self.df = None
        
    def load(self):
        """
        Load data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        
        try:
            self.df = pd.read_csv(self.filepath)
            log_message(f"Loaded data: {len(self.df)} samples")
            return self.df
        except Exception as e:
            log_message(f"Error loading data: {e}", "ERROR")
            raise
    
    def validate(self):
        """Validate data integrity."""
        required_cols = [
            'Time(s)', 'Torso_Pos(rad)', 'Torso_Vel(rad/s)',
            'IMU_AngVel_X', 'IMU_AngVel_Y', 'IMU_AngVel_Z',
            'Gravity_X', 'Gravity_Y', 'Gravity_Z',
            'BaseLinVel_X', 'BaseLinVel_Y', 'BaseLinVel_Z'
        ]
        
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            log_message(f"Missing columns: {missing}", "WARNING")
        
        return len(missing) == 0


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class SensorAnalyzer:
    """Compute statistical metrics for sensor data."""
    
    def __init__(self, df):
        """
        Initialize analyzer with data.
        
        Args:
            df (pd.DataFrame): Sensor data
        """
        self.df = df
        self.stats = {}
        
    def compute_rmse(self, signal):
        """
        Compute mean-centered RMSE.
        
        Formula: RMSE = sqrt(mean((x - mean(x))^2))
        
        Args:
            signal (np.array): Signal data
            
        Returns:
            float: RMSE value
        """
        mean = np.mean(signal)
        return np.sqrt(np.mean((signal - mean) ** 2))
    
    def compute_magnitude(self, x, y, z):
        """
        Compute 3D magnitude: sqrt(x^2 + y^2 + z^2)
        
        Args:
            x, y, z (np.array): Components
            
        Returns:
            np.array: Magnitude array
        """
        return np.sqrt(x**2 + y**2 + z**2)
    
    def analyze_chest(self):
        """Analyze chest (torso) sensor data."""
        torso_pos = self.df['Torso_Pos(rad)'].values
        torso_vel = self.df['Torso_Vel(rad/s)'].values
        
        self.stats['chest'] = {
            'position': {
                'mean': float(np.mean(torso_pos)),
                'std': float(np.std(torso_pos)),
                'rmse': float(self.compute_rmse(torso_pos)),
                'min': float(np.min(torso_pos)),
                'max': float(np.max(torso_pos)),
                'range': float(np.max(torso_pos) - np.min(torso_pos))
            },
            'velocity': {
                'mean': float(np.mean(torso_vel)),
                'std': float(np.std(torso_vel)),
                'rmse': float(self.compute_rmse(torso_vel)),
                'min': float(np.min(torso_vel)),
                'max': float(np.max(torso_vel)),
                'range': float(np.max(torso_vel) - np.min(torso_vel))
            }
        }
    
    def analyze_shoulder(self):
        """Analyze shoulder (IMU angular velocity) data."""
        ang_vel_x = self.df['IMU_AngVel_X'].values
        ang_vel_y = self.df['IMU_AngVel_Y'].values
        ang_vel_z = self.df['IMU_AngVel_Z'].values
        
        # Compute magnitude
        magnitude = self.compute_magnitude(ang_vel_x, ang_vel_y, ang_vel_z)
        
        self.stats['shoulder'] = {
            'angular_velocity_x': {
                'mean': float(np.mean(ang_vel_x)),
                'std': float(np.std(ang_vel_x)),
                'rmse': float(self.compute_rmse(ang_vel_x)),
                'min': float(np.min(ang_vel_x)),
                'max': float(np.max(ang_vel_x))
            },
            'angular_velocity_y': {
                'mean': float(np.mean(ang_vel_y)),
                'std': float(np.std(ang_vel_y)),
                'rmse': float(self.compute_rmse(ang_vel_y)),
                'min': float(np.min(ang_vel_y)),
                'max': float(np.max(ang_vel_y))
            },
            'angular_velocity_z': {
                'mean': float(np.mean(ang_vel_z)),
                'std': float(np.std(ang_vel_z)),
                'rmse': float(self.compute_rmse(ang_vel_z)),
                'min': float(np.min(ang_vel_z)),
                'max': float(np.max(ang_vel_z))
            },
            'total_angular_velocity': {
                'mean': float(np.mean(magnitude)),
                'std': float(np.std(magnitude)),
                'rmse': float(self.compute_rmse(magnitude)),
                'min': float(np.min(magnitude)),
                'max': float(np.max(magnitude))
            }
        }
    
    def analyze_back(self):
        """Analyze back (gravity orientation) data."""
        gravity_x = self.df['Gravity_X'].values
        gravity_y = self.df['Gravity_Y'].values
        gravity_z = self.df['Gravity_Z'].values
        
        magnitude = self.compute_magnitude(gravity_x, gravity_y, gravity_z)
        
        self.stats['back'] = {
            'gravity_x': {
                'mean': float(np.mean(gravity_x)),
                'std': float(np.std(gravity_x)),
                'rmse': float(self.compute_rmse(gravity_x)),
                'min': float(np.min(gravity_x)),
                'max': float(np.max(gravity_x))
            },
            'gravity_y': {
                'mean': float(np.mean(gravity_y)),
                'std': float(np.std(gravity_y)),
                'rmse': float(self.compute_rmse(gravity_y)),
                'min': float(np.min(gravity_y)),
                'max': float(np.max(gravity_y))
            },
            'gravity_z': {
                'mean': float(np.mean(gravity_z)),
                'std': float(np.std(gravity_z)),
                'rmse': float(self.compute_rmse(gravity_z)),
                'min': float(np.min(gravity_z)),
                'max': float(np.max(gravity_z))
            },
            'total_gravity_magnitude': {
                'mean': float(np.mean(magnitude)),
                'std': float(np.std(magnitude)),
                'rmse': float(self.compute_rmse(magnitude)),
                'min': float(np.min(magnitude)),
                'max': float(np.max(magnitude))
            }
        }
    
    def analyze_base_velocity(self):
        """Analyze base linear velocity data."""
        vel_x = self.df['BaseLinVel_X'].values
        vel_y = self.df['BaseLinVel_Y'].values
        vel_z = self.df['BaseLinVel_Z'].values
        
        magnitude = self.compute_magnitude(vel_x, vel_y, vel_z)
        
        self.stats['base_velocity'] = {
            'linear_velocity_x': {
                'mean': float(np.mean(vel_x)),
                'std': float(np.std(vel_x)),
                'rmse': float(self.compute_rmse(vel_x)),
                'min': float(np.min(vel_x)),
                'max': float(np.max(vel_x))
            },
            'linear_velocity_y': {
                'mean': float(np.mean(vel_y)),
                'std': float(np.std(vel_y)),
                'rmse': float(self.compute_rmse(vel_y)),
                'min': float(np.min(vel_y)),
                'max': float(np.max(vel_y))
            },
            'linear_velocity_z': {
                'mean': float(np.mean(vel_z)),
                'std': float(np.std(vel_z)),
                'rmse': float(self.compute_rmse(vel_z)),
                'min': float(np.min(vel_z)),
                'max': float(np.max(vel_z))
            },
            'total_linear_velocity': {
                'mean': float(np.mean(magnitude)),
                'std': float(np.std(magnitude)),
                'rmse': float(self.compute_rmse(magnitude)),
                'min': float(np.min(magnitude)),
                'max': float(np.max(magnitude))
            }
        }
    
    def analyze(self):
        """Run all analyses."""
        log_message("Computing statistical analysis...")
        self.analyze_chest()
        self.analyze_shoulder()
        self.analyze_back()
        self.analyze_base_velocity()
        log_message("Statistical analysis complete")
        return self.stats


# ============================================================================
# VISUALIZATION
# ============================================================================

class SensorVisualizer:
    """Generate research-quality plots."""
    
    def __init__(self, df, stats):
        """
        Initialize visualizer.
        
        Args:
            df (pd.DataFrame): Sensor data
            stats (dict): Statistical analysis results
        """
        self.df = df
        self.stats = stats
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def create_comprehensive_plot(self, output_path):
        """
        Create comprehensive 9-panel research-quality plot.
        
        Args:
            output_path (str): Save path
        """
        log_message("Generating comprehensive 9-panel plot...")
        
        fig = plt.figure(figsize=CONFIG['PLOT_FIGSIZE'])
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        time = self.df['Time(s)'].values
        
        # ====== CHEST ROW ======
        # 1. Chest Position vs Time
        ax = fig.add_subplot(gs[0, 0])
        torso_pos = self.df['Torso_Pos(rad)'].values
        ax.plot(time, torso_pos, 'b-', linewidth=2.5, label='Position')
        ax.fill_between(time, torso_pos, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Position (rad)', fontsize=11, fontweight='bold')
        rmse_val = self.stats['chest']['position']['rmse']
        ax.set_title(f'CHEST: Position vs Time\nRMSE = {rmse_val:.4f} rad', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 2. Chest Velocity vs Time
        ax = fig.add_subplot(gs[0, 1])
        torso_vel = self.df['Torso_Vel(rad/s)'].values
        ax.plot(time, torso_vel, 'r-', linewidth=2.5, label='Velocity')
        ax.fill_between(time, torso_vel, alpha=0.3, color='red')
        ax.axhline(y=np.mean(torso_vel), color='k', linestyle='--', 
                   alpha=0.5, label='Mean')
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Velocity (rad/s)', fontsize=11, fontweight='bold')
        rmse_val = self.stats['chest']['velocity']['rmse']
        ax.set_title(f'CHEST: Velocity vs Time\nRMSE = {rmse_val:.4f} rad/s', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 3. Chest 2D Trajectory
        ax = fig.add_subplot(gs[0, 2])
        base_vel_x = self.df['BaseLinVel_X'].values
        base_vel_y = self.df['BaseLinVel_Y'].values
        ax.plot(base_vel_x, base_vel_y, 'g-', linewidth=2.5)
        ax.fill(base_vel_x, base_vel_y, alpha=0.3, color='green')
        ax.scatter(base_vel_x[0], base_vel_y[0], s=100, c='blue', 
                  marker='o', label='Start', zorder=5)
        ax.scatter(base_vel_x[-1], base_vel_y[-1], s=100, c='red', 
                  marker='s', label='End', zorder=5)
        ax.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
        ax.set_title('CHEST: 2D Trajectory', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # ====== SHOULDER ROW ======
        # 4. Shoulder Angular Velocity - X axis
        ax = fig.add_subplot(gs[1, 0])
        ang_x = self.df['IMU_AngVel_X'].values
        ax.plot(time, ang_x, 'purple', linewidth=2.5, label='X-axis')
        ax.fill_between(time, ang_x, alpha=0.3, color='purple')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
        rmse_val = self.stats['shoulder']['angular_velocity_x']['rmse']
        ax.set_title(f'SHOULDER: X-axis Angular Velocity\nRMSE = {rmse_val:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 5. Shoulder Angular Velocity - Y axis
        ax = fig.add_subplot(gs[1, 1])
        ang_y = self.df['IMU_AngVel_Y'].values
        ax.plot(time, ang_y, 'orange', linewidth=2.5, label='Y-axis')
        ax.fill_between(time, ang_y, alpha=0.3, color='orange')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
        rmse_val = self.stats['shoulder']['angular_velocity_y']['rmse']
        ax.set_title(f'SHOULDER: Y-axis Angular Velocity\nRMSE = {rmse_val:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 6. Shoulder 2D Angular Velocity
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(ang_x, ang_y, 'brown', linewidth=2.5)
        ax.fill(ang_x, ang_y, alpha=0.3, color='brown')
        ax.scatter(ang_x[0], ang_y[0], s=100, c='blue', 
                  marker='o', label='Start', zorder=5)
        ax.scatter(ang_x[-1], ang_y[-1], s=100, c='red', 
                  marker='s', label='End', zorder=5)
        ax.set_xlabel('X Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
        ax.set_title('SHOULDER: 2D Angular Velocity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # ====== BACK ROW ======
        # 7. Back Gravity - X component
        ax = fig.add_subplot(gs[2, 0])
        grav_x = self.df['Gravity_X'].values
        ax.plot(time, grav_x, 'darkgreen', linewidth=2.5, label='Gravity X')
        ax.fill_between(time, grav_x, alpha=0.3, color='darkgreen')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gravity Component', fontsize=11, fontweight='bold')
        rmse_val = self.stats['back']['gravity_x']['rmse']
        ax.set_title(f'BACK: X Gravity vs Time\nRMSE = {rmse_val:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 8. Back Gravity - Y component
        ax = fig.add_subplot(gs[2, 1])
        grav_y = self.df['Gravity_Y'].values
        ax.plot(time, grav_y, 'teal', linewidth=2.5, label='Gravity Y')
        ax.fill_between(time, grav_y, alpha=0.3, color='teal')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gravity Component', fontsize=11, fontweight='bold')
        rmse_val = self.stats['back']['gravity_y']['rmse']
        ax.set_title(f'BACK: Y Gravity vs Time\nRMSE = {rmse_val:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 9. Back 2D Gravity Orientation
        ax = fig.add_subplot(gs[2, 2])
        ax.plot(grav_x, grav_y, 'darkred', linewidth=2.5)
        ax.fill(grav_x, grav_y, alpha=0.3, color='darkred')
        ax.scatter(grav_x[0], grav_y[0], s=100, c='blue', 
                  marker='o', label='Start', zorder=5)
        ax.scatter(grav_x[-1], grav_y[-1], s=100, c='red', 
                  marker='s', label='End', zorder=5)
        ax.set_xlabel('Gravity X', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gravity Y', fontsize=11, fontweight='bold')
        ax.set_title('BACK: 2D Gravity Orientation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.savefig(output_path, dpi=CONFIG['PLOT_DPI'], bbox_inches='tight')
        log_message(f"Saved: {output_path}")
        plt.close()
    
    def create_summary_comparison(self, output_path):
        """
        Create RMSE comparison bar chart.
        
        Args:
            output_path (str): Save path
        """
        log_message("Generating RMSE comparison chart...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # RMSE Comparison
        ax = axes[0, 0]
        rmse_data = {
            'Chest Pos': self.stats['chest']['position']['rmse'],
            'Chest Vel': self.stats['chest']['velocity']['rmse'],
            'Shoulder X': self.stats['shoulder']['angular_velocity_x']['rmse'],
            'Shoulder Y': self.stats['shoulder']['angular_velocity_y']['rmse'],
            'Shoulder Z': self.stats['shoulder']['angular_velocity_z']['rmse'],
            'Back X': self.stats['back']['gravity_x']['rmse'],
            'Back Y': self.stats['back']['gravity_y']['rmse'],
            'Back Z': self.stats['back']['gravity_z']['rmse'],
        }
        colors = ['#FF6B6B']*2 + ['#4ECDC4']*3 + ['#95A5A6']*3
        bars = ax.barh(list(rmse_data.keys()), list(rmse_data.values()), color=colors)
        ax.set_xlabel('RMSE Value', fontsize=12, fontweight='bold')
        ax.set_title('RMSE Comparison Across All Sensors', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, rmse_data.values())):
            ax.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10)
        
        # Chest Stats
        ax = axes[0, 1]
        chest_stats = [
            self.stats['chest']['position']['mean'],
            self.stats['chest']['position']['std'],
            self.stats['chest']['velocity']['mean'],
            self.stats['chest']['velocity']['std'],
        ]
        labels_chest = ['Pos Mean', 'Pos Std', 'Vel Mean', 'Vel Std']
        ax.bar(labels_chest, chest_stats, color=['#FF6B6B', '#FF8E8E', '#FF5252', '#FF3333'])
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('CHEST: Position & Velocity Stats', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Shoulder Stats
        ax = axes[1, 0]
        time = self.df['Time(s)'].values
        ang_x = self.df['IMU_AngVel_X'].values
        ang_y = self.df['IMU_AngVel_Y'].values
        ang_z = self.df['IMU_AngVel_Z'].values
        ax.plot(time, ang_x, label='X', linewidth=2)
        ax.plot(time, ang_y, label='Y', linewidth=2)
        ax.plot(time, ang_z, label='Z', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
        ax.set_title('SHOULDER: 3-Axis Angular Velocity Timeline', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Back Stats
        ax = axes[1, 1]
        grav_x = self.df['Gravity_X'].values
        grav_y = self.df['Gravity_Y'].values
        grav_z = self.df['Gravity_Z'].values
        ax.plot(time, grav_x, label='X', linewidth=2)
        ax.plot(time, grav_y, label='Y', linewidth=2)
        ax.plot(time, grav_z, label='Z', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gravity Component', fontsize=12, fontweight='bold')
        ax.set_title('BACK: 3-Axis Gravity Orientation Timeline', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=CONFIG['PLOT_DPI'], bbox_inches='tight')
        log_message(f"Saved: {output_path}")
        plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Generate detailed analysis reports."""
    
    def __init__(self, df, stats):
        """
        Initialize report generator.
        
        Args:
            df (pd.DataFrame): Sensor data
            stats (dict): Statistical analysis results
        """
        self.df = df
        self.stats = stats
    
    def interpret_chest(self):
        """Interpret chest sensor behavior."""
        pos_rmse = self.stats['chest']['position']['rmse']
        vel_rmse = self.stats['chest']['velocity']['rmse']
        vel_range = self.stats['chest']['velocity']['range']
        
        interpretation = "CHEST ANALYSIS:\n"
        
        if pos_rmse < 0.01:
            interpretation += "  • Position: VERY STABLE (minimal deviation)\n"
        elif pos_rmse < 0.05:
            interpretation += "  • Position: STABLE (low variance)\n"
        else:
            interpretation += "  • Position: DYNAMIC (significant movement)\n"
        
        if vel_rmse < 0.05:
            interpretation += "  • Velocity: MINIMAL oscillation\n"
        elif vel_rmse < 0.15:
            interpretation += "  • Velocity: MODERATE oscillation\n"
        else:
            interpretation += "  • Velocity: HIGH oscillation\n"
        
        if vel_range > 0.3:
            interpretation += "  • Motion: PRONOUNCED (large velocity swings)\n"
        else:
            interpretation += "  • Motion: CONTROLLED (smooth transitions)\n"
        
        return interpretation
    
    def interpret_shoulder(self):
        """Interpret shoulder sensor behavior."""
        mag_rmse = self.stats['shoulder']['total_angular_velocity']['rmse']
        mag_mean = self.stats['shoulder']['total_angular_velocity']['mean']
        
        interpretation = "SHOULDER ANALYSIS:\n"
        
        if mag_mean < 0.2:
            interpretation += "  • Rotation: MINIMAL average rotation\n"
        elif mag_mean < 0.5:
            interpretation += "  • Rotation: MODERATE average shoulder movement\n"
        else:
            interpretation += "  • Rotation: SIGNIFICANT shoulder motion\n"
        
        if mag_rmse < 0.15:
            interpretation += "  • Stability: STABLE rotation pattern\n"
        elif mag_rmse < 0.4:
            interpretation += "  • Stability: MODERATE variation in rotation\n"
        else:
            interpretation += "  • Stability: HIGH variation (dynamic motion)\n"
        
        x_rmse = self.stats['shoulder']['angular_velocity_x']['rmse']
        y_rmse = self.stats['shoulder']['angular_velocity_y']['rmse']
        z_rmse = self.stats['shoulder']['angular_velocity_z']['rmse']
        
        dominant = max([('X', x_rmse), ('Y', y_rmse), ('Z', z_rmse)], key=lambda x: x[1])
        interpretation += f"  • Dominant axis: {dominant[0]}-axis with RMSE={dominant[1]:.4f}\n"
        
        return interpretation
    
    def interpret_back(self):
        """Interpret back sensor behavior."""
        grav_z = self.stats['back']['gravity_z']['mean']
        grav_z_std = self.stats['back']['gravity_z']['std']
        
        interpretation = "BACK ANALYSIS:\n"
        
        # Z should be close to -1 (down)
        z_error = abs(grav_z - (-1.0))
        if z_error < 0.1:
            interpretation += "  • Stability: UPRIGHT posture maintained\n"
        elif z_error < 0.3:
            interpretation += "  • Stability: SLIGHT forward/backward lean\n"
        else:
            interpretation += "  • Stability: SIGNIFICANT posture deviation\n"
        
        if grav_z_std < 0.05:
            interpretation += "  • Consistency: VERY STABLE Z-axis (minimal tilt)\n"
        elif grav_z_std < 0.15:
            interpretation += "  • Consistency: MODERATE Z-axis variation\n"
        else:
            interpretation += "  • Consistency: HIGH Z-axis variation (dynamic tilt)\n"
        
        x_std = self.stats['back']['gravity_x']['std']
        y_std = self.stats['back']['gravity_y']['std']
        
        if x_std > y_std:
            interpretation += f"  • Primary tilt: LATERAL (side-to-side, X-std={x_std:.4f})\n"
        else:
            interpretation += f"  • Primary tilt: FORWARD/BACKWARD (Y-std={y_std:.4f})\n"
        
        return interpretation
    
    def generate_text_report(self, output_path):
        """Generate comprehensive text report."""
        log_message("Generating detailed text report...")
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("UNITREE G1 ROBOT - UPPER BODY IMU SENSOR ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(self.df)}\n")
            f.write(f"Duration: {self.df['Time(s)'].iloc[-1]:.2f} seconds\n")
            f.write(f"Sampling Rate: {len(self.df)/self.df['Time(s)'].iloc[-1]:.2f} Hz\n\n")
            
            # ========== CHEST ==========
            f.write("=" * 80 + "\n")
            f.write("1. CHEST (TORSO) SENSOR ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            chest_pos = self.stats['chest']['position']
            f.write("Position Statistics:\n")
            f.write(f"  Mean: {chest_pos['mean']:>10.6f} rad\n")
            f.write(f"  Std Dev: {chest_pos['std']:>9.6f} rad\n")
            f.write(f"  RMSE: {chest_pos['rmse']:>10.6f} rad (mean-centered)\n")
            f.write(f"  Min: {chest_pos['min']:>11.6f} rad\n")
            f.write(f"  Max: {chest_pos['max']:>11.6f} rad\n")
            f.write(f"  Range: {chest_pos['range']:>10.6f} rad\n\n")
            
            chest_vel = self.stats['chest']['velocity']
            f.write("Velocity Statistics:\n")
            f.write(f"  Mean: {chest_vel['mean']:>10.6f} rad/s\n")
            f.write(f"  Std Dev: {chest_vel['std']:>9.6f} rad/s\n")
            f.write(f"  RMSE: {chest_vel['rmse']:>10.6f} rad/s (mean-centered)\n")
            f.write(f"  Min: {chest_vel['min']:>11.6f} rad/s\n")
            f.write(f"  Max: {chest_vel['max']:>11.6f} rad/s\n")
            f.write(f"  Range: {chest_vel['range']:>10.6f} rad/s\n\n")
            
            f.write("Interpretation:\n")
            f.write(self.interpret_chest())
            
            # ========== SHOULDER ==========
            f.write("\n" + "=" * 80 + "\n")
            f.write("2. SHOULDER (IMU ANGULAR VELOCITY) ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            for axis in ['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z']:
                axis_name = axis.split('_')[-1].upper()
                stats = self.stats['shoulder'][axis]
                f.write(f"{axis_name}-axis Statistics:\n")
                f.write(f"  Mean: {stats['mean']:>10.6f} rad/s\n")
                f.write(f"  Std Dev: {stats['std']:>9.6f} rad/s\n")
                f.write(f"  RMSE: {stats['rmse']:>10.6f} rad/s (mean-centered)\n")
                f.write(f"  Min: {stats['min']:>11.6f} rad/s\n")
                f.write(f"  Max: {stats['max']:>11.6f} rad/s\n\n")
            
            total_vel = self.stats['shoulder']['total_angular_velocity']
            f.write("Total Angular Velocity Magnitude:\n")
            f.write(f"  Mean: {total_vel['mean']:>10.6f} rad/s\n")
            f.write(f"  Std Dev: {total_vel['std']:>9.6f} rad/s\n")
            f.write(f"  RMSE: {total_vel['rmse']:>10.6f} rad/s (mean-centered)\n\n")
            
            f.write("Interpretation:\n")
            f.write(self.interpret_shoulder())
            
            # ========== BACK ==========
            f.write("\n" + "=" * 80 + "\n")
            f.write("3. BACK (GRAVITY ORIENTATION) ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            for axis in ['gravity_x', 'gravity_y', 'gravity_z']:
                axis_name = axis.split('_')[-1].upper()
                stats = self.stats['back'][axis]
                f.write(f"{axis_name}-axis Statistics:\n")
                f.write(f"  Mean: {stats['mean']:>10.6f}\n")
                f.write(f"  Std Dev: {stats['std']:>9.6f}\n")
                f.write(f"  RMSE: {stats['rmse']:>10.6f} (mean-centered)\n")
                f.write(f"  Min: {stats['min']:>11.6f}\n")
                f.write(f"  Max: {stats['max']:>11.6f}\n\n")
            
            f.write("Interpretation:\n")
            f.write(self.interpret_back())
            
            # ========== BASE VELOCITY ==========
            f.write("\n" + "=" * 80 + "\n")
            f.write("4. BASE LINEAR VELOCITY ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            for axis in ['linear_velocity_x', 'linear_velocity_y', 'linear_velocity_z']:
                axis_name = axis.split('_')[-1].upper()
                stats = self.stats['base_velocity'][axis]
                f.write(f"{axis_name}-axis Statistics:\n")
                f.write(f"  Mean: {stats['mean']:>10.6f} m/s\n")
                f.write(f"  Std Dev: {stats['std']:>9.6f} m/s\n")
                f.write(f"  RMSE: {stats['rmse']:>10.6f} m/s (mean-centered)\n")
                f.write(f"  Min: {stats['min']:>11.6f} m/s\n")
                f.write(f"  Max: {stats['max']:>11.6f} m/s\n\n")
            
            total_vel = self.stats['base_velocity']['total_linear_velocity']
            f.write("Total Linear Velocity Magnitude:\n")
            f.write(f"  Mean: {total_vel['mean']:>10.6f} m/s\n")
            f.write(f"  Std Dev: {total_vel['std']:>9.6f} m/s\n")
            f.write(f"  RMSE: {total_vel['rmse']:>10.6f} m/s\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        log_message(f"Saved: {output_path}")
    
    def generate_json_report(self, output_path):
        """Generate machine-readable JSON report."""
        log_message("Generating JSON report...")
        
        json_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(self.df),
                'duration_seconds': float(self.df['Time(s)'].iloc[-1]),
                'sampling_rate_hz': float(len(self.df) / self.df['Time(s)'].iloc[-1])
            },
            'statistics': self.stats
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        log_message(f"Saved: {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def save_processed_data(df, output_path):
    """Save processed data to CSV."""
    log_message(f"Saving processed data to: {output_path}")
    df.to_csv(output_path, index=False)


def main():
    """Execute complete pipeline."""
    print("\n" + "=" * 80)
    print("UNITREE G1 ROBOT - COMPREHENSIVE IMU SENSOR ANALYSIS PIPELINE")
    print("=" * 80 + "\n")
    
    # Step 0: Prepare output structure
    base_dir = CONFIG['OUTPUT_BASE']
    run_num = get_next_run_number(base_dir)
    dirs = create_output_directories(base_dir, run_num)
    
    print(f"Run Number: {run_num:03d}")
    print(f"Output Directory: {dirs['run']}\n")
    
    try:
        # Step 1: Load data
        log_message("Step 1: Loading sensor data...")
        loader = SensorDataLoader(CONFIG['DATA_FILE'])
        df = loader.load()
        
        if not loader.validate():
            log_message("Data validation failed!", "WARNING")
        
        # Step 2: Analyze data
        log_message("Step 2: Computing statistics...")
        analyzer = SensorAnalyzer(df)
        stats = analyzer.analyze()
        
        # Step 3: Generate visualizations
        log_message("Step 3: Generating visualizations...")
        visualizer = SensorVisualizer(df, stats)
        visualizer.create_comprehensive_plot(f"{dirs['graphs']}/comprehensive_analysis.png")
        visualizer.create_summary_comparison(f"{dirs['graphs']}/summary_comparison.png")
        
        # Step 4: Generate reports
        log_message("Step 4: Generating reports...")
        reporter = ReportGenerator(df, stats)
        reporter.generate_text_report(f"{dirs['report']}/sensor_analysis_report.txt")
        reporter.generate_json_report(f"{dirs['report']}/analysis_summary.json")
        
        # Step 5: Save processed data
        log_message("Step 5: Saving processed data...")
        save_processed_data(df, f"{dirs['data']}/processed_sensor_data.csv")
        
        # Step 6: Create README
        log_message("Step 6: Creating directory documentation...")
        readme_content = f"""# Run {run_num:03d} - Sensor Analysis Results

## Overview
Complete IMU sensor analysis for Unitree G1 robot upper body sensors.

## Generated Files

### Graphs (`graphs/`)
- **comprehensive_analysis.png**: 9-panel research-quality plot
  - Chest: Position, Velocity, 2D Trajectory
  - Shoulder: X/Y Angular Velocity, 2D Angular Velocity
  - Back: X/Y Gravity, 2D Gravity Orientation

- **summary_comparison.png**: Summary statistics and comparisons
  - RMSE comparison across all sensors
  - Timeline plots for all axes

### Data (`data/`)
- **processed_sensor_data.csv**: Complete processed sensor readings

### Reports (`report/`)
- **sensor_analysis_report.txt**: Comprehensive analysis with interpretations
- **analysis_summary.json**: Machine-readable statistics

## Key Metrics

### RMSE Values (mean-centered)
- Chest Position RMSE: {stats['chest']['position']['rmse']:.6f} rad
- Chest Velocity RMSE: {stats['chest']['velocity']['rmse']:.6f} rad/s
- Shoulder Total Ang Vel RMSE: {stats['shoulder']['total_angular_velocity']['rmse']:.6f} rad/s
- Back Total Gravity RMSE: {stats['back']['total_gravity_magnitude']['rmse']:.6f}

### Data Statistics
- Total Samples: {len(df)}
- Duration: {df['Time(s)'].iloc[-1]:.2f} seconds
- Sampling Rate: {len(df)/df['Time(s)'].iloc[-1]:.2f} Hz

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(f"{dirs['run']}/README.md", 'w') as f:
            f.write(readme_content)
        
        # Final Summary
        print("\n" + "=" * 80)
        print("✅ PIPELINE EXECUTION COMPLETE")
        print("=" * 80)
        print(f"\n📁 Output Directory: {dirs['run']}/")
        print(f"\n📊 Generated Files:")
        print(f"   Graphs:")
        print(f"     • {dirs['graphs']}/comprehensive_analysis.png")
        print(f"     • {dirs['graphs']}/summary_comparison.png")
        print(f"   Data:")
        print(f"     • {dirs['data']}/processed_sensor_data.csv")
        print(f"   Reports:")
        print(f"     • {dirs['report']}/sensor_analysis_report.txt")
        print(f"     • {dirs['report']}/analysis_summary.json")
        print(f"   Documentation:")
        print(f"     • {dirs['run']}/README.md")
        print(f"\n{80*'='}\n")
        
        return True
        
    except Exception as e:
        log_message(f"Pipeline failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
