"""
================================================================================
UNITREE G1 ROBOT - COMPLETE SENSOR DATA PIPELINE
================================================================================

Comprehensive pipeline that for EACH RUN creates:

  run_001/
    ├── data/
    │   ├── raw_sensor_data.csv
    │   └── processed_data.csv
    ├── graphs/
    │   ├── chest_analysis.png
    │   ├── shoulder_analysis.png
    │   ├── back_analysis.png
    │   ├── summary_comparison.png
    │   └── rmse_comparison.png
    ├── report/
    │   └── sensor_analysis_report.txt
    └── README.md

================================================================================
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================================
# UTILITIES
# ============================================================================

def log_message(msg, level="INFO"):
    """Print formatted log message."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def get_next_run_number(base_dir):
    """Get next available run number."""
    if not os.path.exists(base_dir):
        return 1
    
    runs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    if not runs:
        return 1
    
    numbers = [int(d.split('_')[1]) for d in runs if d.split('_')[1].isdigit()]
    return max(numbers) + 1 if numbers else 1


def create_run_directory(base_dir, run_num):
    """Create complete directory structure for a run."""
    run_dir = os.path.join(base_dir, f"run_{run_num:03d}")
    
    # Create subdirectories
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "report"), exist_ok=True)
    
    return run_dir


# ============================================================================
# STEP 1: RUN SIMULATION & COLLECT SENSOR DATA
# ============================================================================

def run_mujoco_simulation(output_data_dir):
    """
    Run MuJoCo simulation and save sensor data to specified directory.
    
    Args:
        output_data_dir (str): Directory to save sensor data CSV
        
    Returns:
        bool: True if successful
    """
    log_message("=" * 80)
    log_message("STEP 1: Running MuJoCo Simulation & Collecting Sensor Data")
    log_message("=" * 80)
    
    try:
        cmd = [
            sys.executable,
            "deploy/deploy_mujoco/deploy_mujoco.py",
            "g1.yaml"
        ]
        
        log_message(f"Starting simulation... (output will be saved to {output_data_dir})")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            log_message(f"Simulation failed with error: {result.stderr}", "ERROR")
            return False
        
        # Move generated CSV to data directory
        if os.path.exists("upper_body_sensor_data.csv"):
            dest_path = os.path.join(output_data_dir, "data", "raw_sensor_data.csv")
            shutil.move("upper_body_sensor_data.csv", dest_path)
            log_message(f"[OK] Sensor data saved: {dest_path}")
            return True
        else:
            log_message("No sensor data file generated", "ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        log_message("Simulation timeout (>120 seconds)", "ERROR")
        return False
    except Exception as e:
        log_message(f"Simulation error: {e}", "ERROR")
        return False


# ============================================================================
# STEP 2: DATA PROCESSING & ANALYSIS (MODULAR FUNCTIONS)
# ============================================================================

def load_data(csv_path):
    """Load raw sensor data from CSV."""
    try:
        df = pd.read_csv(csv_path)
        log_message(f"[OK] Loaded {len(df)} samples from CSV")
        return df
    except Exception as e:
        log_message(f"Failed to load data: {e}", "ERROR")
        return None


def process_data(df):
    """Clean and process raw data."""
    try:
        # Create processed dataframe with computed features
        processed = df.copy()
        
        # Compute 3D magnitudes
        processed['Shoulder_AngVel_Magnitude'] = np.sqrt(
            processed['IMU_AngVel_X']**2 + 
            processed['IMU_AngVel_Y']**2 + 
            processed['IMU_AngVel_Z']**2
        )
        
        processed['Gravity_Magnitude'] = np.sqrt(
            processed['Gravity_X']**2 + 
            processed['Gravity_Y']**2 + 
            processed['Gravity_Z']**2
        )
        
        processed['BaseLinVel_Magnitude'] = np.sqrt(
            processed['BaseLinVel_X']**2 + 
            processed['BaseLinVel_Y']**2 + 
            processed['BaseLinVel_Z']**2
        )
        
        log_message("[OK] Data processing complete")
        return processed
    except Exception as e:
        log_message(f"Failed to process data: {e}", "ERROR")
        return None


def compute_rmse(signal):
    """Compute mean-centered RMSE."""
    mean = np.mean(signal)
    return np.sqrt(np.mean((signal - mean) ** 2))


def analyze_data(df):
    """Compute comprehensive statistics."""
    try:
        stats = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'samples': len(df),
                'duration_s': float(df['Time(s)'].iloc[-1]),
                'sampling_rate_hz': float(len(df) / df['Time(s)'].iloc[-1])
            }
        }
        
        # CHEST ANALYSIS
        stats['chest'] = {
            'position': {
                'mean': float(np.mean(df['Torso_Pos(rad)'])),
                'std': float(np.std(df['Torso_Pos(rad)'])),
                'rmse': float(compute_rmse(df['Torso_Pos(rad)'])),
                'min': float(np.min(df['Torso_Pos(rad)'])),
                'max': float(np.max(df['Torso_Pos(rad)']))
            },
            'velocity': {
                'mean': float(np.mean(df['Torso_Vel(rad/s)'])),
                'std': float(np.std(df['Torso_Vel(rad/s)'])),
                'rmse': float(compute_rmse(df['Torso_Vel(rad/s)'])),
                'min': float(np.min(df['Torso_Vel(rad/s)'])),
                'max': float(np.max(df['Torso_Vel(rad/s)']))
            }
        }
        
        # SHOULDER ANALYSIS
        stats['shoulder'] = {
            'ang_vel_x': {
                'mean': float(np.mean(df['IMU_AngVel_X'])),
                'std': float(np.std(df['IMU_AngVel_X'])),
                'rmse': float(compute_rmse(df['IMU_AngVel_X']))
            },
            'ang_vel_y': {
                'mean': float(np.mean(df['IMU_AngVel_Y'])),
                'std': float(np.std(df['IMU_AngVel_Y'])),
                'rmse': float(compute_rmse(df['IMU_AngVel_Y']))
            },
            'ang_vel_z': {
                'mean': float(np.mean(df['IMU_AngVel_Z'])),
                'std': float(np.std(df['IMU_AngVel_Z'])),
                'rmse': float(compute_rmse(df['IMU_AngVel_Z']))
            },
            'magnitude': {
                'mean': float(np.mean(df['Shoulder_AngVel_Magnitude'])),
                'rmse': float(compute_rmse(df['Shoulder_AngVel_Magnitude']))
            }
        }
        
        # BACK ANALYSIS
        stats['back'] = {
            'gravity_x': {
                'mean': float(np.mean(df['Gravity_X'])),
                'std': float(np.std(df['Gravity_X'])),
                'rmse': float(compute_rmse(df['Gravity_X']))
            },
            'gravity_y': {
                'mean': float(np.mean(df['Gravity_Y'])),
                'std': float(np.std(df['Gravity_Y'])),
                'rmse': float(compute_rmse(df['Gravity_Y']))
            },
            'gravity_z': {
                'mean': float(np.mean(df['Gravity_Z'])),
                'std': float(np.std(df['Gravity_Z'])),
                'rmse': float(compute_rmse(df['Gravity_Z']))
            }
        }
        
        log_message("[OK] Statistical analysis complete")
        return stats
    except Exception as e:
        log_message(f"Failed to analyze data: {e}", "ERROR")
        return None


def generate_plots(df, stats, graphs_dir):
    """Generate all visualization plots."""
    try:
        time = df['Time(s)'].values
        
        # ===== PLOT 1: CHEST ANALYSIS =====
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        ax = axes[0]
        ax.plot(time, df['Torso_Pos(rad)'], 'b-', linewidth=2.5, label='Position')
        ax.fill_between(time, df['Torso_Pos(rad)'], alpha=0.3)
        ax.set_title(f"CHEST: Position (RMSE={stats['chest']['position']['rmse']:.6f} rad)", 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (rad)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(time, df['Torso_Vel(rad/s)'], 'r-', linewidth=2.5, label='Velocity')
        ax.fill_between(time, df['Torso_Vel(rad/s)'], alpha=0.3, color='red')
        ax.set_title(f"CHEST: Velocity (RMSE={stats['chest']['velocity']['rmse']:.6f} rad/s)", 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Velocity (rad/s)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chest_path = os.path.join(graphs_dir, "chest_analysis.png")
        plt.savefig(chest_path, dpi=300, bbox_inches='tight')
        log_message(f"[OK] Saved: {chest_path}")
        plt.close()
        
        # ===== PLOT 2: SHOULDER ANALYSIS =====
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        ax = axes[0]
        ax.plot(time, df['IMU_AngVel_X'], 'purple', linewidth=2.5)
        ax.set_title(f"SHOULDER: X Angular Vel (RMSE={stats['shoulder']['ang_vel_x']['rmse']:.6f})", 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Ang Vel (rad/s)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(time, df['IMU_AngVel_Y'], 'orange', linewidth=2.5)
        ax.set_title(f"SHOULDER: Y Angular Vel (RMSE={stats['shoulder']['ang_vel_y']['rmse']:.6f})", 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Ang Vel (rad/s)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax = axes[2]
        ax.plot(time, df['IMU_AngVel_Z'], 'green', linewidth=2.5)
        ax.set_title(f"SHOULDER: Z Angular Vel (RMSE={stats['shoulder']['ang_vel_z']['rmse']:.6f})", 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Ang Vel (rad/s)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        shoulder_path = os.path.join(graphs_dir, "shoulder_analysis.png")
        plt.savefig(shoulder_path, dpi=300, bbox_inches='tight')
        log_message(f"[OK] Saved: {shoulder_path}")
        plt.close()
        
        # ===== PLOT 3: BACK ANALYSIS =====
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        ax = axes[0]
        ax.plot(time, df['Gravity_X'], 'darkgreen', linewidth=2.5)
        ax.set_title(f"BACK: X Gravity (RMSE={stats['back']['gravity_x']['rmse']:.6f})", 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Gravity Component', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(time, df['Gravity_Y'], 'teal', linewidth=2.5)
        ax.set_title(f"BACK: Y Gravity (RMSE={stats['back']['gravity_y']['rmse']:.6f})", 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Gravity Component', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax = axes[2]
        ax.plot(time, df['Gravity_Z'], 'darkred', linewidth=2.5)
        ax.set_title(f"BACK: Z Gravity (RMSE={stats['back']['gravity_z']['rmse']:.6f})", 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Gravity Component', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        back_path = os.path.join(graphs_dir, "back_analysis.png")
        plt.savefig(back_path, dpi=300, bbox_inches='tight')
        log_message(f"[OK] Saved: {back_path}")
        plt.close()
        
        # ===== PLOT 4: RMSE COMPARISON =====
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rmse_data = {
            'Chest Pos': stats['chest']['position']['rmse'],
            'Chest Vel': stats['chest']['velocity']['rmse'],
            'Shoulder X': stats['shoulder']['ang_vel_x']['rmse'],
            'Shoulder Y': stats['shoulder']['ang_vel_y']['rmse'],
            'Shoulder Z': stats['shoulder']['ang_vel_z']['rmse'],
            'Back X': stats['back']['gravity_x']['rmse'],
            'Back Y': stats['back']['gravity_y']['rmse'],
            'Back Z': stats['back']['gravity_z']['rmse'],
        }
        colors = ['#FF6B6B']*2 + ['#4ECDC4']*3 + ['#95A5A6']*3
        
        ax.barh(list(rmse_data.keys()), list(rmse_data.values()), color=colors)
        ax.set_xlabel('RMSE Value', fontsize=12)
        ax.set_title('RMSE Comparison Across All Sensors', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        rmse_path = os.path.join(graphs_dir, "rmse_comparison.png")
        plt.savefig(rmse_path, dpi=300, bbox_inches='tight')
        log_message(f"[OK] Saved: {rmse_path}")
        plt.close()
        
        # ===== PLOT 5: SUMMARY COMPARISON =====
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # STD DEV comparison
        ax = axes[0, 0]
        std_data = {
            'Chest Pos': stats['chest']['position']['std'],
            'Chest Vel': stats['chest']['velocity']['std'],
            'Shoulder X': stats['shoulder']['ang_vel_x']['std'],
            'Shoulder Y': stats['shoulder']['ang_vel_y']['std'],
            'Shoulder Z': stats['shoulder']['ang_vel_z']['std'],
        }
        ax.bar(range(len(std_data)), list(std_data.values()), color='steelblue')
        ax.set_xticks(range(len(std_data)))
        ax.set_xticklabels(list(std_data.keys()), rotation=45, ha='right')
        ax.set_ylabel('Std Dev', fontsize=11)
        ax.set_title('Standard Deviation Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Chest timeline
        ax = axes[0, 1]
        ax.plot(time, df['Torso_Pos(rad)'], 'r-', linewidth=2, label='Position', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(time, df['Torso_Vel(rad/s)'], 'b-', linewidth=2, label='Velocity', alpha=0.7)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (rad)', color='r', fontsize=11)
        ax2.set_ylabel('Velocity (rad/s)', color='b', fontsize=11)
        ax.set_title('CHEST: Position & Velocity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Shoulder 3-axis
        ax = axes[1, 0]
        ax.plot(time, df['IMU_AngVel_X'], label='X', linewidth=2)
        ax.plot(time, df['IMU_AngVel_Y'], label='Y', linewidth=2)
        ax.plot(time, df['IMU_AngVel_Z'], label='Z', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
        ax.set_title('SHOULDER: 3-Axis Angular Velocity', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Back 3-axis
        ax = axes[1, 1]
        ax.plot(time, df['Gravity_X'], label='X', linewidth=2)
        ax.plot(time, df['Gravity_Y'], label='Y', linewidth=2)
        ax.plot(time, df['Gravity_Z'], label='Z', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Gravity Component', fontsize=11)
        ax.set_title('BACK: 3-Axis Gravity Orientation', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_path = os.path.join(graphs_dir, "summary_comparison.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        log_message(f"[OK] Saved: {summary_path}")
        plt.close()
        
        log_message("[OK] All plots generated successfully")
        return True
        
    except Exception as e:
        log_message(f"Failed to generate plots: {e}", "ERROR")
        return False


def generate_report(df, stats, report_dir):
    """Generate text report with analysis and observations."""
    try:
        report_path = os.path.join(report_dir, "sensor_analysis_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("UNITREE G1 ROBOT - SENSOR ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples: {len(df)}\n")
            f.write(f"Duration: {df['Time(s)'].iloc[-1]:.2f} seconds\n")
            f.write(f"Sampling Rate: {len(df)/df['Time(s)'].iloc[-1]:.2f} Hz\n\n")
            
            # CHEST ANALYSIS
            f.write("=" * 80 + "\n")
            f.write("CHEST ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Position:\n")
            f.write(f"  Mean:    {stats['chest']['position']['mean']:.6f} rad\n")
            f.write(f"  Std Dev: {stats['chest']['position']['std']:.6f} rad\n")
            f.write(f"  RMSE:    {stats['chest']['position']['rmse']:.6f} rad (mean-centered)\n")
            f.write(f"  Min:     {stats['chest']['position']['min']:.6f} rad\n")
            f.write(f"  Max:     {stats['chest']['position']['max']:.6f} rad\n\n")
            
            f.write("Velocity:\n")
            f.write(f"  Mean:    {stats['chest']['velocity']['mean']:.6f} rad/s\n")
            f.write(f"  Std Dev: {stats['chest']['velocity']['std']:.6f} rad/s\n")
            f.write(f"  RMSE:    {stats['chest']['velocity']['rmse']:.6f} rad/s (mean-centered)\n")
            f.write(f"  Min:     {stats['chest']['velocity']['min']:.6f} rad/s\n")
            f.write(f"  Max:     {stats['chest']['velocity']['max']:.6f} rad/s\n\n")
            
            chest_rmse = stats['chest']['position']['rmse']
            if chest_rmse < 0.01:
                chest_status = "VERY STABLE"
            elif chest_rmse < 0.05:
                chest_status = "STABLE"
            else:
                chest_status = "DYNAMIC"
            f.write(f"Observation: Chest motion is {chest_status}\n\n")
            
            # SHOULDER ANALYSIS
            f.write("=" * 80 + "\n")
            f.write("SHOULDER ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("X-axis Angular Velocity:\n")
            f.write(f"  Mean: {stats['shoulder']['ang_vel_x']['mean']:.6f} rad/s\n")
            f.write(f"  Std:  {stats['shoulder']['ang_vel_x']['std']:.6f} rad/s\n")
            f.write(f"  RMSE: {stats['shoulder']['ang_vel_x']['rmse']:.6f} rad/s\n\n")
            
            f.write("Y-axis Angular Velocity:\n")
            f.write(f"  Mean: {stats['shoulder']['ang_vel_y']['mean']:.6f} rad/s\n")
            f.write(f"  Std:  {stats['shoulder']['ang_vel_y']['std']:.6f} rad/s\n")
            f.write(f"  RMSE: {stats['shoulder']['ang_vel_y']['rmse']:.6f} rad/s\n\n")
            
            f.write("Z-axis Angular Velocity:\n")
            f.write(f"  Mean: {stats['shoulder']['ang_vel_z']['mean']:.6f} rad/s\n")
            f.write(f"  Std:  {stats['shoulder']['ang_vel_z']['std']:.6f} rad/s\n")
            f.write(f"  RMSE: {stats['shoulder']['ang_vel_z']['rmse']:.6f} rad/s\n\n")
            
            f.write("Total Magnitude:\n")
            f.write(f"  Mean: {stats['shoulder']['magnitude']['mean']:.6f} rad/s\n")
            f.write(f"  RMSE: {stats['shoulder']['magnitude']['rmse']:.6f} rad/s\n\n")
            
            rmse_vals = [
                stats['shoulder']['ang_vel_x']['rmse'],
                stats['shoulder']['ang_vel_y']['rmse'],
                stats['shoulder']['ang_vel_z']['rmse']
            ]
            dominant_idx = np.argmax(rmse_vals)
            dominant_axis = ['X', 'Y', 'Z'][dominant_idx]
            f.write(f"Observation: Dominant rotation axis is {dominant_axis}\n\n")
            
            # BACK ANALYSIS
            f.write("=" * 80 + "\n")
            f.write("BACK ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("X-axis Gravity:\n")
            f.write(f"  Mean: {stats['back']['gravity_x']['mean']:.6f}\n")
            f.write(f"  Std:  {stats['back']['gravity_x']['std']:.6f}\n")
            f.write(f"  RMSE: {stats['back']['gravity_x']['rmse']:.6f}\n\n")
            
            f.write("Y-axis Gravity:\n")
            f.write(f"  Mean: {stats['back']['gravity_y']['mean']:.6f}\n")
            f.write(f"  Std:  {stats['back']['gravity_y']['std']:.6f}\n")
            f.write(f"  RMSE: {stats['back']['gravity_y']['rmse']:.6f}\n\n")
            
            f.write("Z-axis Gravity:\n")
            f.write(f"  Mean: {stats['back']['gravity_z']['mean']:.6f}\n")
            f.write(f"  Std:  {stats['back']['gravity_z']['std']:.6f}\n")
            f.write(f"  RMSE: {stats['back']['gravity_z']['rmse']:.6f}\n\n")
            
            gz_rmse = stats['back']['gravity_z']['rmse']
            if gz_rmse < 0.001:
                back_status = "UPRIGHT - gravity correctly oriented on Z-axis"
            elif gz_rmse < 0.01:
                back_status = "MOSTLY UPRIGHT - minor tilt detected"
            else:
                back_status = "SIGNIFICANT TILT - check orientation"
            f.write(f"Observation: {back_status}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        log_message(f"[OK] Report generated: {report_path}")
        return True
        
    except Exception as e:
        log_message(f"Failed to generate report: {e}", "ERROR")
        return False


def save_processed_data(processed_df, data_dir):
    """Save processed data to CSV."""
    try:
        csv_path = os.path.join(data_dir, "processed_data.csv")
        processed_df.to_csv(csv_path, index=False)
        log_message(f"[OK] Processed data saved: {csv_path}")
        return True
    except Exception as e:
        log_message(f"Failed to save processed data: {e}", "ERROR")
        return False


def save_statistics_json(stats, report_dir):
    """Save statistics as JSON."""
    try:
        json_path = os.path.join(report_dir, "statistics.json")
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=4)
        log_message(f"[OK] Statistics saved: {json_path}")
        return True
    except Exception as e:
        log_message(f"Failed to save JSON: {e}", "ERROR")
        return False


def create_run_readme(run_dir, run_num, stats):
    """Create README for the run."""
    try:
        readme_path = os.path.join(run_dir, "README.md")
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# Run {run_num:03d} - Complete Sensor Analysis\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Contents\n\n")
            f.write("### data/\n")
            f.write("- `raw_sensor_data.csv` - Original sensor readings\n")
            f.write("- `processed_data.csv` - Cleaned and computed data\n\n")
            f.write("### graphs/\n")
            f.write("- `chest_analysis.png` - Chest position and velocity\n")
            f.write("- `shoulder_analysis.png` - Shoulder angular velocities\n")
            f.write("- `back_analysis.png` - Back gravity orientation\n")
            f.write("- `rmse_comparison.png` - RMSE across all sensors\n")
            f.write("- `summary_comparison.png` - Comprehensive summary plots\n\n")
            f.write("### report/\n")
            f.write("- `sensor_analysis_report.txt` - Detailed analysis with observations\n")
            f.write("- `statistics.json` - Machine-readable statistics\n\n")
            f.write("## Key Metrics\n\n")
            f.write(f"- Samples: {stats['metadata']['samples']}\n")
            f.write(f"- Duration: {stats['metadata']['duration_s']:.2f}s\n")
            f.write(f"- Sampling Rate: {stats['metadata']['sampling_rate_hz']:.2f} Hz\n")
        
        return True
    except Exception as e:
        log_message(f"Failed to create README: {e}", "ERROR")
        return False


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Execute complete integrated pipeline."""
    print("\n" + "=" * 80)
    print("UNITREE G1 ROBOT - COMPLETE SENSOR DATA PIPELINE")
    print("=" * 80 + "\n")
    
    base_dir = "collected_sensor_data"
    run_num = get_next_run_number(base_dir)
    run_dir = create_run_directory(base_dir, run_num)
    
    log_message(f"Run Number: {run_num:03d}")
    log_message(f"Run Directory: {run_dir}\n")
    
    try:
        # STEP 1: Collect sensor data
        if not run_mujoco_simulation(run_dir):
            log_message("Simulation failed!", "ERROR")
            return False
        
        # STEP 2: Load data
        log_message("")
        log_message("=" * 80)
        log_message("STEP 2: Loading & Processing Data")
        log_message("=" * 80)
        
        raw_csv = os.path.join(run_dir, "data", "raw_sensor_data.csv")
        df = load_data(raw_csv)
        if df is None:
            return False
        
        # STEP 3: Process data
        processed_df = process_data(df)
        if processed_df is None:
            return False
        
        # STEP 4: Analyze data
        log_message("")
        log_message("=" * 80)
        log_message("STEP 3: Statistical Analysis")
        log_message("=" * 80)
        
        stats = analyze_data(processed_df)
        if stats is None:
            return False
        
        # STEP 5: Generate plots
        log_message("")
        log_message("=" * 80)
        log_message("STEP 4: Generating Visualizations")
        log_message("=" * 80)
        
        if not generate_plots(processed_df, stats, os.path.join(run_dir, "graphs")):
            return False
        
        # STEP 6: Generate report
        log_message("")
        log_message("=" * 80)
        log_message("STEP 5: Creating Report")
        log_message("=" * 80)
        
        if not generate_report(df, stats, os.path.join(run_dir, "report")):
            return False
        
        # STEP 7: Save processed data and statistics
        if not save_processed_data(processed_df, os.path.join(run_dir, "data")):
            return False
        
        if not save_statistics_json(stats, os.path.join(run_dir, "report")):
            return False
        
        # STEP 8: Create README
        create_run_readme(run_dir, run_num, stats)
        
        # FINAL SUMMARY
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 80)
        print(f"\nRun {run_num:03d} Results:\n")
        print(f"Location: {run_dir}\n")
        print(f"Contents:")
        print(f"  data/")
        print(f"    - raw_sensor_data.csv")
        print(f"    - processed_data.csv")
        print(f"  graphs/")
        print(f"    - chest_analysis.png")
        print(f"    - shoulder_analysis.png")
        print(f"    - back_analysis.png")
        print(f"    - rmse_comparison.png")
        print(f"    - summary_comparison.png")
        print(f"  report/")
        print(f"    - sensor_analysis_report.txt")
        print(f"    - statistics.json")
        print(f"  README.md\n")
        print(f"Next Run: {run_num+1:03d} will create: {base_dir}/run_{run_num+1:03d}/\n")
        print("=" * 80 + "\n")
        
        return True
        
    except Exception as e:
        log_message(f"Pipeline error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
