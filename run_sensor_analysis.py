"""
Automated Sensor Data Collection and Analysis Pipeline
This script orchestrates the entire workflow:
1. Runs the MuJoCo simulation
2. Collects sensor data
3. Generates graphs and analysis
4. Organizes results in timestamped folders
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
import json

def create_results_directory():
    """Create a timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/run_{timestamp}"
    
    # Create directory structure
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/data", exist_ok=True)
    os.makedirs(f"{results_dir}/graphs", exist_ok=True)
    os.makedirs(f"{results_dir}/analysis", exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📁 Created Results Directory: {results_dir}")
    print(f"{'='*70}\n")
    
    return results_dir, timestamp

def run_simulation():
    """Run the MuJoCo simulation"""
    print(f"\n{'='*70}")
    print("▶️  STEP 1: Running MuJoCo Simulation...")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable,
        "deploy/deploy_mujoco/deploy_mujoco.py",
        "g1.yaml"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running simulation: {e}")
        print(e.stderr)
        return False

def generate_analysis(results_dir):
    """Generate graphs and analysis from sensor data"""
    print(f"\n{'='*70}")
    print("▶️  STEP 2: Generating Graphs and Analysis...")
    print(f"{'='*70}\n")
    
    # Create the analysis script inline
    analysis_script = f"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import seaborn as sns
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")

# Load sensor data
df = pd.read_csv('upper_body_sensor_data.csv')

# Extract data
time = df['Time(s)'].values
torso_pos = df['Torso_Pos(rad)'].values
torso_vel = df['Torso_Vel(rad/s)'].values
imu_ang_vel_x = df['IMU_AngVel_X'].values
imu_ang_vel_y = df['IMU_AngVel_Y'].values
imu_ang_vel_z = df['IMU_AngVel_Z'].values
gravity_x = df['Gravity_X'].values
gravity_y = df['Gravity_Y'].values
gravity_z = df['Gravity_Z'].values
base_lin_vel_x = df['BaseLinVel_X'].values
base_lin_vel_y = df['BaseLinVel_Y'].values
base_lin_vel_z = df['BaseLinVel_Z'].values

# Calculate RMSE for each sensor
def calculate_rmse(signal):
    return np.sqrt(np.mean(signal**2))

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 14))

# CHEST (Torso Position)
ax1 = plt.subplot(3, 3, 1)
ax1.plot(base_lin_vel_x, base_lin_vel_y, 'b-', linewidth=2)
ax1.fill_between(base_lin_vel_x, base_lin_vel_y, alpha=0.3)
ax1.set_xlabel('X (m)', fontsize=10)
ax1.set_ylabel('Y (m)', fontsize=10)
area_diff = abs(np.trapz(base_lin_vel_y, base_lin_vel_x)) / max(1, max(base_lin_vel_x))
ax1.set_title(f'CHEST: 2D Trajectory\\nArea Difference = {{area_diff:.1f}}%', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
ax2.plot(time, base_lin_vel_y, 'b-', linewidth=2)
ax2.scatter(time[::5], base_lin_vel_y[::5], s=20, alpha=0.5)
rmse_y = np.sqrt(np.mean(base_lin_vel_y**2))
ax2.set_title(f'CHEST: Y Position vs Time\\nRMSE = {{rmse_y:.4f}} m', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Y (m)', fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 3, 3)
ax3.plot(time, torso_vel, 'b-', linewidth=2)
ax3.scatter(time[::5], torso_vel[::5], s=20, alpha=0.5)
rmse_vel = np.sqrt(np.mean(torso_vel**2))
ax3.set_title(f'CHEST: Angular Velocity\\nRMSE = {{rmse_vel:.4f}} rad/s', fontsize=11, fontweight='bold')
ax3.set_xlabel('Time (s)', fontsize=10)
ax3.set_ylabel('Velocity (rad/s)', fontsize=10)
ax3.grid(True, alpha=0.3)

# SHOULDER (IMU Angular Velocity)
ax4 = plt.subplot(3, 3, 4)
ax4.plot(imu_ang_vel_x, imu_ang_vel_y, 'g-', linewidth=2)
ax4.fill_between(imu_ang_vel_x, imu_ang_vel_y, alpha=0.3, color='green')
ax4.set_xlabel('Angular Vel X (rad/s)', fontsize=10)
ax4.set_ylabel('Angular Vel Y (rad/s)', fontsize=10)
area_diff_shoulder = abs(np.trapz(imu_ang_vel_y, imu_ang_vel_x)) / max(1, max(imu_ang_vel_x))
ax4.set_title(f'SHOULDER: 2D Angular Velocity\\nArea Difference = {{area_diff_shoulder:.1f}}%', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(3, 3, 5)
ax5.plot(time, imu_ang_vel_x, 'g-', linewidth=2)
ax5.scatter(time[::5], imu_ang_vel_x[::5], s=20, alpha=0.5)
rmse_x = np.sqrt(np.mean(imu_ang_vel_x**2))
ax5.set_title(f'SHOULDER: X-axis Angular Velocity\\nRMSE = {{rmse_x:.4f}} rad/s', fontsize=11, fontweight='bold')
ax5.set_xlabel('Time (s)', fontsize=10)
ax5.set_ylabel('Angular Vel (rad/s)', fontsize=10)
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 3, 6)
ax6.plot(time, imu_ang_vel_y, 'g-', linewidth=2)
ax6.scatter(time[::5], imu_ang_vel_y[::5], s=20, alpha=0.5)
rmse_y_ang = np.sqrt(np.mean(imu_ang_vel_y**2))
ax6.set_title(f'SHOULDER: Y-axis Angular Velocity\\nRMSE = {{rmse_y_ang:.4f}} rad/s', fontsize=11, fontweight='bold')
ax6.set_xlabel('Time (s)', fontsize=10)
ax6.set_ylabel('Angular Vel (rad/s)', fontsize=10)
ax6.grid(True, alpha=0.3)

# BACK (Gravity Orientation)
ax7 = plt.subplot(3, 3, 7)
ax7.plot(gravity_x, gravity_y, 'm-', linewidth=2)
ax7.fill_between(gravity_x, gravity_y, alpha=0.3, color='magenta')
ax7.set_xlabel('Gravity X', fontsize=10)
ax7.set_ylabel('Gravity Y', fontsize=10)
area_diff_back = abs(np.trapz(gravity_y, gravity_x)) / max(1, max(gravity_x))
ax7.set_title(f'BACK: 2D Gravity Orientation\\nArea Difference = {{area_diff_back:.1f}}%', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

ax8 = plt.subplot(3, 3, 8)
ax8.plot(time, gravity_y, 'm-', linewidth=2)
ax8.scatter(time[::5], gravity_y[::5], s=20, alpha=0.5)
rmse_grav_y = np.sqrt(np.mean(gravity_y**2))
ax8.set_title(f'BACK: Y-axis Orientation\\nRMSE = {{rmse_grav_y:.4f}}', fontsize=11, fontweight='bold')
ax8.set_xlabel('Time (s)', fontsize=10)
ax8.set_ylabel('Gravity Y', fontsize=10)
ax8.grid(True, alpha=0.3)

ax9 = plt.subplot(3, 3, 9)
ax9.plot(time, gravity_z, 'm-', linewidth=2)
ax9.scatter(time[::5], gravity_z[::5], s=20, alpha=0.5)
rmse_grav_z = np.sqrt(np.mean(gravity_z**2))
ax9.set_title(f'BACK: Z-axis Orientation\\nRMSE = {{rmse_grav_z:.4f}}', fontsize=11, fontweight='bold')
ax9.set_xlabel('Time (s)', fontsize=10)
ax9.set_ylabel('Gravity Z', fontsize=10)
ax9.grid(True, alpha=0.3)

plt.tight_layout()
graph_path = '{results_dir}/graphs/sensor_analysis_9panel.png'
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {{graph_path}}")
plt.close()

# Summary figure
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

summary_data = {{
    'Chest (Torso Vel)': np.std(torso_vel),
    'Shoulder (IMU X)': np.std(imu_ang_vel_x),
    'Shoulder (IMU Y)': np.std(imu_ang_vel_y),
    'Shoulder (IMU Z)': np.std(imu_ang_vel_z),
    'Back (Gravity X)': np.std(gravity_x),
    'Back (Gravity Y)': np.std(gravity_y),
    'Back (Gravity Z)': np.std(gravity_z),
}}

ax = axes[0, 0]
sensors = list(summary_data.keys())
rmse_values = list(summary_data.values())
colors = ['#FF6B6B', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#95A5A6', '#95A5A6', '#95A5A6']
bars = ax.barh(sensors, rmse_values, color=colors)
ax.set_xlabel('RMSE / Std Dev', fontsize=11)
ax.set_title('Upper Body Sensor RMSE Summary', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

ax = axes[0, 1]
ax.plot(time, torso_pos, 'r-', linewidth=2, label='Position', alpha=0.7)
ax2_twin = ax.twinx()
ax2_twin.plot(time, torso_vel, 'b-', linewidth=2, label='Velocity', alpha=0.7)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Position (rad)', color='r', fontsize=11)
ax2_twin.set_ylabel('Velocity (rad/s)', color='b', fontsize=11)
ax.set_title('CHEST: Position & Velocity', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(time, imu_ang_vel_x, label='X', linewidth=2)
ax.plot(time, imu_ang_vel_y, label='Y', linewidth=2)
ax.plot(time, imu_ang_vel_z, label='Z', linewidth=2)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
ax.set_title('SHOULDER: 3-Axis Angular Velocity', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(time, gravity_x, label='X', linewidth=2)
ax.plot(time, gravity_y, label='Y', linewidth=2)
ax.plot(time, gravity_z, label='Z', linewidth=2)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Gravity Component', fontsize=11)
ax.set_title('BACK: 3-Axis Gravity Orientation', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
summary_path = '{results_dir}/graphs/sensor_summary.png'
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {{summary_path}}")
plt.close()

# Generate detailed report
report_path = '{results_dir}/analysis/sensor_report.txt'
with open(report_path, 'w') as f:
    f.write("="*70 + "\\n")
    f.write("UPPER BODY SENSOR ANALYSIS REPORT\\n")
    f.write("="*70 + "\\n")
    f.write(f"Generated: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\\n")
    f.write(f"Total Samples: {{len(time)}}\\n")
    f.write(f"Time Duration: {{time[-1]:.2f}} seconds\\n")
    f.write(f"Sampling Rate: {{len(time)/max(time[-1], 1):.2f}} Hz\\n\\n")
    
    f.write("CHEST (Torso) ANALYSIS\\n")
    f.write("-"*70 + "\\n")
    f.write(f"Position - Mean: {{np.mean(torso_pos):.6f}} rad, Std: {{np.std(torso_pos):.6f}} rad\\n")
    f.write(f"Velocity - Mean: {{np.mean(torso_vel):.6f}} rad/s, Std: {{np.std(torso_vel):.6f}} rad/s\\n")
    f.write(f"Velocity - RMSE: {{np.sqrt(np.mean(torso_vel**2)):.6f}} rad/s\\n\\n")
    
    f.write("SHOULDER (IMU Angular Velocity) ANALYSIS\\n")
    f.write("-"*70 + "\\n")
    f.write(f"X-Axis - Mean: {{np.mean(imu_ang_vel_x):.6f}} rad/s, Std: {{np.std(imu_ang_vel_x):.6f}} rad/s\\n")
    f.write(f"Y-Axis - Mean: {{np.mean(imu_ang_vel_y):.6f}} rad/s, Std: {{np.std(imu_ang_vel_y):.6f}} rad/s\\n")
    f.write(f"Z-Axis - Mean: {{np.mean(imu_ang_vel_z):.6f}} rad/s, Std: {{np.std(imu_ang_vel_z):.6f}} rad/s\\n")
    f.write(f"Total Angular Velocity: {{np.mean(np.sqrt(imu_ang_vel_x**2 + imu_ang_vel_y**2 + imu_ang_vel_z**2)):.6f}} rad/s\\n\\n")
    
    f.write("BACK (Gravity Orientation) ANALYSIS\\n")
    f.write("-"*70 + "\\n")
    f.write(f"Gravity X - Mean: {{np.mean(gravity_x):.6f}}, Std: {{np.std(gravity_x):.6f}}\\n")
    f.write(f"Gravity Y - Mean: {{np.mean(gravity_y):.6f}}, Std: {{np.std(gravity_y):.6f}}\\n")
    f.write(f"Gravity Z - Mean: {{np.mean(gravity_z):.6f}}, Std: {{np.std(gravity_z):.6f}}\\n\\n")
    
    f.write("BASE LINEAR VELOCITY ANALYSIS\\n")
    f.write("-"*70 + "\\n")
    f.write(f"X-Axis - Mean: {{np.mean(base_lin_vel_x):.6f}} m/s, Std: {{np.std(base_lin_vel_x):.6f}} m/s\\n")
    f.write(f"Y-Axis - Mean: {{np.mean(base_lin_vel_y):.6f}} m/s, Std: {{np.std(base_lin_vel_y):.6f}} m/s\\n")
    f.write(f"Z-Axis - Mean: {{np.mean(base_lin_vel_z):.6f}} m/s, Std: {{np.std(base_lin_vel_z):.6f}} m/s\\n")

print(f"✓ Saved: {{report_path}}")

# Generate JSON summary
json_path = '{results_dir}/analysis/summary.json'
summary_json = {{
    "timestamp": "{{datetime.now().isoformat()}}",
    "total_samples": len(time),
    "duration_seconds": float(time[-1]),
    "sampling_rate_hz": float(len(time)/max(time[-1], 1)),
    "chest": {{
        "torso_pos_mean": float(np.mean(torso_pos)),
        "torso_pos_std": float(np.std(torso_pos)),
        "torso_vel_mean": float(np.mean(torso_vel)),
        "torso_vel_std": float(np.std(torso_vel)),
        "torso_vel_rmse": float(np.sqrt(np.mean(torso_vel**2)))
    }},
    "shoulder": {{
        "imu_x_mean": float(np.mean(imu_ang_vel_x)),
        "imu_x_std": float(np.std(imu_ang_vel_x)),
        "imu_y_mean": float(np.mean(imu_ang_vel_y)),
        "imu_y_std": float(np.std(imu_ang_vel_y)),
        "imu_z_mean": float(np.mean(imu_ang_vel_z)),
        "imu_z_std": float(np.std(imu_ang_vel_z))
    }},
    "back": {{
        "gravity_x_mean": float(np.mean(gravity_x)),
        "gravity_x_std": float(np.std(gravity_x)),
        "gravity_y_mean": float(np.mean(gravity_y)),
        "gravity_y_std": float(np.std(gravity_y)),
        "gravity_z_mean": float(np.mean(gravity_z)),
        "gravity_z_std": float(np.std(gravity_z))
    }}
}}

with open(json_path, 'w') as f:
    json.dump(summary_json, f, indent=4)

print(f"✓ Saved: {{json_path}}")
print("\\n✓ Analysis complete!")
"""
    
    # Write and run analysis script
    with open("_temp_analysis.py", "w") as f:
        f.write(analysis_script)
    
    try:
        result = subprocess.run(
            [sys.executable, "_temp_analysis.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        # Clean up temp script
        os.remove("_temp_analysis.py")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating analysis: {e}")
        print(e.stderr)
        return False

def organize_results(results_dir):
    """Move generated files to results directory"""
    print(f"\n{'='*70}")
    print("▶️  STEP 3: Organizing Results...")
    print(f"{'='*70}\n")
    
    files_to_move = {
        'upper_body_sensor_data.csv': f'{results_dir}/data/',
    }
    
    for src, dest in files_to_move.items():
        if os.path.exists(src):
            shutil.move(src, dest)
            print(f"✓ Moved: {src} → {dest}")
        else:
            print(f"⚠️  File not found: {src}")

def create_summary_markdown(results_dir, timestamp):
    """Create a summary README for the results"""
    readme_content = f"""# Sensor Analysis Results - {timestamp}

## Overview
This folder contains the complete sensor data analysis and visualization for the G1 robot upper body sensors.

## Directory Structure
```
├── data/
│   └── upper_body_sensor_data.csv      # Raw sensor readings (Step, Time, Position, Velocity, etc.)
├── graphs/
│   ├── sensor_analysis_9panel.png      # 9-panel research-style graph (Chest, Shoulder, Back)
│   └── sensor_summary.png              # Summary statistics and RMSE comparison
└── analysis/
    ├── sensor_report.txt               # Detailed statistical analysis
    └── summary.json                    # JSON summary of key metrics
```

## What's Included

### 1. Data Files
- **upper_body_sensor_data.csv**: Raw sensor readings collected during simulation
  - Columns: Step, Time, Torso Position, Torso Velocity
  - IMU Angular Velocities (X, Y, Z)
  - Gravity Orientation (X, Y, Z)
  - Base Linear Velocities (X, Y, Z)

### 2. Graphs

#### sensor_analysis_9panel.png
A research-style 9-panel figure showing:
- **CHEST (Row 1)**:
  - 2D Trajectory plot
  - Y-Position vs Time
  - Angular Velocity vs Time
  
- **SHOULDER (Row 2)**:
  - 2D Angular Velocity trajectory
  - X-axis Angular Velocity timeline
  - Y-axis Angular Velocity timeline
  
- **BACK (Row 3)**:
  - 2D Gravity Orientation plot
  - Y-axis Orientation timeline
  - Z-axis Orientation timeline

Each panel includes RMSE or statistical error measurements.

#### sensor_summary.png
A 4-panel summary showing:
- RMSE comparison bar chart
- Chest position & velocity timeline
- Shoulder 3-axis angular velocity
- Back 3-axis gravity orientation

### 3. Analysis Files
- **sensor_report.txt**: Comprehensive statistics including mean, std dev, min, max for all sensors
- **summary.json**: Machine-readable summary of key metrics

## Key Metrics

### CHEST (Torso Sensors)
- Measures body core rotation and movement
- Includes flexion/extension movements

### SHOULDER (IMU Angular Velocity)
- Captures 3-axis rotations around shoulders
- Indicates arm swing and upper body twist

### BACK (Gravity Orientation)
- Tracks back posture relative to gravity
- Indicates forward/backward lean and lateral tilt

## Generated Timestamp
"{timestamp}"

## How to Use
1. Review the PNG graphs for visual analysis
2. Check sensor_report.txt for statistical details
3. Load summary.json for data integration into other systems
4. Use upper_body_sensor_data.csv for custom analysis

---
*Generated by Automated Sensor Analysis Pipeline*
"""
    
    with open(f'{results_dir}/README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Created: {results_dir}/README.md")

def main():
    print("\n" + "="*70)
    print("🤖 UNITREE G1 ROBOT - UPPER BODY SENSOR ANALYSIS PIPELINE")
    print("="*70)
    
    # Create results directory
    results_dir, timestamp = create_results_directory()
    
    # Run simulation
    if not run_simulation():
        print("❌ Simulation failed!")
        return False
    
    # Generate analysis
    if not generate_analysis(results_dir):
        print("❌ Analysis generation failed!")
        return False
    
    # Organize results
    organize_results(results_dir)
    
    # Create summary
    create_summary_markdown(results_dir, timestamp)
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 Results saved in: {results_dir}/")
    print(f"\nGenerated Files:")
    print(f"  📈 Graphs:")
    print(f"     • {results_dir}/graphs/sensor_analysis_9panel.png")
    print(f"     • {results_dir}/graphs/sensor_summary.png")
    print(f"  📊 Analysis:")
    print(f"     • {results_dir}/analysis/sensor_report.txt")
    print(f"     • {results_dir}/analysis/summary.json")
    print(f"  📁 Data:")
    print(f"     • {results_dir}/data/upper_body_sensor_data.csv")
    print(f"  📖 Documentation:")
    print(f"     • {results_dir}/README.md")
    print(f"\n{'='*70}\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
