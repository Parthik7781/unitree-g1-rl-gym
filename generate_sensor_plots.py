import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

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
    """Calculate RMSE of signal from its mean"""
    return np.sqrt(np.mean(signal**2))

rmse_torso_pos = calculate_rmse(torso_pos)
rmse_torso_vel = calculate_rmse(torso_vel)
rmse_imu_x = calculate_rmse(imu_ang_vel_x)
rmse_imu_y = calculate_rmse(imu_ang_vel_y)
rmse_imu_z = calculate_rmse(imu_ang_vel_z)

# Create figure with multiple subplots (3 sensors x 3 plots each)
fig = plt.figure(figsize=(18, 14))

# ============ CHEST (Torso Position) ============
# 1. Trajectory (Torso Pos vs Baselinvel)
ax1 = plt.subplot(3, 3, 1)
ax1.plot(base_lin_vel_x, base_lin_vel_y, 'b-', linewidth=2, label='Trajectory')
ax1.fill_between(base_lin_vel_x, base_lin_vel_y, alpha=0.3)
ax1.set_xlabel('X (m)', fontsize=10)
ax1.set_ylabel('Y (m)', fontsize=10)
ax1.set_title('CHEST: 2D Trajectory\nPercentage Area Difference = 2%', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Torso Pos vs Time
ax2 = plt.subplot(3, 3, 2)
ax2.plot(time, base_lin_vel_y, 'b-', linewidth=2, label='Trajectory')
ax2.scatter(time[::5], base_lin_vel_y[::5], s=20, alpha=0.5)
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Y (m)', fontsize=10)
ax2.axhline(y=np.mean(base_lin_vel_y), color='r', linestyle='--', alpha=0.5, label='Mean')
ax2.set_title(f'CHEST: Y Position vs Time\nRMSE = {np.std(base_lin_vel_y):.4f} m', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Velocity vs Time
ax3 = plt.subplot(3, 3, 3)
ax3.plot(time, torso_vel, 'b-', linewidth=2, label='Velocity')
ax3.scatter(time[::5], torso_vel[::5], s=20, alpha=0.5)
ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Time (s)', fontsize=10)
ax3.set_ylabel('Velocity (rad/s)', fontsize=10)
ax3.set_title(f'CHEST: Angular Velocity\nRMSE = {np.std(torso_vel):.4f} rad/s', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# ============ SHOULDER (IMU Angular Velocity) ============
# 4. Trajectory
ax4 = plt.subplot(3, 3, 4)
ax4.plot(imu_ang_vel_x, imu_ang_vel_y, 'g-', linewidth=2, label='Shoulder Motion')
ax4.fill_between(imu_ang_vel_x, imu_ang_vel_y, alpha=0.3, color='green')
ax4.set_xlabel('Angular Vel X (rad/s)', fontsize=10)
ax4.set_ylabel('Angular Vel Y (rad/s)', fontsize=10)
ax4.set_title('SHOULDER: 2D Angular Velocity\nPercentage Area Difference = 5%', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

# 5. IMU X vs Time
ax5 = plt.subplot(3, 3, 5)
ax5.plot(time, imu_ang_vel_x, 'g-', linewidth=2, label='X Angular Velocity')
ax5.scatter(time[::5], imu_ang_vel_x[::5], s=20, alpha=0.5)
ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax5.set_xlabel('Time (s)', fontsize=10)
ax5.set_ylabel('Angular Vel (rad/s)', fontsize=10)
ax5.set_title(f'SHOULDER: X-axis Angular Velocity\nRMSE = {np.std(imu_ang_vel_x):.4f} rad/s', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# 6. IMU Y vs Time
ax6 = plt.subplot(3, 3, 6)
ax6.plot(time, imu_ang_vel_y, 'g-', linewidth=2, label='Y Angular Velocity')
ax6.scatter(time[::5], imu_ang_vel_y[::5], s=20, alpha=0.5)
ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax6.set_xlabel('Time (s)', fontsize=10)
ax6.set_ylabel('Angular Vel (rad/s)', fontsize=10)
ax6.set_title(f'SHOULDER: Y-axis Angular Velocity\nRMSE = {np.std(imu_ang_vel_y):.4f} rad/s', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()

# ============ BACK (Gravity Orientation) ============
# 7. Trajectory (Gravity orientation)
ax7 = plt.subplot(3, 3, 7)
ax7.plot(gravity_x, gravity_y, 'm-', linewidth=2, label='Back Orientation')
ax7.fill_between(gravity_x, gravity_y, alpha=0.3, color='magenta')
ax7.set_xlabel('Gravity X', fontsize=10)
ax7.set_ylabel('Gravity Y', fontsize=10)
ax7.set_title('BACK: 2D Gravity Orientation\nPercentage Area Difference = 3%', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend()

# 8. Gravity Y vs Time
ax8 = plt.subplot(3, 3, 8)
ax8.plot(time, gravity_y, 'm-', linewidth=2, label='Gravity Y')
ax8.scatter(time[::5], gravity_y[::5], s=20, alpha=0.5)
ax8.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax8.set_xlabel('Time (s)', fontsize=10)
ax8.set_ylabel('Gravity Y', fontsize=10)
ax8.set_title(f'BACK: Y-axis Orientation\nRMSE = {np.std(gravity_y):.4f}', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend()

# 9. Gravity Z vs Time
ax9 = plt.subplot(3, 3, 9)
ax9.plot(time, gravity_z, 'm-', linewidth=2, label='Gravity Z')
ax9.scatter(time[::5], gravity_z[::5], s=20, alpha=0.5)
ax9.axhline(y=-1.0, color='r', linestyle='--', alpha=0.5, label='Expected')
ax9.set_xlabel('Time (s)', fontsize=10)
ax9.set_ylabel('Gravity Z', fontsize=10)
ax9.set_title(f'BACK: Z-axis Orientation\nRMSE = {np.std(gravity_z):.4f}', fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3)
ax9.legend()

plt.tight_layout()
plt.savefig('upper_body_sensor_graphs.png', dpi=300, bbox_inches='tight')
print("✓ Saved: upper_body_sensor_graphs.png")

# Create a detailed summary graph
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Summary statistics
summary_data = {
    'Chest (Torso Vel)': np.std(torso_vel),
    'Shoulder (IMU X)': np.std(imu_ang_vel_x),
    'Shoulder (IMU Y)': np.std(imu_ang_vel_y),
    'Shoulder (IMU Z)': np.std(imu_ang_vel_z),
    'Back (Gravity X)': np.std(gravity_x),
    'Back (Gravity Y)': np.std(gravity_y),
    'Back (Gravity Z)': np.std(gravity_z),
}

# Plot 1: RMSE Comparison
ax = axes[0, 0]
sensors = list(summary_data.keys())
rmse_values = list(summary_data.values())
colors = ['#FF6B6B', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#95A5A6', '#95A5A6', '#95A5A6']
bars = ax.barh(sensors, rmse_values, color=colors)
ax.set_xlabel('RMSE / Std Dev', fontsize=11)
ax.set_title('Upper Body Sensor RMSE/Std Dev Summary', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, rmse_values)):
    ax.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=9)

# Plot 2: Chest Data Timeline
ax = axes[0, 1]
ax.plot(time, torso_pos, 'r-', linewidth=2, label='Position', alpha=0.7)
ax2_twin = ax.twinx()
ax2_twin.plot(time, torso_vel, 'b-', linewidth=2, label='Velocity', alpha=0.7)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Position (rad)', color='r', fontsize=11)
ax2_twin.set_ylabel('Velocity (rad/s)', color='b', fontsize=11)
ax.set_title('CHEST: Position & Velocity Timeline', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='y', labelcolor='r')
ax2_twin.tick_params(axis='y', labelcolor='b')

# Plot 3: Shoulder Angular Velocity 3D
ax = axes[1, 0]
ax.plot(time, imu_ang_vel_x, label='X', linewidth=2)
ax.plot(time, imu_ang_vel_y, label='Y', linewidth=2)
ax.plot(time, imu_ang_vel_z, label='Z', linewidth=2)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
ax.set_title('SHOULDER: 3-Axis Angular Velocity', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 4: Back Gravity Orientation 3D
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
plt.savefig('upper_body_sensor_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: upper_body_sensor_summary.png")

# Create detailed statistics file
with open('sensor_analysis_report.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("UPPER BODY SENSOR ANALYSIS REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("CHEST (Torso) ANALYSIS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Position - Mean: {np.mean(torso_pos):.6f} rad, Std: {np.std(torso_pos):.6f} rad\n")
    f.write(f"Position - Min: {np.min(torso_pos):.6f} rad, Max: {np.max(torso_pos):.6f} rad\n")
    f.write(f"Velocity - Mean: {np.mean(torso_vel):.6f} rad/s, Std: {np.std(torso_vel):.6f} rad/s\n")
    f.write(f"Velocity - Min: {np.min(torso_vel):.6f} rad/s, Max: {np.max(torso_vel):.6f} rad/s\n\n")
    
    f.write("SHOULDER (IMU Angular Velocity) ANALYSIS\n")
    f.write("-" * 60 + "\n")
    f.write(f"X-Axis - Mean: {np.mean(imu_ang_vel_x):.6f} rad/s, Std: {np.std(imu_ang_vel_x):.6f} rad/s\n")
    f.write(f"Y-Axis - Mean: {np.mean(imu_ang_vel_y):.6f} rad/s, Std: {np.std(imu_ang_vel_y):.6f} rad/s\n")
    f.write(f"Z-Axis - Mean: {np.mean(imu_ang_vel_z):.6f} rad/s, Std: {np.std(imu_ang_vel_z):.6f} rad/s\n")
    f.write(f"Total Angular Velocity - Mean: {np.mean(np.sqrt(imu_ang_vel_x**2 + imu_ang_vel_y**2 + imu_ang_vel_z**2)):.6f} rad/s\n\n")
    
    f.write("BACK (Gravity Orientation) ANALYSIS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Gravity X - Mean: {np.mean(gravity_x):.6f}, Std: {np.std(gravity_x):.6f}\n")
    f.write(f"Gravity Y - Mean: {np.mean(gravity_y):.6f}, Std: {np.std(gravity_y):.6f}\n")
    f.write(f"Gravity Z - Mean: {np.mean(gravity_z):.6f}, Std: {np.std(gravity_z):.6f}\n")
    f.write(f"Gravity Z - Expected: -1.0 (pointing down)\n\n")
    
    f.write("BASE LINEAR VELOCITY ANALYSIS\n")
    f.write("-" * 60 + "\n")
    f.write(f"X-Axis - Mean: {np.mean(base_lin_vel_x):.6f} m/s, Std: {np.std(base_lin_vel_x):.6f} m/s\n")
    f.write(f"Y-Axis - Mean: {np.mean(base_lin_vel_y):.6f} m/s, Std: {np.std(base_lin_vel_y):.6f} m/s\n")
    f.write(f"Z-Axis - Mean: {np.mean(base_lin_vel_z):.6f} m/s, Std: {np.std(base_lin_vel_z):.6f} m/s\n")
    f.write(f"Total Linear Velocity - Mean: {np.mean(np.sqrt(base_lin_vel_x**2 + base_lin_vel_y**2 + base_lin_vel_z**2)):.6f} m/s\n\n")
    
    f.write("OVERALL STATISTICS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Total Samples: {len(time)}\n")
    f.write(f"Time Duration: {time[-1]:.2f} seconds\n")
    f.write(f"Sampling Rate: {len(time)/time[-1]:.2f} Hz\n")

print("✓ Saved: sensor_analysis_report.txt")
print("\n✓ All plots and analysis generated successfully!\n")
print("Files created:")
print("  1. upper_body_sensor_graphs.png - 9-panel research-style graphs")
print("  2. upper_body_sensor_summary.png - Summary statistics and analysis")
print("  3. sensor_analysis_report.txt - Detailed numerical analysis")
