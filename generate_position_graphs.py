import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Read the CSV file
csv_path = "upper_body_sensor_data.csv"
df = pd.read_csv(csv_path)

# Create graphs folder if it doesn't exist
graphs_folder = "graphs_position_vs_time"
Path(graphs_folder).mkdir(exist_ok=True)

print(f"Reading data from: {csv_path}")
print(f"Creating graphs in folder: {graphs_folder}")

# Extract time and velocity data
time = df['Time(s)'].values
vel_x = df['BaseLinVel_X'].values
vel_y = df['BaseLinVel_Y'].values
vel_z = df['BaseLinVel_Z'].values

# Calculate position by integrating velocity
# Using cumulative trapezoidal integration
dt = np.diff(time)
dt = np.append(dt, dt[-1])  # Add last dt for consistency

# Trapezoidal integration for smoother position calculation
pos_x = np.zeros_like(time)
pos_y = np.zeros_like(time)
pos_z = np.zeros_like(time)

for i in range(1, len(time)):
    pos_x[i] = pos_x[i-1] + (vel_x[i-1] + vel_x[i]) / 2 * (time[i] - time[i-1])
    pos_y[i] = pos_y[i-1] + (vel_y[i-1] + vel_y[i]) / 2 * (time[i] - time[i-1])
    pos_z[i] = pos_z[i-1] + (vel_z[i-1] + vel_z[i]) / 2 * (time[i] - time[i-1])

# Create figure with multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Robot Position vs Time (Clockwise Circular Motion)', fontsize=16, fontweight='bold')

# Position X vs Time
axes[0].plot(time, pos_x, 'b-', linewidth=2, label='X Position')
axes[0].set_ylabel('X Position (m)', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)
axes[0].set_title('Forward/Backward Position vs Time', fontsize=12)

# Position Y vs Time
axes[1].plot(time, pos_y, 'g-', linewidth=2, label='Y Position')
axes[1].set_ylabel('Y Position (m)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)
axes[1].set_title('Lateral Position vs Time', fontsize=12)

# Position Z vs Time
axes[2].plot(time, pos_z, 'r-', linewidth=2, label='Z Position')
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Z Position (m)', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=10)
axes[2].set_title('Vertical Position vs Time', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, '01_position_vs_time_all.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 01_position_vs_time_all.png")
plt.close()

# Create 2D trajectory plot (XY plane - circular motion view)
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(pos_x, pos_y, c=time, cmap='viridis', s=50, alpha=0.6)
ax.plot(pos_x, pos_y, 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('X Position (m)', fontsize=12)
ax.set_ylabel('Y Position (m)', fontsize=12)
ax.set_title('Robot XY Trajectory (Clockwise Circular Motion)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Time (s)', fontsize=11)

# Add start and end markers
ax.plot(pos_x[0], pos_y[0], 'go', markersize=12, label='Start', zorder=5)
ax.plot(pos_x[-1], pos_y[-1], 'r^', markersize=12, label='End', zorder=5)
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, '02_trajectory_xy_plane.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 02_trajectory_xy_plane.png")
plt.close()

# Create velocity vs time plots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Robot Velocity vs Time (Clockwise Circular Motion)', fontsize=16, fontweight='bold')

axes[0].plot(time, vel_x, 'b-', linewidth=2)
axes[0].set_ylabel('X Velocity (m/s)', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Forward/Backward Velocity vs Time', fontsize=12)

axes[1].plot(time, vel_y, 'g-', linewidth=2)
axes[1].set_ylabel('Y Velocity (m/s)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Lateral Velocity vs Time', fontsize=12)

axes[2].plot(time, vel_z, 'r-', linewidth=2)
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Z Velocity (m/s)', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].set_title('Vertical Velocity vs Time', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, '03_velocity_vs_time_all.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 03_velocity_vs_time_all.png")
plt.close()

# Create combined position and velocity plot
fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:blue'
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('X Position (m)', color=color, fontsize=12)
line1 = ax1.plot(time, pos_x, color=color, linewidth=2, label='X Position')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('X Velocity (m/s)', color=color, fontsize=12)
line2 = ax2.plot(time, vel_x, color=color, linewidth=2, linestyle='--', label='X Velocity')
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle('Forward Motion: Position and Velocity vs Time', fontsize=14, fontweight='bold')
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, '04_position_velocity_x_combined.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 04_position_velocity_x_combined.png")
plt.close()

# Create 3D trajectory plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pos_x, pos_y, pos_z, c=time, cmap='plasma', s=30, alpha=0.6)
ax.plot(pos_x, pos_y, pos_z, 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('X Position (m)', fontsize=11)
ax.set_ylabel('Y Position (m)', fontsize=11)
ax.set_zlabel('Z Position (m)', fontsize=11)
ax.set_title('Robot 3D Trajectory (Clockwise Circular Motion)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Time (s)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, '05_trajectory_3d.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 05_trajectory_3d.png")
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("MOTION ANALYSIS SUMMARY")
print("="*60)
print(f"Total simulation time: {time[-1]:.2f} seconds")
print(f"Number of data points: {len(time)}")
print(f"\nPosition Range:")
print(f"  X: {pos_x.min():.4f} to {pos_x.max():.4f} m (range: {pos_x.max() - pos_x.min():.4f} m)")
print(f"  Y: {pos_y.min():.4f} to {pos_y.max():.4f} m (range: {pos_y.max() - pos_y.min():.4f} m)")
print(f"  Z: {pos_z.min():.4f} to {pos_z.max():.4f} m (range: {pos_z.max() - pos_z.min():.4f} m)")
print(f"\nAverage Velocity:")
print(f"  X: {vel_x.mean():.4f} m/s")
print(f"  Y: {vel_y.mean():.4f} m/s")
print(f"  Z: {vel_z.mean():.4f} m/s")
print(f"\nVelocity Range:")
print(f"  X: {vel_x.min():.4f} to {vel_x.max():.4f} m/s")
print(f"  Y: {vel_y.min():.4f} to {vel_y.max():.4f} m/s")
print(f"  Z: {vel_z.min():.4f} to {vel_z.max():.4f} m/s")

# Calculate total distance traveled
distance_traveled = np.sum(np.sqrt(np.diff(pos_x)**2 + np.diff(pos_y)**2))
print(f"\nTotal distance traveled: {distance_traveled:.4f} m")

# Estimate circle radius (for circular motion)
center_x = np.mean(pos_x)
center_y = np.mean(pos_y)
radius_values = np.sqrt((pos_x - center_x)**2 + (pos_y - center_y)**2)
avg_radius = np.mean(radius_values)
print(f"Estimated circular motion parameters:")
print(f"  Center: ({center_x:.4f}, {center_y:.4f})")
print(f"  Average radius: {avg_radius:.4f} m")
print("="*60)

print(f"\n✅ All graphs successfully saved to '{graphs_folder}' folder!")
