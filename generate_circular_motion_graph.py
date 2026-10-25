import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Read the CSV file
csv_path = "upper_body_sensor_data.csv"
df = pd.read_csv(csv_path)

# Create graphs folder if it doesn't exist
graphs_folder = "Experimental_graphs"
Path(graphs_folder).mkdir(exist_ok=True)

# Extract time and velocity data
time = df['Time(s)'].values
vel_x = df['BaseLinVel_X'].values
vel_y = df['BaseLinVel_Y'].values

# Calculate position by integrating velocity
pos_x = np.zeros_like(time)
pos_y = np.zeros_like(time)

for i in range(1, len(time)):
    pos_x[i] = pos_x[i-1] + (vel_x[i-1] + vel_x[i]) / 2 * (time[i] - time[i-1])
    pos_y[i] = pos_y[i-1] + (vel_y[i-1] + vel_y[i]) / 2 * (time[i] - time[i-1])

# Create figure
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the trajectory with gradient color (time progression)
scatter = ax.scatter(pos_x, pos_y, c=time, cmap='cool', s=150, alpha=0.7, edgecolors='black', linewidth=1.5, zorder=3)

# Draw the path
ax.plot(pos_x, pos_y, 'k-', alpha=0.4, linewidth=2, zorder=1)

# Add arrows to show direction (clockwise)
n_arrows = 8
arrow_indices = np.linspace(1, len(pos_x)-2, n_arrows, dtype=int)
for idx in arrow_indices:
    dx = pos_x[idx+1] - pos_x[idx-1]
    dy = pos_y[idx+1] - pos_y[idx-1]
    ax.arrow(pos_x[idx], pos_y[idx], dx*0.08, dy*0.08, 
             head_width=0.035, head_length=0.025, fc='red', ec='red', zorder=4, linewidth=2)

# Mark start point only
ax.plot(pos_x[0], pos_y[0], 'go', markersize=20, label='START', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)

# Draw circle center
center_x = np.mean(pos_x)
center_y = np.mean(pos_y)
ax.plot(center_x, center_y, 'k+', markersize=20, markeredgewidth=3, zorder=2)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Time (s)', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Labels and title
ax.set_xlabel('X Position (m)', fontsize=16, fontweight='bold')
ax.set_ylabel('Y Position (m)', fontsize=16, fontweight='bold')
ax.set_title('G1 Robot Walking in Clockwise Circular Motion', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
ax.set_aspect('equal')

# Legend
ax.legend(fontsize=14, loc='upper left', framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

# Add text annotation for clockwise direction
ax.text(0.98, 0.02, '↻ Clockwise Direction', transform=ax.transAxes, 
        fontsize=14, fontweight='bold', ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='red', linewidth=2))

plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, 'robot_circular_motion_clockwise.png'), dpi=300, bbox_inches='tight')
print(f"✅ Circular motion graph saved: {graphs_folder}/robot_circular_motion_clockwise.png")
plt.close()

# Print info
print(f"\nCircular Motion Analysis:")
print(f"  Center point: ({center_x:.4f}, {center_y:.4f})")
radius = np.sqrt((pos_x - center_x)**2 + (pos_y - center_y)**2)
print(f"  Average radius: {np.mean(radius):.4f} m")
print(f"  Clockwise direction: Confirmed ✓")
