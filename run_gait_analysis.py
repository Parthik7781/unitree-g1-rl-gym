import time
import sys
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


def run_experiment(arm_posture):
    """Run a single gait analysis experiment"""
    print(f"\n{'='*70}")
    print(f"Starting experiment: {arm_posture}")
    print(f"{'='*70}")
    
    # Use the 12-DOF model config
    config_file = "g1.yaml"
    
    cmd = [
        sys.executable,
        "deploy/deploy_mujoco/deploy_gait_analysis.py",
        config_file,
        "--arm_posture", arm_posture
    ]
    
    try:
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Warning: Experiment {arm_posture} returned code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
        return True
    except subprocess.TimeoutExpired:
        print(f"Error: Experiment {arm_posture} timed out")
        return False
    except Exception as e:
        print(f"Error running experiment {arm_posture}: {e}")
        return False


def compute_metrics(trajectory_df, left_imu_df, right_imu_df):
    """Compute motion metrics from collected data"""
    metrics = {}
    
    # Trajectory metrics
    if not trajectory_df.empty:
        x = trajectory_df['X_Pos(m)'].values
        y = trajectory_df['Y_Pos(m)'].values
        
        # Path length
        dx = np.diff(x)
        dy = np.diff(y)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        metrics['Path_Length'] = path_length
        
        # Average speed
        time_vals = trajectory_df['Time(s)'].values
        duration = time_vals[-1] - time_vals[0]
        metrics['Avg_Speed'] = path_length / duration if duration > 0 else 0
    else:
        metrics['Path_Length'] = 0
        metrics['Avg_Speed'] = 0
    
    # Hand acceleration metrics
    for label, imu_df in [('Left', left_imu_df), ('Right', right_imu_df)]:
        if not imu_df.empty:
            ax = imu_df['ax'].values
            ay = imu_df['ay'].values
            az = imu_df['az'].values
            acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
            
            gx = imu_df['gx'].values
            gy = imu_df['gy'].values
            gz = imu_df['gz'].values
            ang_vel_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            
            metrics[f'{label}_RMS_Accel'] = np.sqrt(np.mean(acc_mag**2))
            metrics[f'{label}_Peak_Accel'] = np.max(acc_mag)
            metrics[f'{label}_RMS_AngVel'] = np.sqrt(np.mean(ang_vel_mag**2))
            metrics[f'{label}_Peak_AngVel'] = np.max(ang_vel_mag)
        else:
            metrics[f'{label}_RMS_Accel'] = 0
            metrics[f'{label}_Peak_Accel'] = 0
            metrics[f'{label}_RMS_AngVel'] = 0
            metrics[f'{label}_Peak_AngVel'] = 0
    
    return metrics


def generate_summary_figure(arm_posture, trajectory_df, left_imu_df, right_imu_df, metrics):
    """Generate summary figure with 4 subplots"""
    
    output_dir = Path("Experimental_graphs") / arm_posture
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure title mapping
    title_map = {
        "P1_Natural_Walk": "P1: Natural Walk - Gait Summary",
        "P2_Arms_Down": "P2: Arms Down - Gait Summary",
        "P3_Arms_Forward": "P3: Arms Forward - Gait Summary",
        "P4_Arms_Sideways": "P4: Arms Sideways - Gait Summary",
        "P5_Arms_Up": "P5: Arms Up - Gait Summary",
    }
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Subplot 1: XY Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    if not trajectory_df.empty:
        x = trajectory_df['X_Pos(m)'].values
        y = trajectory_df['Y_Pos(m)'].values
        ax1.plot(x, y, 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(x[0], y[0], color='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(x[-1], y[-1], color='red', s=100, marker='^', label='End', zorder=5)
        ax1.set_xlabel('X Position (m)', fontsize=11)
        ax1.set_ylabel('Y Position (m)', fontsize=11)
        ax1.set_title('XY Trajectory', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')
    
    # Subplot 2: Hand Acceleration Magnitude
    ax2 = fig.add_subplot(gs[0, 1])
    if not left_imu_df.empty:
        time_left = left_imu_df['Time(s)'].values
        ax_l = left_imu_df['ax'].values
        ay_l = left_imu_df['ay'].values
        az_l = left_imu_df['az'].values
        acc_mag_left = np.sqrt(ax_l**2 + ay_l**2 + az_l**2)
        ax2.plot(time_left, acc_mag_left, 'b-', linewidth=2, label='Left Hand', alpha=0.7)
    
    if not right_imu_df.empty:
        time_right = right_imu_df['Time(s)'].values
        ax_r = right_imu_df['ax'].values
        ay_r = right_imu_df['ay'].values
        az_r = right_imu_df['az'].values
        acc_mag_right = np.sqrt(ax_r**2 + ay_r**2 + az_r**2)
        ax2.plot(time_right, acc_mag_right, 'r-', linewidth=2, label='Right Hand', alpha=0.7)
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Acceleration Magnitude (m/s²)', fontsize=11)
    ax2.set_title('Hand Acceleration Magnitude', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Hand Angular Velocity Magnitude
    ax3 = fig.add_subplot(gs[1, 0])
    if not left_imu_df.empty:
        time_left = left_imu_df['Time(s)'].values
        gx_l = left_imu_df['gx'].values
        gy_l = left_imu_df['gy'].values
        gz_l = left_imu_df['gz'].values
        ang_vel_mag_left = np.sqrt(gx_l**2 + gy_l**2 + gz_l**2)
        ax3.plot(time_left, ang_vel_mag_left, 'b-', linewidth=2, label='Left Hand', alpha=0.7)
    
    if not right_imu_df.empty:
        time_right = right_imu_df['Time(s)'].values
        gx_r = right_imu_df['gx'].values
        gy_r = right_imu_df['gy'].values
        gz_r = right_imu_df['gz'].values
        ang_vel_mag_right = np.sqrt(gx_r**2 + gy_r**2 + gz_r**2)
        ax3.plot(time_right, ang_vel_mag_right, 'r-', linewidth=2, label='Right Hand', alpha=0.7)
    
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Angular Velocity Magnitude (rad/s)', fontsize=11)
    ax3.set_title('Hand Angular Velocity Magnitude', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Subplot 4: Metrics Summary Panel
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create text summary
    summary_text = f"""
    MOTION METRICS SUMMARY
    
    Trajectory:
    • Path Length: {metrics.get('Path_Length', 0):.4f} m
    • Average Speed: {metrics.get('Avg_Speed', 0):.4f} m/s
    
    Left Hand:
    • RMS Acceleration: {metrics.get('Left_RMS_Accel', 0):.4f} m/s²
    • Peak Acceleration: {metrics.get('Left_Peak_Accel', 0):.4f} m/s²
    • RMS Ang. Velocity: {metrics.get('Left_RMS_AngVel', 0):.4f} rad/s
    • Peak Ang. Velocity: {metrics.get('Left_Peak_AngVel', 0):.4f} rad/s
    
    Right Hand:
    • RMS Acceleration: {metrics.get('Right_RMS_Accel', 0):.4f} m/s²
    • Peak Acceleration: {metrics.get('Right_Peak_Accel', 0):.4f} m/s²
    • RMS Ang. Velocity: {metrics.get('Right_RMS_AngVel', 0):.4f} rad/s
    • Peak Ang. Velocity: {metrics.get('Right_Peak_AngVel', 0):.4f} rad/s
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title_map.get(arm_posture, f"{arm_posture} - Gait Summary"),
                 fontsize=16, fontweight='bold', y=0.995)
    
    fig.savefig(output_dir / "gait_summary.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary figure: {output_dir}/gait_summary.png")
    plt.close()
    
    return metrics


def main():
    """Main orchestration function"""
    print(f"\n{'='*70}")
    print("UNITREE G1 GAIT ANALYSIS - COMPARATIVE ARM POSTURE STUDY")
    print(f"{'='*70}")
    
    arm_postures = [
        "P1_Natural_Walk",
        "P2_Arms_Down",
        "P3_Arms_Forward",
        "P4_Arms_Sideways",
        "P5_Arms_Up"
    ]
    
    results = {}
    all_metrics = {}
    
    # Run all experiments
    for posture in arm_postures:
        # Run experiment
        if not run_experiment(posture):
            print(f"⚠ Skipping {posture} due to execution error")
            continue
        
        # Load collected data
        output_dir = Path("Experimental_graphs") / posture
        
        try:
            trajectory_df = pd.read_csv(output_dir / "trajectory.csv")
        except FileNotFoundError:
            print(f"⚠ Trajectory data not found for {posture}")
            trajectory_df = pd.DataFrame()
        
        try:
            left_imu_df = pd.read_csv(output_dir / "left_hand_imu.csv")
        except FileNotFoundError:
            print(f"⚠ Left hand IMU data not found for {posture}")
            left_imu_df = pd.DataFrame()
        
        try:
            right_imu_df = pd.read_csv(output_dir / "right_hand_imu.csv")
        except FileNotFoundError:
            print(f"⚠ Right hand IMU data not found for {posture}")
            right_imu_df = pd.DataFrame()
        
        # Compute metrics
        metrics = compute_metrics(trajectory_df, left_imu_df, right_imu_df)
        all_metrics[posture] = metrics
        
        # Generate summary figure
        generate_summary_figure(posture, trajectory_df, left_imu_df, right_imu_df, metrics)
        
        results[posture] = {
            'trajectory': trajectory_df,
            'left_imu': left_imu_df,
            'right_imu': right_imu_df,
            'metrics': metrics
        }
    
    # Generate comparison figure
    generate_comparison_figure(results, all_metrics)
    
    # Generate analysis summary
    generate_analysis_summary(all_metrics)
    
    print(f"\n{'='*70}")
    print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")


def generate_comparison_figure(results, all_metrics):
    """Generate comparison figure with all 5 experiments"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    postures = list(results.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(postures)))
    
    # Panel 1: Combined XY Trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    for posture, color in zip(postures, colors):
        traj_df = results[posture]['trajectory']
        if not traj_df.empty:
            x = traj_df['X_Pos(m)'].values
            y = traj_df['Y_Pos(m)'].values
            ax1.plot(x, y, color=color, linewidth=2, label=posture, alpha=0.7)
    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title('Combined XY Trajectories', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_aspect('equal')
    
    # Panel 2: Acceleration Magnitude Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    for posture, color in zip(postures, colors):
        left_imu = results[posture]['left_imu']
        if not left_imu.empty:
            time = left_imu['Time(s)'].values
            ax = left_imu['ax'].values
            ay = left_imu['ay'].values
            az = left_imu['az'].values
            acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
            ax2.plot(time, acc_mag, color=color, linewidth=1.5, label=posture, alpha=0.7)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Acceleration Magnitude (m/s²)', fontsize=11)
    ax2.set_title('Average Hand Acceleration Magnitude', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Panel 3: Angular Velocity Magnitude Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    for posture, color in zip(postures, colors):
        left_imu = results[posture]['left_imu']
        if not left_imu.empty:
            time = left_imu['Time(s)'].values
            gx = left_imu['gx'].values
            gy = left_imu['gy'].values
            gz = left_imu['gz'].values
            ang_vel_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            ax3.plot(time, ang_vel_mag, color=color, linewidth=1.5, label=posture, alpha=0.7)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Angular Velocity Magnitude (rad/s)', fontsize=11)
    ax3.set_title('Average Hand Angular Velocity Magnitude', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Panel 4: Comparison Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create comparison table
    table_data = []
    table_data.append(['Posture', 'Path (m)', 'Speed (m/s)', 'L_RMS_Acc', 'L_Peak_Acc', 'L_RMS_AngV', 'L_Peak_AngV'])
    
    for posture in postures:
        metrics = all_metrics[posture]
        table_data.append([
            posture,
            f"{metrics.get('Path_Length', 0):.3f}",
            f"{metrics.get('Avg_Speed', 0):.3f}",
            f"{metrics.get('Left_RMS_Accel', 0):.3f}",
            f"{metrics.get('Left_Peak_Accel', 0):.3f}",
            f"{metrics.get('Left_RMS_AngVel', 0):.3f}",
            f"{metrics.get('Left_Peak_AngVel', 0):.3f}",
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.15, 0.12, 0.12, 0.13, 0.13, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    fig.suptitle('Arm Posture Comparison Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    fig.savefig(Path("Experimental_graphs") / "Arm_Posture_Comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison figure: Experimental_graphs/Arm_Posture_Comparison.png")
    plt.close()


def generate_analysis_summary(all_metrics):
    """Generate concise analysis summary"""
    
    summary_file = Path("Experimental_graphs") / "GAIT_ANALYSIS_SUMMARY.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("UNITREE G1 GAIT ANALYSIS - ARM POSTURE COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Find extremes
        max_path_posture = max(all_metrics, key=lambda x: all_metrics[x]['Path_Length'])
        max_speed_posture = max(all_metrics, key=lambda x: all_metrics[x]['Avg_Speed'])
        max_accel_posture = max(all_metrics, key=lambda x: all_metrics[x]['Left_RMS_Accel'])
        max_ang_vel_posture = max(all_metrics, key=lambda x: all_metrics[x]['Left_RMS_AngVel'])
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("1. MOST STABLE MOTION:\n")
        min_accel_posture = min(all_metrics, key=lambda x: all_metrics[x]['Left_RMS_Accel'])
        f.write(f"   {min_accel_posture}\n")
        f.write(f"   RMS Hand Acceleration: {all_metrics[min_accel_posture]['Left_RMS_Accel']:.4f} m/s²\n\n")
        
        f.write("2. HIGHEST HAND MOTION:\n")
        f.write(f"   {max_accel_posture}\n")
        f.write(f"   Peak Hand Acceleration: {all_metrics[max_accel_posture]['Left_Peak_Accel']:.4f} m/s²\n\n")
        
        f.write("3. LARGEST TRAJECTORY DEVIATION:\n")
        f.write(f"   {max_path_posture}\n")
        f.write(f"   Path Length: {all_metrics[max_path_posture]['Path_Length']:.4f} m\n\n")
        
        f.write("4. HIGHEST HAND ACCELERATION:\n")
        f.write(f"   {max_accel_posture}\n")
        f.write(f"   Peak Acceleration: {all_metrics[max_accel_posture]['Left_Peak_Accel']:.4f} m/s²\n\n")
        
        f.write("5. HIGHEST HAND ANGULAR VELOCITY:\n")
        f.write(f"   {max_ang_vel_posture}\n")
        f.write(f"   Peak Angular Velocity: {all_metrics[max_ang_vel_posture]['Left_Peak_AngVel']:.4f} rad/s\n\n")
        
        f.write("DETAILED METRICS TABLE:\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"{'Posture':<20} {'Path(m)':<12} {'Speed(m/s)':<12} {'RMS_Acc':<12} {'Peak_Acc':<12}\n")
        f.write("-" * 70 + "\n")
        
        for posture in sorted(all_metrics.keys()):
            metrics = all_metrics[posture]
            f.write(f"{posture:<20} {metrics['Path_Length']:<12.4f} {metrics['Avg_Speed']:<12.4f} "
                   f"{metrics['Left_RMS_Accel']:<12.4f} {metrics['Left_Peak_Accel']:<12.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Saved analysis summary: {summary_file}")


if __name__ == "__main__":
    main()
