"""
Human vs G1 Robot Walking - Comprehensive IMU & Gait Comparison
================================================================
Compares human motion capture (AMC) data with Unitree G1 robot
IMU data under normal walking conditions and generates publication-quality
comparison plots.

Usage:
    cd c:\\unitree_rl_gym-main\\unitree_rl_gym-main
    python compare_human_robot_walking.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#30363d',
    'text.color': '#c9d1d9',
    'legend.framealpha': 0.3,
    'legend.facecolor': '#21262d',
    'legend.edgecolor': '#30363d',
})

HUMAN_COLOR   = '#58a6ff'   # Blue
ROBOT_COLOR   = '#3fb950'   # Green
ACCENT_COLOR  = '#f78166'   # Coral/Red

# ─── AMC Parser ───────────────────────────────────────────────────────────────
def parse_amc(filepath):
    """
    Parse an AMC (Acclaim Motion Capture) file.
    Returns a dict: frame_id -> {bone_name -> [values]}
    AMC captures at 120 fps.
    """
    frames = {}
    current_frame = None
    current_data = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith(':'):
                continue
            parts = line.split()
            # Frame number line (integer only)
            if len(parts) == 1:
                try:
                    fnum = int(parts[0])
                    if current_frame is not None:
                        frames[current_frame] = current_data
                    current_frame = fnum
                    current_data = {}
                    continue
                except ValueError:
                    pass
            # Bone data line
            if current_frame is not None and len(parts) >= 2:
                bone = parts[0]
                try:
                    vals = [float(v) for v in parts[1:]]
                    current_data[bone] = vals
                except ValueError:
                    pass
    if current_frame is not None:
        frames[current_frame] = current_data

    return frames


def amc_to_dataframe(frames, fps=120.0):
    """
    Convert AMC frames to a pandas DataFrame with derived IMU-like signals.
    Key signals extracted:
      - trunk_pitch  : lowerback Y rotation (forward/backward lean)
      - trunk_roll   : lowerback X rotation (side lean)
      - trunk_yaw    : lowerback Z rotation
      - rknee_angle  : rtibia (right knee flexion)
      - lknee_angle  : ltibia (left knee flexion)
      - rhip_pitch   : rfemur Y (right hip flex/extension)
      - lhip_pitch   : lfemur Y
      - root_x, root_y, root_z : pelvis position
    """
    records = []
    for fnum in sorted(frames.keys()):
        d = frames[fnum]
        t = fnum / fps

        def get(bone, idx=0, default=0.0):
            vals = d.get(bone, [default])
            return vals[idx] if idx < len(vals) else default

        # Root position & orientation (cols: tx, ty, tz, rx, ry, rz)
        root = d.get('root', [0]*6)
        root_x  = root[0] if len(root) > 0 else 0.0
        root_y  = root[1] if len(root) > 1 else 0.0
        root_z  = root[2] if len(root) > 2 else 0.0
        root_rx = root[3] if len(root) > 3 else 0.0  # trunk pitch
        root_ry = root[4] if len(root) > 4 else 0.0  # trunk lean
        root_rz = root[5] if len(root) > 5 else 0.0  # trunk yaw

        lb = d.get('lowerback', [0]*3)
        trunk_pitch = lb[0] if len(lb) > 0 else 0.0
        trunk_roll  = lb[1] if len(lb) > 1 else 0.0
        trunk_yaw   = lb[2] if len(lb) > 2 else 0.0

        rknee = get('rtibia',  0)
        lknee = get('ltibia',  0)

        rfemur = d.get('rfemur', [0]*3)
        lfemur = d.get('lfemur', [0]*3)
        rhip_pitch = rfemur[0] if len(rfemur) > 0 else 0.0
        lhip_pitch = lfemur[0] if len(lfemur) > 0 else 0.0

        records.append({
            'Time(s)':      t,
            'root_x':       root_x,
            'root_y':       root_y,
            'root_z':       root_z,
            'trunk_pitch':  trunk_pitch,   # deg
            'trunk_roll':   trunk_roll,    # deg
            'trunk_yaw':    root_rz,       # deg
            'rknee_angle':  rknee,         # deg
            'lknee_angle':  lknee,         # deg
            'rhip_pitch':   rhip_pitch,    # deg
            'lhip_pitch':   lhip_pitch,    # deg
        })

    df = pd.DataFrame(records)
    dt = 1.0 / fps

    # Derive angular velocities (like a gyroscope) from joint angles
    df['gyro_pitch'] = np.gradient(df['trunk_pitch'].values, dt)   # deg/s
    df['gyro_roll']  = np.gradient(df['trunk_roll'].values,  dt)   # deg/s
    df['gyro_yaw']   = np.gradient(df['trunk_yaw'].values,   dt)   # deg/s

    # Derive linear acceleration from root position
    df['vel_x']   = np.gradient(df['root_x'].values, dt)
    df['vel_y']   = np.gradient(df['root_y'].values, dt)
    df['vel_z']   = np.gradient(df['root_z'].values, dt)
    df['accel_x'] = np.gradient(df['vel_x'].values,  dt)
    df['accel_y'] = np.gradient(df['vel_y'].values,  dt)
    df['accel_z'] = np.gradient(df['vel_z'].values,  dt)

    # Convert deg/s → rad/s for Gyro
    df['gyro_pitch_rad'] = np.radians(df['gyro_pitch'])
    df['gyro_roll_rad']  = np.radians(df['gyro_roll'])
    df['gyro_yaw_rad']   = np.radians(df['gyro_yaw'])

    return df


def smooth(series, window=5):
    """Apply rolling mean smoothing."""
    return pd.Series(series).rolling(window=window, center=True).mean().bfill().ffill().values


def normalize_time(df, duration=10.0):
    """Return subset of first `duration` seconds, time-normalized to [0, duration]."""
    return df[df['Time(s)'] <= duration].copy()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).parent

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading human walking data (AMC)...")
    amc_path = Path(r"C:\Users\parth\Downloads\walking_human.amc")
    frames   = parse_amc(amc_path)
    human_df = amc_to_dataframe(frames, fps=120.0)
    print(f"  -> {len(human_df)} frames | duration: {human_df['Time(s)'].max():.1f}s")

    print("Loading G1 robot normal walking data (CSV)...")
    robot_csv = base_dir / "imu_data_collect" / "normal_imu_data.csv"
    robot_df  = pd.read_csv(robot_csv)
    # Derive knee angles from robot CSV if available
    has_knee = 'Left_Knee(rad)' in robot_df.columns
    print(f"  -> {len(robot_df)} rows | duration: {robot_df['Time(s)'].max():.1f}s")
    if has_knee:
        print("  -> Knee angle columns found in robot data.")

    # ── Align to first 10s ────────────────────────────────────────────────────
    DURATION = 10.0
    h = normalize_time(human_df, DURATION)
    r = robot_df[robot_df['Time(s)'] <= DURATION].copy()

    # ── Smooth for readability ─────────────────────────────────────────────────
    W = 7  # window for smoothing

    # ─────────────────────────────────────────────────────────────────────────
    #  FIGURE LAYOUT  (4 rows × 3 cols + info panel)
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#0d1117')

    # Title
    fig.text(0.5, 0.975, 'Human Walking  vs  Unitree G1 Robot Walking',
             ha='center', va='top', fontsize=20, fontweight='bold',
             color='#e6edf3')
    fig.text(0.5, 0.955, 'Normal Gait Comparison — IMU & Joint Kinematics',
             ha='center', va='top', fontsize=12, color='#8b949e')

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           top=0.93, bottom=0.06,
                           left=0.06, right=0.97,
                           hspace=0.52, wspace=0.38)

    # ── Legend patches ────────────────────────────────────────────────────────
    human_patch = mpatches.Patch(color=HUMAN_COLOR,  label='Human Walking (MoCap AMC)')
    robot_patch = mpatches.Patch(color=ROBOT_COLOR,  label='G1 Robot Walking (IMU CSV)')
    fig.legend(handles=[human_patch, robot_patch],
               loc='upper right', bbox_to_anchor=(0.97, 0.975),
               fontsize=10, framealpha=0.4)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 0 — Gyroscope / Angular Velocity
    # ═══════════════════════════════════════════════════════════════════════════

    # --- Plot 0,0: Gyro Pitch (Y) ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(h['Time(s)'], smooth(h['gyro_pitch_rad'], W),
            color=HUMAN_COLOR, lw=1.5, label='Human')
    ax.plot(r['Time(s)'], smooth(r['Gyro_Y(rad/s)'], W),
            color=ROBOT_COLOR, lw=1.5, label='G1 Robot')
    ax.set_title('Gyro — Pitch (Y-axis)', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Vel (rad/s)')
    ax.fill_between(h['Time(s)'], smooth(h['gyro_pitch_rad'], W), alpha=0.15, color=HUMAN_COLOR)
    ax.fill_between(r['Time(s)'], smooth(r['Gyro_Y(rad/s)'], W), alpha=0.15, color=ROBOT_COLOR)

    # --- Plot 0,1: Gyro Roll (X) ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(h['Time(s)'], smooth(h['gyro_roll_rad'], W),
            color=HUMAN_COLOR, lw=1.5)
    ax.plot(r['Time(s)'], smooth(r['Gyro_X(rad/s)'], W),
            color=ROBOT_COLOR, lw=1.5)
    ax.set_title('Gyro — Roll (X-axis)', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Vel (rad/s)')
    ax.fill_between(h['Time(s)'], smooth(h['gyro_roll_rad'], W), alpha=0.15, color=HUMAN_COLOR)
    ax.fill_between(r['Time(s)'], smooth(r['Gyro_X(rad/s)'], W), alpha=0.15, color=ROBOT_COLOR)

    # --- Plot 0,2: Gyro Yaw (Z) ---
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(h['Time(s)'], smooth(h['gyro_yaw_rad'], W),
            color=HUMAN_COLOR, lw=1.5)
    ax.plot(r['Time(s)'], smooth(r['Gyro_Z(rad/s)'], W),
            color=ROBOT_COLOR, lw=1.5)
    ax.set_title('Gyro — Yaw (Z-axis)', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Vel (rad/s)')
    ax.fill_between(h['Time(s)'], smooth(h['gyro_yaw_rad'], W), alpha=0.15, color=HUMAN_COLOR)
    ax.fill_between(r['Time(s)'], smooth(r['Gyro_Z(rad/s)'], W), alpha=0.15, color=ROBOT_COLOR)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 1 — Trunk / Base Orientation & Tilt
    # ═══════════════════════════════════════════════════════════════════════════

    # --- Plot 1,0: Trunk Pitch (Gravity X — forward lean) ---
    ax = fig.add_subplot(gs[1, 0])
    # Human trunk pitch in degrees; robot gravity_x is unitless (projection)
    h_pitch_norm = h['trunk_pitch'] / 90.0   # normalize deg to [-1,1]
    ax.plot(h['Time(s)'], smooth(h_pitch_norm.values, W),
            color=HUMAN_COLOR, lw=1.5, label='Human (norm)')
    ax.plot(r['Time(s)'], smooth(r['Gravity_X'], W),
            color=ROBOT_COLOR, lw=1.5, label='G1 Robot')
    ax.set_title('Trunk Forward Lean (Gravity X / Pitch)', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized / Gravity Component')
    ax.axhline(0, color='#8b949e', lw=0.7, linestyle=':')

    # --- Plot 1,1: Trunk Lateral Lean (Gravity Y / Roll) ---
    ax = fig.add_subplot(gs[1, 1])
    h_roll_norm = h['trunk_roll'] / 90.0
    ax.plot(h['Time(s)'], smooth(h_roll_norm.values, W),
            color=HUMAN_COLOR, lw=1.5)
    ax.plot(r['Time(s)'], smooth(r['Gravity_Y'], W),
            color=ROBOT_COLOR, lw=1.5)
    ax.set_title('Trunk Lateral Lean (Gravity Y / Roll)', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized / Gravity Component')
    ax.axhline(0, color='#8b949e', lw=0.7, linestyle=':')

    # --- Plot 1,2: Vertical stability (Gravity Z) ---
    ax = fig.add_subplot(gs[1, 2])
    h_gz_norm = h['root_z'] / h['root_z'].abs().max()
    ax.plot(h['Time(s)'], smooth(h_gz_norm.values, W),
            color=HUMAN_COLOR, lw=1.5, label='Human (norm. height)')
    ax.plot(r['Time(s)'], smooth(r['Gravity_Z'], W),
            color=ROBOT_COLOR, lw=1.5, label='G1 Robot (Gravity Z)')
    ax.set_title('Vertical Stability (Gravity Z / Pelvis Height)', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized')
    ax.axhline(-1.0, color=ROBOT_COLOR, lw=0.6, linestyle=':', alpha=0.5)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 2 — Linear Velocity & Acceleration
    # ═══════════════════════════════════════════════════════════════════════════

    # --- Plot 2,0: Forward Velocity (X) ---
    ax = fig.add_subplot(gs[2, 0])
    h_vx = smooth(h['vel_x'].values, W)
    ax.plot(h['Time(s)'], h_vx,
            color=HUMAN_COLOR, lw=1.5)
    ax.plot(r['Time(s)'], smooth(r['LinVel_X(m/s)'], W),
            color=ROBOT_COLOR, lw=1.5)
    ax.set_title('Forward Velocity (X)', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.fill_between(h['Time(s)'], h_vx, alpha=0.12, color=HUMAN_COLOR)
    ax.fill_between(r['Time(s)'], smooth(r['LinVel_X(m/s)'], W), alpha=0.12, color=ROBOT_COLOR)

    # --- Plot 2,1: Lateral Velocity (Y) ---
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(h['Time(s)'], smooth(h['vel_y'].values, W),
            color=HUMAN_COLOR, lw=1.5)
    ax.plot(r['Time(s)'], smooth(r['LinVel_Y(m/s)'], W),
            color=ROBOT_COLOR, lw=1.5)
    ax.axhline(0, color='#8b949e', lw=0.7, linestyle=':')
    ax.set_title('Lateral Velocity (Y)', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')

    # --- Plot 2,2: Vertical Acceleration (Z) ---
    ax = fig.add_subplot(gs[2, 2])
    h_az = smooth(h['accel_z'].values, W)
    ax.plot(h['Time(s)'], h_az,
            color=HUMAN_COLOR, lw=1.5)
    ax.plot(r['Time(s)'], smooth(r['Accel_Z(m/s2)'], W),
            color=ROBOT_COLOR, lw=1.5)
    ax.set_title('Vertical Acceleration (Z)', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accel (m/s²)')
    ax.axhline(0, color='#8b949e', lw=0.7, linestyle=':')

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 3 — Knee Angles & Statistical Summary
    # ═══════════════════════════════════════════════════════════════════════════

    # --- Plot 3,0: Right Knee Angle ---
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(h['Time(s)'], smooth(h['rknee_angle'].values, W),
            color=HUMAN_COLOR, lw=1.5, label='Human R-Knee (°)')
    if has_knee:
        robot_knee_deg = np.degrees(r['Right_Knee(rad)'].values)
        ax.plot(r['Time(s)'], smooth(robot_knee_deg, W),
                color=ROBOT_COLOR, lw=1.5, label='G1 R-Knee (°)')
    ax.set_title('Right Knee Flexion Angle', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.legend(fontsize=8, loc='upper right')
    ax.fill_between(h['Time(s)'], smooth(h['rknee_angle'].values, W), alpha=0.15, color=HUMAN_COLOR)

    # --- Plot 3,1: Left Knee Angle ---
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(h['Time(s)'], smooth(h['lknee_angle'].values, W),
            color=HUMAN_COLOR, lw=1.5, label='Human L-Knee (°)')
    if has_knee:
        robot_lknee_deg = np.degrees(r['Left_Knee(rad)'].values)
        ax.plot(r['Time(s)'], smooth(robot_lknee_deg, W),
                color=ROBOT_COLOR, lw=1.5, label='G1 L-Knee (°)')
    ax.set_title('Left Knee Flexion Angle', pad=6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.legend(fontsize=8, loc='upper right')
    ax.fill_between(h['Time(s)'], smooth(h['lknee_angle'].values, W), alpha=0.15, color=HUMAN_COLOR)

    # --- Plot 3,2: Statistical Summary Bar Chart ---
    ax = fig.add_subplot(gs[3, 2])
    ax.set_facecolor('#161b22')

    metrics = {
        'Gyro\nPitch\n(rad/s)': (
            np.std(h['gyro_pitch_rad']),
            np.std(r['Gyro_Y(rad/s)'])
        ),
        'Gyro\nRoll\n(rad/s)': (
            np.std(h['gyro_roll_rad']),
            np.std(r['Gyro_X(rad/s)'])
        ),
        'Fwd\nVel\n(m/s)': (
            np.std(h['vel_x']),
            np.std(r['LinVel_X(m/s)'])
        ),
        'Vert\nAccel\n(m/s²)': (
            np.std(h['accel_z']) / 10,   # scale down for vis
            np.std(r['Accel_Z(m/s2)'])
        ),
    }

    x       = np.arange(len(metrics))
    h_vals  = [v[0] for v in metrics.values()]
    r_vals  = [v[1] for v in metrics.values()]
    width   = 0.35

    bars1 = ax.bar(x - width/2, h_vals, width,
                   color=HUMAN_COLOR, alpha=0.85, label='Human (std)')
    bars2 = ax.bar(x + width/2, r_vals, width,
                   color=ROBOT_COLOR, alpha=0.85, label='G1 Robot (std)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys(), fontsize=8)
    ax.set_title('Signal Variability (Std Dev)', pad=6)
    ax.set_ylabel('Std Deviation')
    ax.legend(fontsize=8)

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=7, color=HUMAN_COLOR)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=7, color=ROBOT_COLOR)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = base_dir / "human_vs_robot_walking_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nFigure saved -> {out_path}")
    plt.show()

    # ── Print summary statistics ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY — Normal Walking")
    print("="*60)
    print(f"{'Metric':<30} {'Human':>12} {'G1 Robot':>12}")
    print("-"*60)

    def stat_row(label, h_vals, r_vals):
        hm, hs = np.mean(h_vals), np.std(h_vals)
        rm, rs = np.mean(r_vals), np.std(r_vals)
        print(f"  {label:<28} {hm:+7.3f}±{hs:.3f}  {rm:+7.3f}±{rs:.3f}")

    stat_row("Gyro Pitch (rad/s)",   h['gyro_pitch_rad'], r['Gyro_Y(rad/s)'])
    stat_row("Gyro Roll  (rad/s)",   h['gyro_roll_rad'],  r['Gyro_X(rad/s)'])
    stat_row("Gyro Yaw   (rad/s)",   h['gyro_yaw_rad'],   r['Gyro_Z(rad/s)'])
    stat_row("Forward Vel (m/s)",    h['vel_x'],          r['LinVel_X(m/s)'])
    stat_row("Lateral Vel (m/s)",    h['vel_y'],          r['LinVel_Y(m/s)'])
    stat_row("Vertical Accel (m/s²)",h['accel_z'],        r['Accel_Z(m/s2)'])
    if has_knee:
        stat_row("R-Knee (deg)",
                 h['rknee_angle'],
                 np.degrees(r['Right_Knee(rad)']))
        stat_row("L-Knee (deg)",
                 h['lknee_angle'],
                 np.degrees(r['Left_Knee(rad)']))
    print("="*60)
    print("\n  Human data: MoCap AMC @ 120 Hz (walking_human.amc)")
    print("  Robot data: Unitree G1 Torso IMU @ 50 Hz (normal_imu_data.csv)")
    print("="*60)


if __name__ == "__main__":
    main()
