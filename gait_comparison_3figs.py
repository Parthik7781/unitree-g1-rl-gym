#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gait Comparison: Human (CMU MoCap AMC) vs Unitree G1 Robot (MuJoCo IMU)
========================================================================
Generates three publication-quality figures:
  Fig 1 - Torso Pitch over One Normalized Gait Cycle (with RMSE + Pearson r)
  Fig 2 - Cadence Bar Chart (steps/min + % difference)
  Fig 3 - Spatiotemporal Parameters Grouped Bar Chart

Saved as PNG + PDF at 300 dpi.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AMC_PATH  = Path(r"C:\Users\parth\Downloads\walking_human.amc")
ROBOT_CSV = Path(r"c:\unitree_rl_gym-main\unitree_rl_gym-main\imu_data_collect\normal_imu_data.csv")
OUT_DIR   = Path(r"c:\unitree_rl_gym-main\unitree_rl_gym-main")

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
HUMAN_COLOR = '#1f77b4'   # blue  (per specification)
ROBOT_COLOR = '#d62728'   # red   (per specification)
DPI         = 300
N_GC        = 101         # points per normalized gait cycle (0-100 %)

# Human AMC position-unit scale estimate:
#   root_ty ~ 16.7 units; typical hip height ~ 90 cm  =>  1 unit ~ 5.4 cm = 0.054 m
HUMAN_POS_SCALE = 0.054   # metres per AMC position unit

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.titleweight':   'bold',
    'axes.labelsize':     11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.45,
    'grid.linestyle':     '--',
    'grid.linewidth':     0.6,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#aaaaaa',
    'legend.fontsize':    10,
    'figure.facecolor':   'white',
    'axes.facecolor':     '#fafafa',
})


# ===========================================================================
# AMC Parser
# ===========================================================================
def parse_amc(filepath):
    """
    Parse a CMU AMC motion-capture file.
    Returns dict { frame_number: { bone_name: [float, ...] } }
    """
    frames, cur_frame, cur_data = {}, None, {}
    with open(str(filepath), 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line[0] in ('#', ':'):
                continue
            parts = line.split()
            # A bare integer signals a new frame
            if len(parts) == 1:
                try:
                    fn = int(parts[0])
                    if cur_frame is not None:
                        frames[cur_frame] = cur_data
                    cur_frame, cur_data = fn, {}
                    continue
                except ValueError:
                    pass
            # Otherwise: bone + values
            if cur_frame is not None and len(parts) >= 2:
                try:
                    cur_data[parts[0]] = [float(v) for v in parts[1:]]
                except ValueError:
                    pass
    if cur_frame is not None:
        frames[cur_frame] = cur_data
    return frames


def amc_to_df(frames, fps=120.0):
    """Convert AMC frame dict to a pandas DataFrame with derived signals."""
    rows = []
    for fn in sorted(frames.keys()):
        d  = frames[fn]
        t  = fn / fps

        def g(bone, i=0):
            v = d.get(bone, [])
            return float(v[i]) if i < len(v) else 0.0

        root = d.get('root', [0.0] * 6)
        r    = lambda i: float(root[i]) if i < len(root) else 0.0

        # root = [tx, ty, tz, rx, ry, rz]  (position in AMC units, angles in degrees)
        # Walking direction: root_tz (Z axis)
        # Global trunk pitch rotation: root_ry (rotation about mediolateral axis)
        rows.append({
            'time':        t,
            'root_tx':     r(0),  'root_ty':    r(1),  'root_tz':     r(2),
            'root_rx':     r(3),  'root_ry':    r(4),  'root_rz':     r(5),
            # sagittal (flex/ext) DOF of each spinal segment
            'lowerback_0': g('lowerback', 0),
            'upperback_0': g('upperback', 0),
            'thorax_0':    g('thorax',    0),
            # knee flexion angles (single DOF in AMC)
            'rtibia':      g('rtibia', 0),
            'ltibia':      g('ltibia', 0),
        })

    df = pd.DataFrame(rows)

    # Composite trunk pitch = global orientation + segmental contributions
    df['trunk_pitch'] = (df['root_ry']    +
                         df['lowerback_0'] +
                         df['upperback_0'] +
                         df['thorax_0'])

    # Forward velocity in m/s (walking direction = root_tz)
    dt = 1.0 / fps
    df['fwd_vel_ms'] = np.gradient(df['root_tz'].values, dt) * HUMAN_POS_SCALE

    return df


# ===========================================================================
# Robot loader
# ===========================================================================
def load_robot(filepath):
    """Load robot IMU CSV and derive trunk pitch from quaternion."""
    df = pd.read_csv(str(filepath)).rename(columns={'Time(s)': 'time'})

    # Trunk pitch from quaternion: standard aerospace convention
    # pitch = arctan2(2(qw*qy + qx*qz), 1 - 2(qy^2 + qx^2))
    qw = df['Quat_W'].values;  qx = df['Quat_X'].values
    qy = df['Quat_Y'].values;  qz = df['Quat_Z'].values
    df['trunk_pitch'] = np.degrees(
        np.arctan2(2.0 * (qw*qy + qx*qz),
                   1.0 - 2.0 * (qy**2 + qx**2)))
    return df


# ===========================================================================
# Signal utilities
# ===========================================================================
def smooth(x, fps, window_s=0.10, poly=2):
    """Savitzky-Golay filter with a window length given in seconds."""
    wl = max(5, int(window_s * fps))
    wl += (wl % 2 == 0)          # must be odd
    if len(x) <= wl:
        return x.copy()
    return savgol_filter(x, wl, poly)


def gc_interp(sig, n=N_GC):
    """Interpolate an extracted gait cycle to n equally spaced points (0-100 %)."""
    t_in  = np.linspace(0.0, 100.0, len(sig))
    t_out = np.linspace(0.0, 100.0, n)
    return interp1d(t_in, sig, kind='cubic')(t_out)


def detect_events(sig, fps, min_dist_s, find_minima=False):
    """Find peaks (or minima) in a smoothed signal."""
    sm   = smooth(sig, fps, window_s=0.08)
    mdst = max(1, int(min_dist_s * fps))
    idx, props = find_peaks(-sm if find_minima else sm, distance=mdst)
    return idx, sm


def save_fig(fig, stem):
    """Save figure as both PNG and PDF at DPI resolution."""
    for ext in ('png', 'pdf'):
        p = OUT_DIR / f'{stem}.{ext}'
        fig.savefig(str(p), dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"  Saved -> {p}")
    plt.close(fig)


# ===========================================================================
# Figure 1: Torso Pitch over Normalized Gait Cycle
# ===========================================================================
def figure1_torso_pitch(h_df, r_df):
    print("\n" + "="*60)
    print("[Figure 1]  Torso Pitch - Normalized Gait Cycle")
    print("="*60)
    fps_h, fps_r = 120.0, 50.0

    # -------------------------------------------------------------------------
    # HUMAN  gait cycle
    # -------------------------------------------------------------------------
    h_pitch_raw = h_df['trunk_pitch'].values
    h_time      = h_df['time'].values
    h_pitch_sm  = smooth(h_pitch_raw, fps_h, window_s=0.08)

    # Right heel strike = local minimum of rtibia (knee maximally extended)
    rt_sm       = smooth(h_df['rtibia'].values, fps_h, window_s=0.08)
    hs_h, _     = detect_events(rt_sm, fps_h, min_dist_s=0.40, find_minima=True)

    print(f"  Human: {len(hs_h)} heel-strike events over {h_time[-1]:.2f} s  (@ 120 Hz)")

    if len(hs_h) >= 2:
        i0, i1   = hs_h[0], hs_h[1]
        h_gc_sig = h_pitch_sm[i0:i1 + 1]
        h_dur    = h_time[i1] - h_time[i0]
        method_h = "rtibia minima (right heel strike)"
    else:
        print("  LIMITATION: < 2 human gait events detected. "
              "Using full recording as a proxy gait cycle.")
        h_gc_sig = h_pitch_sm
        h_dur    = h_time[-1] - h_time[0]
        method_h = "full recording (insufficient heel-strike events)"

    h_norm     = gc_interp(h_gc_sig)
    h_centered = h_norm - h_norm.mean()
    h_range    = h_norm.max() - h_norm.min()

    # -------------------------------------------------------------------------
    # ROBOT  gait cycle
    # -------------------------------------------------------------------------
    r_pitch_raw = r_df['trunk_pitch'].values
    r_time      = r_df['time'].values
    r_pitch_sm  = smooth(r_pitch_raw, fps_r, window_s=0.12)

    # Gait events from Gyro_Y (pitch angular velocity) after startup transient
    skip_s      = 1.5                        # seconds of startup to ignore
    skip_idx    = int(skip_s * fps_r)
    gy_sm       = smooth(r_df['Gyro_Y(rad/s)'].values, fps_r, window_s=0.10)
    ev_rel, _   = detect_events(gy_sm[skip_idx:], fps_r,
                                min_dist_s=0.30, find_minima=False)
    ev_r        = ev_rel + skip_idx          # map back to full array indices

    print(f"  Robot:  {len(ev_r)} Gyro_Y events over {r_time[-1]:.2f} s  (@ 50 Hz, "
          f"first {skip_s:.1f} s skipped)")

    # Identify one complete STRIDE (= peak-to-peak interval ~0.6-1.2 s)
    r_dur     = None
    r_gc_sig  = None
    method_r  = "Gyro_Y peaks (pitch angular velocity)"

    for j in range(len(ev_r) - 1):
        for k in range(j + 1, len(ev_r)):
            dt = r_time[ev_r[k]] - r_time[ev_r[j]]
            if 0.55 <= dt <= 1.20:
                i0r, i1r = ev_r[j], ev_r[k]
                r_gc_sig = r_pitch_sm[i0r:i1r + 1]
                r_dur    = dt
                break
        if r_dur is not None:
            break

    if r_gc_sig is None:
        print("  LIMITATION: Could not auto-detect robot stride boundary. "
              "Using phase-period window of 0.8 s (from controller design).")
        r_dur    = 0.80
        t0_idx   = int(3.0 * fps_r)
        t1_idx   = min(len(r_pitch_sm) - 1, int((3.0 + r_dur) * fps_r))
        r_gc_sig = r_pitch_sm[t0_idx:t1_idx + 1]
        method_r = "controller phase period (0.8 s, fallback)"

    r_norm     = gc_interp(r_gc_sig)
    r_centered = r_norm - r_norm.mean()
    r_range    = r_norm.max() - r_norm.min()

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    rmse = np.sqrt(np.mean((h_centered - r_centered) ** 2))
    corr, pval = pearsonr(h_centered, r_centered)

    print(f"\n  Human trunk pitch range : {h_range:.2f} deg   (cycle dur = {h_dur:.3f} s)")
    print(f"  Robot trunk pitch range : {r_range:.2f} deg   (cycle dur = {r_dur:.3f} s)")
    print(f"  RMSE (zero-mean curves) : {rmse:.3f} deg")
    print(f"  Pearson r               : {corr:.3f}   (p = {pval:.4f})")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    gc = np.linspace(0, 100, N_GC)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(gc, h_centered, color=HUMAN_COLOR, lw=2.2,
            label='Human (CMU MoCap AMC)', zorder=4)
    ax.fill_between(gc, h_centered, alpha=0.14, color=HUMAN_COLOR)

    ax.plot(gc, r_centered, color=ROBOT_COLOR, lw=2.2, linestyle='--',
            label='Unitree G1 Robot (MuJoCo IMU)', zorder=4)
    ax.fill_between(gc, r_centered, alpha=0.14, color=ROBOT_COLOR)

    ax.axhline(0.0, color='#888888', lw=0.8, linestyle=':', zorder=1)

    # Gait phase shading
    y0, y1 = ax.get_ylim()
    ax.axvspan( 0,  60, alpha=0.04, color='#888888', zorder=0)
    ax.axvspan(60, 100, alpha=0.07, color='#ffc000', zorder=0)

    # Phase event lines
    for pct in (0, 60, 100):
        ax.axvline(pct, color='#555555', lw=0.9, ls='--', alpha=0.55, zorder=2)

    # Phase labels - positioned after plot is drawn
    ylim = ax.get_ylim()
    dy   = (ylim[1] - ylim[0])
    for pct, lbl, ha in [(2, 'IC\n(Heel Strike)', 'left'),
                          (62, 'TO\n(Toe-Off)', 'left'),
                          (98, 'IC', 'right')]:
        ax.text(pct, ylim[0] + 0.05 * dy, lbl,
                fontsize=8, color='#333333', ha=ha, va='bottom',
                style='italic', fontweight='bold')

    # Stance / Swing labels
    ax.text(30, ylim[1] - 0.06 * dy, 'STANCE  (0-60%)',
            fontsize=9, color='#555555', ha='center', va='top', style='italic')
    ax.text(80, ylim[1] - 0.06 * dy, 'SWING (60-100%)',
            fontsize=9, color='#996600', ha='center', va='top', style='italic')

    ax.set_xlim(0, 100)
    ax.set_xlabel('Gait Cycle (%)', fontsize=11)
    ax.set_ylabel('Torso Pitch Angle (degrees, zero-mean)', fontsize=11)
    ax.set_title('Figure 1: Torso Pitch over One Normalized Gait Cycle\n'
                 'Human (CMU MoCap) vs Unitree G1 Robot (MuJoCo IMU)', fontsize=13)

    # Legend
    h_patch = mpatches.Patch(color=HUMAN_COLOR, label=f'Human  — range {h_range:.1f} deg')
    r_patch = mpatches.Patch(color=ROBOT_COLOR,
                             label=f'G1 Robot — range {r_range:.1f} deg')
    ax.legend(handles=[h_patch, r_patch], loc='upper right', fontsize=10)

    # Stats box inside the plot
    stats_txt = (f'RMSE = {rmse:.2f} deg     '
                 f'Pearson r = {corr:.3f}     '
                 f'p-value = {pval:.4f}')
    ax.text(0.50, 0.04, stats_txt, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#cccccc', alpha=0.96))

    # Methodological note at figure bottom
    note = (f'Note: Both curves are zero-mean centred for shape comparison. '
            f'Human gait events: {method_h}. '
            f'Robot gait events: {method_r}. '
            f'IC = Initial Contact; TO = Toe-Off.')
    fig.text(0.5, 0.005, note, ha='center', fontsize=8,
             color='#555555', style='italic')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, 'fig1_torso_pitch')

    return h_dur, r_dur, rmse, corr, pval


# ===========================================================================
# Cadence helpers
# ===========================================================================
def compute_cadence(h_df, r_df):
    """
    Human:
      - Detect peaks in rtibia and ltibia separately.
      - Use prominence + height threshold to keep only true swing-phase peaks
        (knee maximally flexed ~ 50-80 deg; filter out the secondary ~35 deg bumps).
      - Merge left/right events in time-order and deduplicate events within 0.15 s
        (same step cannot appear twice).
    Robot:
      - Peaks in lateral velocity (LinVel_Y); peaks + troughs each = one step.
      - Skip first 1 s startup transient.
    """
    fps_h, fps_r = 120.0, 50.0
    h_dur = h_df['time'].values[-1] - h_df['time'].values[0]

    # ---- Human ----
    rt = smooth(h_df['rtibia'].values, fps_h, window_s=0.08)
    lt = smooth(h_df['ltibia'].values, fps_h, window_s=0.08)
    h_time = h_df['time'].values

    # Keep only the large swing-phase peaks (>= 55 deg) to avoid counting
    # secondary partial-flexion bumps as separate steps
    min_knee_angle = 50.0  # degrees
    rp, rprops = find_peaks(rt, distance=int(0.35 * fps_h),
                            height=min_knee_angle, prominence=10.0)
    lp, lprops = find_peaks(lt, distance=int(0.35 * fps_h),
                            height=min_knee_angle, prominence=10.0)

    # Merge into one sorted list and deduplicate events within 0.15 s
    all_ev = sorted([(h_time[i], 'R') for i in rp] +
                    [(h_time[i], 'L') for i in lp])
    dedup = []
    for t, side in all_ev:
        if not dedup or (t - dedup[-1][0]) > 0.15:
            dedup.append((t, side))

    h_steps   = len(dedup)
    h_cadence = (h_steps / h_dur) * 60.0 if h_dur > 0 else 0.0

    # ---- Robot (skip first 1 s) ----
    r_skip = int(1.0 * fps_r)
    r_sub  = r_df[r_df['time'] > 1.0].copy()
    r_edur = r_sub['time'].values[-1] - r_sub['time'].values[0]
    lv     = smooth(r_sub['LinVel_Y(m/s)'].values, fps_r, window_s=0.08)
    rp2, _ = find_peaks( lv, distance=int(0.25 * fps_r))
    rt2, _ = find_peaks(-lv, distance=int(0.25 * fps_r))
    r_steps   = len(rp2) + len(rt2)
    r_cadence = (r_steps / r_edur) * 60.0 if r_edur > 0 else 0.0

    return h_cadence, r_cadence, h_steps, r_steps, h_dur, r_sub['time'].values[-1]



# ===========================================================================
# Figure 2: Cadence Comparison
# ===========================================================================
def figure2_cadence(h_df, r_df):
    print("\n" + "="*60)
    print("[Figure 2]  Cadence Comparison")
    print("="*60)

    h_cad, r_cad, h_s, r_s, h_dur, r_total = compute_cadence(h_df, r_df)
    pct_diff = abs(h_cad - r_cad) / h_cad * 100.0 if h_cad > 0 else 0.0

    print(f"  Human  : {h_s} steps in {h_dur:.2f} s => {h_cad:.1f} steps/min")
    print(f"  Robot  : {r_s} steps in {r_total:.2f} s => {r_cad:.1f} steps/min")
    print(f"  Difference : {abs(h_cad - r_cad):.1f} steps/min ({pct_diff:.1f} %)")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    bars = ax.bar(
        ['Human\n(CMU MoCap)', 'Unitree G1\n(Robot)'],
        [h_cad, r_cad],
        color=[HUMAN_COLOR, ROBOT_COLOR],
        width=0.42, edgecolor='white', linewidth=1.5, zorder=3
    )

    # Value labels on top of each bar
    for bar, val in zip(bars, [h_cad, r_cad]):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.5,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=13, fontweight='bold')

    # Difference annotation between bars
    mid_y = (h_cad + r_cad) / 2.0
    ax.annotate('', xy=(1.0, r_cad * 0.97), xytext=(0.0, h_cad * 0.97),
                xycoords='data',
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.8))
    ax.text(0.50, mid_y,
            f'{pct_diff:.1f} %\ndifference',
            ha='center', va='center', fontsize=10, color='#222222',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.95))

    # Normal-walking reference band (95-125 steps/min, clinical norm)
    ax.axhspan(95, 125, alpha=0.07, color=HUMAN_COLOR, zorder=0)
    ax.axhline(95,  color=HUMAN_COLOR, lw=1.0, ls=':', alpha=0.6)
    ax.axhline(125, color=HUMAN_COLOR, lw=1.0, ls=':', alpha=0.6)
    ax.text(1.48, 110,
            'Normal range\n(95-125 spm)',
            ha='right', va='center', fontsize=8.5,
            color='#1a5fa8', style='italic')

    ymax = max(h_cad, r_cad) * 1.30
    ax.set_ylim(0, ymax)
    ax.set_ylabel('Cadence  (steps / min)', fontsize=11)
    ax.set_title('Figure 2: Walking Cadence Comparison\n'
                 'Human (CMU MoCap) vs Unitree G1 Robot', fontsize=13)

    h_lbl = mpatches.Patch(color=HUMAN_COLOR,
                           label=f'Human :  {h_cad:.1f} steps/min')
    r_lbl = mpatches.Patch(color=ROBOT_COLOR,
                           label=f'G1 Robot:  {r_cad:.1f} steps/min')
    ax.legend(handles=[h_lbl, r_lbl], loc='upper left', fontsize=9.5)

    note = ('Human cadence: right + left knee-flexion peaks (rtibia + ltibia).\n'
            'Robot cadence: peaks + troughs in lateral velocity (LinVel_Y), '
            'startup transient (first 1 s) excluded.')
    fig.text(0.5, 0.01, note, ha='center', fontsize=8,
             color='#555555', style='italic')

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    save_fig(fig, 'fig2_cadence')

    return h_cad, r_cad


# ===========================================================================
# Figure 3: Spatiotemporal Gait Parameters
# ===========================================================================
def figure3_spatiotemporal(h_df, r_df, h_cad, r_cad):
    print("\n" + "="*60)
    print("[Figure 3]  Spatiotemporal Parameters")
    print("="*60)

    # Walking Speed
    h_speed = float(np.abs(np.mean(h_df['fwd_vel_ms'].values)))
    r_speed = float(np.abs(np.mean(r_df['LinVel_X(m/s)'].values)))

    # Step frequency (steps per second)
    h_sf = h_cad / 60.0
    r_sf = r_cad / 60.0

    # Step Length = speed / step_frequency
    # (spatial distance covered per step)
    h_step_len   = h_speed / h_sf  if h_sf > 0 else 0.0
    r_step_len   = r_speed / r_sf  if r_sf > 0 else 0.0

    # Stride Length = 2 x Step Length
    h_stride_len = 2.0 * h_step_len
    r_stride_len = 2.0 * r_step_len

    # Gait Cycle Duration (stride period) = 2 / step_frequency
    h_gc_dur = 2.0 / h_sf if h_sf > 0 else 0.0
    r_gc_dur = 2.0 / r_sf if r_sf > 0 else 0.0

    params = [
        ('Walking Speed\n(m/s)',         h_speed,      r_speed),
        ('Step Length\n(m)',             h_step_len,   r_step_len),
        ('Stride Length\n(m)',           h_stride_len, r_stride_len),
        ('Gait Cycle\nDuration (s)',     h_gc_dur,     r_gc_dur),
    ]

    print(f"\n  {'Parameter':<32}  {'Human':>8}  {'G1 Robot':>8}  {'Diff %':>8}")
    print("  " + "-" * 60)
    for lbl, hv, rv in params:
        diff_pct = abs(hv - rv) / hv * 100.0 if hv > 0 else float('nan')
        print(f"  {lbl.replace(chr(10), ' '):<32}  {hv:8.3f}  {rv:8.3f}  {diff_pct:7.1f}%")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x     = np.arange(len(params))
    w     = 0.30
    hvals = [p[1] for p in params]
    rvals = [p[2] for p in params]

    bh = ax.bar(x - w/2, hvals, w,
                color=HUMAN_COLOR, alpha=0.88,
                edgecolor='white', linewidth=1.2,
                label='Human (CMU MoCap)', zorder=3)
    br = ax.bar(x + w/2, rvals, w,
                color=ROBOT_COLOR,  alpha=0.88,
                edgecolor='white', linewidth=1.2,
                label='Unitree G1 Robot (IMU)', zorder=3)

    # Value labels
    for bar, val in zip(bh, hvals):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.006,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#0a4a8f')
    for bar, val in zip(br, rvals):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.006,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#a01010')

    # % difference markers above each group
    for i, (lbl, hv, rv) in enumerate(params):
        dp = abs(hv - rv) / hv * 100.0 if hv > 0 else 0.0
        ymark = max(hv, rv) + 0.05 * max(hvals + rvals)
        ax.text(i, ymark, f'{dp:.1f} %', ha='center', va='bottom',
                fontsize=9, color='#444444',
                bbox=dict(boxstyle='round,pad=0.2', fc='#ffffee',
                          ec='#cccc88', alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in params], fontsize=10.5)
    ax.set_ylabel('Value  (units shown on x-axis)', fontsize=11)
    ax.set_title('Figure 3: Spatiotemporal Gait Parameters\n'
                 'Human (CMU MoCap) vs Unitree G1 Robot (MuJoCo IMU)', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(max(hvals), max(rvals)) * 1.35)

    # Limitation footnote
    note = ('LIMITATION - Human walking speed: AMC position unit estimated at 0.054 m '
            '(hip height root_ty ~16.7 units ~ 90 cm); '
            'walking speed and derived spatial metrics carry ~+/-15% uncertainty. '
            'Step length = speed / cadence;  Stride = 2 x Step;  '
            'GC duration = 2 / step_frequency.')
    fig.text(0.5, 0.01, note, ha='center', fontsize=8,
             color='#555555', style='italic')

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    save_fig(fig, 'fig3_spatiotemporal')


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("\n" + "=" * 60)
    print("  Gait Comparison: CMU MoCap Human  vs  Unitree G1 Robot")
    print("=" * 60)

    print("\n[Loading]  Human AMC ...")
    h_df = amc_to_df(parse_amc(AMC_PATH), fps=120.0)
    print(f"  {len(h_df):,} frames  |  {h_df['time'].max():.2f} s  @  120 Hz")

    print("[Loading]  Robot IMU CSV ...")
    r_df = load_robot(ROBOT_CSV)
    print(f"  {len(r_df):,} rows   |  {r_df['time'].max():.2f} s  @  50 Hz")

    h_dur, r_dur, rmse, corr, pval = figure1_torso_pitch(h_df, r_df)
    h_cad, r_cad                   = figure2_cadence(h_df, r_df)
    figure3_spatiotemporal(h_df, r_df, h_cad, r_cad)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Fig 1 - RMSE: {rmse:.2f} deg  |  Pearson r: {corr:.3f}  |  p={pval:.4f}")
    print(f"  Fig 1 - Human GC duration: {h_dur:.3f} s  |  Robot GC: {r_dur:.3f} s")
    print(f"  Fig 2 - Human cadence: {h_cad:.1f} spm  |  Robot: {r_cad:.1f} spm")
    print(f"\n  Outputs saved in:  {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
