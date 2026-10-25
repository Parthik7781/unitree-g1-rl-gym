"""
Research Paper Style Gait Analysis Plots
Generates publication-quality figures matching journal formatting.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

# numpy compat
try:
    from numpy import trapezoid as np_trapz
except ImportError:
    from numpy import trapz as np_trapz

# ── CONFIG ───────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "validation_gait_data" / "data"
OUTPUT_DIR = Path(__file__).parent / "validation_gait_data" / "paper_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT_CSV   = DATA_DIR / "left_ankle.csv"
RIGHT_CSV  = DATA_DIR / "right_ankle.csv"

SAMPLE_RATE = 100.0
DT = 1.0 / SAMPLE_RATE

# Journal-style matplotlib settings
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'text.usetex': False,
    'mathtext.fontset': 'dejavuserif',
})

CYCLE_COLORS = ['#000000','#1f77b4','#d62728','#2ca02c','#ff7f0e',
                '#9467bd','#8c564b','#e377c2','#17becf','#bcbd22']


def load_and_smooth(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]
    for c in ["ax","ay","az","gx","gy","gz"]:
        if c in df.columns:
            df[c] = savgol_filter(df[c].values, 11, 3)
    return df


def estimate_position(df):
    ax = df["ax"].values.copy()
    ay = df["ay"].values.copy()
    az = df["az"].values.copy()
    roll  = df["roll"].values if "roll" in df.columns else np.zeros(len(df))
    pitch = df["pitch"].values if "pitch" in df.columns else np.zeros(len(df))

    g = 9.81
    ax_clean = ax - g * np.sin(pitch)
    ay_clean = ay - (-g * np.sin(roll) * np.cos(pitch))

    # Tighter bandpass to keep only gait frequencies (0.5-6 Hz)
    sos_bp = signal.butter(4, [0.5, 6.0], btype="band", fs=SAMPLE_RATE, output="sos")
    ax_filt = signal.sosfiltfilt(sos_bp, ax_clean)
    ay_filt = signal.sosfiltfilt(sos_bp, ay_clean)

    # Stance detection for ZUPT
    accel_mag = np.sqrt(ax_filt**2 + ay_filt**2 + (az - g*np.cos(roll)*np.cos(pitch))**2)
    accel_mag_s = savgol_filter(accel_mag, 15, 3)
    is_stance = accel_mag_s < np.percentile(accel_mag_s, 30)

    vx = cumtrapz(ax_filt, dx=DT, initial=0)
    vy = cumtrapz(ay_filt, dx=DT, initial=0)
    vx[is_stance] = 0.0
    vy[is_stance] = 0.0
    vx = savgol_filter(signal.detrend(vx, type="linear"), 15, 3)
    vy = savgol_filter(signal.detrend(vy, type="linear"), 15, 3)

    px = cumtrapz(vx, dx=DT, initial=0)
    py = cumtrapz(vy, dx=DT, initial=0)
    sos_hp = signal.butter(3, 0.3, btype="high", fs=SAMPLE_RATE, output="sos")
    px = savgol_filter(signal.sosfiltfilt(sos_hp, px), 21, 3)
    py = savgol_filter(signal.sosfiltfilt(sos_hp, py), 21, 3)

    # Extra smoothing for cleaner trajectory
    px = savgol_filter(px, 31, 3)
    py = savgol_filter(py, 31, 3)

    return df["timestamp"].values, px, py


def detect_cycles_from_gyro(df):
    """Detect gait cycle boundaries using gyroscope gy (swing peaks)."""
    gy = savgol_filter(df["gy"].values, 15, 3)
    peaks, _ = find_peaks(gy, distance=int(SAMPLE_RATE * 0.4),
                          prominence=np.std(gy) * 0.5)
    if len(peaks) < 2:
        peaks, _ = find_peaks(gy, distance=int(SAMPLE_RATE * 0.3))
    return peaks


def detect_cycles(py):
    """Detect gait cycles using Y-position peaks."""
    py_smooth = savgol_filter(py, 25, 3)
    peaks, _ = find_peaks(py_smooth, distance=30, prominence=np.std(py_smooth)*0.3)
    if len(peaks) < 2:
        peaks, _ = find_peaks(py_smooth, distance=20)
    return peaks


def extract_cycles(px, py, peaks):
    """Extract individual gait cycles between consecutive peaks."""
    cycles = []
    for i in range(len(peaks)-1):
        s, e = peaks[i], peaks[i+1]
        cx = px[s:e+1].copy()
        cy = py[s:e+1].copy()
        # Per-cycle detrend: remove linear drift within each cycle
        cx = signal.detrend(cx, type='linear')
        cy = signal.detrend(cy, type='constant')  # center vertically
        cycles.append((cx, cy))
    return cycles


def to_cm(arr):
    return arr * 100.0


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Multi-panel (A,B,C,D) - like Fig.5 from the paper
# ═══════════════════════════════════════════════════════════════════════
def plot_fig5_style(px_l, py_l, px_r, py_r, left_df=None):
    """4-panel figure: A=trajectories, B=X amplitude, C=Y amplitude, D=comparison"""
    # Use gyro-based cycle detection if dataframe available, else position-based
    if left_df is not None:
        peaks_l = detect_cycles_from_gyro(left_df)
    else:
        peaks_l = detect_cycles(py_l)
    cycles_l = extract_cycles(px_l, py_l, peaks_l)

    # Convert to cm
    px_l_cm, py_l_cm = to_cm(px_l), to_cm(py_l)
    px_r_cm, py_r_cm = to_cm(px_r), to_cm(py_r)
    cycles_cm = [(to_cm(cx), to_cm(cy)) for cx, cy in cycles_l]

    fig = plt.figure(figsize=(7, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.50)

    # ── Panel A: Gait Trajectories in X-Y plane (overlaid cycles) ──
    ax_a = fig.add_subplot(gs[0])

    # Resample all cycles to common length for overlay & envelope
    if len(cycles_cm) >= 1:
        n_resamp = 100
        cx_stack, cy_stack = [], []
        for cx, cy in cycles_cm:
            t_norm = np.linspace(0, 1, len(cx))
            t_new = np.linspace(0, 1, n_resamp)
            cx_stack.append(np.interp(t_new, t_norm, cx))
            cy_stack.append(np.interp(t_new, t_norm, cy))
        cx_stack = np.array(cx_stack)
        cy_stack = np.array(cy_stack)

        # Plot each cycle in black (like paper)
        for i in range(len(cx_stack)):
            ax_a.plot(cx_stack[i], cy_stack[i], 'k-', lw=0.8, alpha=0.6)

        if len(cx_stack) >= 2:
            mean_x = cx_stack.mean(axis=0)
            mean_y = cy_stack.mean(axis=0)
            upper_x, upper_y = cx_stack.max(axis=0), cy_stack.max(axis=0)
            lower_x, lower_y = cx_stack.min(axis=0), cy_stack.min(axis=0)

            ax_a.plot(upper_x, upper_y, 'r-', lw=1.3, label='Upper extreme')
            ax_a.plot(lower_x, lower_y, 'g-', lw=1.3, label='Lower extreme')
            ax_a.plot(mean_x, mean_y, 'b--', lw=1.8, label='Mean trajectory')
            ax_a.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax_a.text(0.02, 0.90, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold')
    ax_a.set_xlabel('X axis(cm)')
    ax_a.set_ylabel('Y axis(cm)')

    # ── Panel B: X Amplitude vs Samples ──
    ax_b = fig.add_subplot(gs[1])
    n_samples = len(px_l_cm)
    samples = np.arange(n_samples)

    # Plot raw signal in black
    ax_b.plot(samples, px_l_cm, 'k-', lw=0.8, alpha=0.6)

    # Overlay envelope from cycle extremes
    if len(cycles_cm) >= 2:
        cx_full = np.array([np.interp(np.linspace(0,1,n_samples),
                            np.linspace(0,1,len(c[0])), c[0]) for c in cycles_cm])
        ax_b.plot(samples, cx_full.max(axis=0), 'r-', lw=1.0)
        ax_b.plot(samples, cx_full.min(axis=0), 'g-', lw=1.0)
        ax_b.plot(samples, cx_full.mean(axis=0), 'b--', lw=1.2)

    ax_b.text(0.02, 0.90, 'B', transform=ax_b.transAxes, fontsize=14, fontweight='bold')
    ax_b.set_xlabel('X Samples')
    ax_b.set_ylabel('Amplitude(cm)')

    # ── Panel C: Y Amplitude vs Samples ──
    ax_c = fig.add_subplot(gs[2])
    ax_c.plot(samples, py_l_cm, 'k-', lw=0.8, alpha=0.6)

    if len(cycles_cm) >= 2:
        cy_full = np.array([np.interp(np.linspace(0,1,n_samples),
                            np.linspace(0,1,len(c[1])), c[1]) for c in cycles_cm])
        ax_c.plot(samples, cy_full.max(axis=0), 'r-', lw=1.0)
        ax_c.plot(samples, cy_full.min(axis=0), 'g-', lw=1.0)
        ax_c.plot(samples, cy_full.mean(axis=0), 'b--', lw=1.2)

    ax_c.text(0.02, 0.90, 'C', transform=ax_c.transAxes, fontsize=14, fontweight='bold')
    ax_c.set_xlabel('Y Samples')
    ax_c.set_ylabel('Amplitude(cm)')

    # ── Panel D: Left vs Right comparison ──
    ax_d = fig.add_subplot(gs[3])
    ax_d.plot(px_l_cm, py_l_cm, 'r-', lw=1.3, label='Left Ankle')
    # Right ankle with circle markers (like "simulation" in paper)
    step = max(1, len(px_r_cm) // 80)
    ax_d.plot(px_r_cm, py_r_cm, color='blue', lw=0.3, alpha=0.3)
    ax_d.plot(px_r_cm[::step], py_r_cm[::step], 'o', color='blue',
              markersize=3, markerfacecolor='blue', markeredgewidth=0.3,
              markeredgecolor='darkblue', label='Right Ankle', zorder=3)

    ax_d.text(0.02, 0.90, 'D', transform=ax_d.transAxes, fontsize=14, fontweight='bold')
    ax_d.set_xlabel('X axis(cm)')
    ax_d.set_ylabel('Y axis(cm)')
    ax_d.legend(loc='lower right', fontsize=8, framealpha=0.9)

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.tick_params(direction='in', top=True, right=True)

    plt.savefig(str(OUTPUT_DIR / "fig5_gait_trajectories.png"), bbox_inches='tight', dpi=300)
    plt.savefig(str(OUTPUT_DIR / "fig5_gait_trajectories.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] fig5_gait_trajectories.png/pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Trial domain analysis - like Fig.7/8 from the paper
# ═══════════════════════════════════════════════════════════════════════
def compute_auct(px, py):
    """Area Under Curve of Trajectory for each detected cycle."""
    peaks = detect_cycles(py)
    aucts = []
    for i in range(len(peaks)-1):
        s, e = peaks[i], peaks[i+1]
        cx, cy = px[s:e+1], py[s:e+1]
        area = np_trapz(np.abs(cy), cx) if len(cx) > 1 else 0
        aucts.append(abs(area) * 10000)  # convert to cm²
    return np.array(aucts)


def plot_fig7_style(px_l, py_l, px_r, py_r, label="Subject 1"):
    """2-row figure: top=trajectory phases, bottom=AUCT trial domain"""
    px_l_cm, py_l_cm = to_cm(px_l), to_cm(py_l)
    px_r_cm, py_r_cm = to_cm(px_r), to_cm(py_r)

    n = len(px_l)
    seg_size = n // 5

    # Define 5 phases
    phases = [
        ("Baseline", 0, seg_size),
        ("Early\nSwing", seg_size, 2*seg_size),
        ("Mid\nSwing", 2*seg_size, 3*seg_size),
        ("Late\nSwing", 3*seg_size, 4*seg_size),
        ("Terminal\nStance", 4*seg_size, n),
    ]

    phase_colors = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd']

    fig = plt.figure(figsize=(10, 6.5))
    gs = fig.add_gridspec(2, 5, hspace=0.5, wspace=0.35)

    # ── Top row: trajectory segments ──
    for i, (title, s, e) in enumerate(phases):
        ax = fig.add_subplot(gs[0, i])
        # Full trajectory in light gray
        ax.plot(px_l_cm, py_l_cm, color='#cccccc', lw=0.5, alpha=0.5)
        # Highlighted segment
        ax.plot(px_l_cm[s:e], py_l_cm[s:e], color=phase_colors[i], lw=1.8)
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xlabel('X (cm)', fontsize=7)
        if i == 0:
            ax.set_ylabel('Y (cm)', fontsize=7)
        ax.tick_params(labelsize=6, direction='in')
        ax.set_aspect('equal', adjustable='datalim')

    # ── Bottom row: AUCT-like metric per segment ──
    # Create pseudo-trial data by computing metrics over sliding windows
    window = max(10, n // 30)
    stride = max(3, window // 3)

    # Compute running AUCT metric
    trial_metrics = []
    positions = []
    for start in range(0, n - window, stride):
        seg_x = px_l_cm[start:start+window]
        seg_y = py_l_cm[start:start+window]
        area = np_trapz(np.abs(seg_y), seg_x)
        trial_metrics.append(abs(area))
        positions.append(start + window//2)

    trial_metrics = np.array(trial_metrics)
    positions = np.array(positions)

    # Split into 3 phases for bottom row
    n_trials = len(trial_metrics)
    third = n_trials // 3

    phase_names = ['Baseline', 'Swing Phase', 'Post Swing']
    ranges = [(0, third), (third, 2*third), (2*third, n_trials)]

    for j, (pname, (rs, re)) in enumerate(zip(phase_names, ranges)):
        col_start = [0, 2, 3][j]
        col_span = [2, 1, 2][j]

        if col_span == 2:
            ax = fig.add_subplot(gs[1, col_start:col_start+col_span])
        else:
            ax = fig.add_subplot(gs[1, col_start])

        trials = np.arange(re - rs)
        data = trial_metrics[rs:re]

        # Scatter with red line (like paper)
        ax.plot(trials, data, 'r-', lw=0.6, alpha=0.5)
        ax.plot(trials, data, 'k.', markersize=3, alpha=0.7)

        # Trend curve (polynomial fit) in blue
        if len(trials) > 3:
            z = np.polyfit(trials, data, min(3, len(trials)-1))
            p = np.poly1d(z)
            t_smooth = np.linspace(trials[0], trials[-1], 100)
            ax.plot(t_smooth, p(t_smooth), 'b-', lw=2.0, label='Trend')

            # Dashed reference line (mean)
            ax.axhline(np.mean(data), color='m', ls='--', lw=1.0, alpha=0.7, label='Reference')

        ax.set_title(pname, fontsize=9, fontweight='bold')
        ax.set_xlabel('Trial Number', fontsize=8)
        if j == 0:
            ax.set_ylabel('AUCT (cm²)', fontsize=8)
        ax.tick_params(labelsize=7, direction='in')
        if j == 0:
            ax.legend(fontsize=6, loc='best')

    plt.savefig(str(OUTPUT_DIR / "fig7_trial_domain.png"), bbox_inches='tight', dpi=300)
    plt.savefig(str(OUTPUT_DIR / "fig7_trial_domain.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] fig7_trial_domain.png/pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Bilateral comparison (clean research style)
# ═══════════════════════════════════════════════════════════════════════
def plot_bilateral_research(px_l, py_l, px_r, py_r, phase_deg):
    fig, ax = plt.subplots(figsize=(7, 5))

    px_l_cm, py_l_cm = to_cm(px_l), to_cm(py_l)
    px_r_cm, py_r_cm = to_cm(px_r), to_cm(py_r)

    ax.plot(px_l_cm, py_l_cm, 'r-', lw=1.5, label='Left Ankle')
    ax.plot(px_r_cm, py_r_cm, 'b-', lw=1.5, label='Right Ankle')

    # Mark start/end
    for px, py, clr, mk in [(px_l_cm,py_l_cm,'red','o'),(px_r_cm,py_r_cm,'blue','s')]:
        ax.scatter(px[0], py[0], color=clr, s=60, marker=mk, zorder=5, edgecolors='k', lw=0.8)
        ax.scatter(px[-1], py[-1], color=clr, s=60, marker='X', zorder=5, edgecolors='k', lw=0.8)

    ax.set_xlabel('X axis (cm)')
    ax.set_ylabel('Y axis (cm)')
    ax.set_title(f'Bilateral Gait Trajectory Comparison (Phase: {phase_deg:.1f}°)')
    ax.legend(loc='best', framealpha=0.9)
    ax.tick_params(direction='in', top=True, right=True)
    ax.set_aspect('equal', adjustable='datalim')

    plt.savefig(str(OUTPUT_DIR / "fig_bilateral.png"), bbox_inches='tight', dpi=300)
    plt.savefig(str(OUTPUT_DIR / "fig_bilateral.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] fig_bilateral.png/pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Sensor signals (acceleration & gyroscope)
# ═══════════════════════════════════════════════════════════════════════
def plot_sensor_signals(left_df, right_df):
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    t_l = left_df["timestamp"].values
    t_r = right_df["timestamp"].values

    accel_cols = [("ax","$a_x$"), ("ay","$a_y$"), ("az","$a_z$")]
    gyro_cols = [("gx","$\\omega_x$"), ("gy","$\\omega_y$"), ("gz","$\\omega_z$")]

    for i, (col, label) in enumerate(accel_cols):
        ax = axes[i, 0]
        ax.plot(t_l, left_df[col], 'r-', lw=0.8, alpha=0.8, label='Left')
        ax.plot(t_r, right_df[col], 'b-', lw=0.8, alpha=0.8, label='Right')
        ax.set_ylabel(f'{label} (m/s²)', fontsize=9)
        ax.tick_params(direction='in', labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right')
            ax.set_title('Accelerometer', fontsize=10, fontweight='bold')

    for i, (col, label) in enumerate(gyro_cols):
        ax = axes[i, 1]
        ax.plot(t_l, left_df[col], 'r-', lw=0.8, alpha=0.8, label='Left')
        ax.plot(t_r, right_df[col], 'b-', lw=0.8, alpha=0.8, label='Right')
        ax.set_ylabel(f'{label} (rad/s)', fontsize=9)
        ax.tick_params(direction='in', labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right')
            ax.set_title('Gyroscope', fontsize=10, fontweight='bold')

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "fig_sensor_signals.png"), bbox_inches='tight', dpi=300)
    plt.savefig(str(OUTPUT_DIR / "fig_sensor_signals.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] fig_sensor_signals.png/pdf")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  RESEARCH PAPER STYLE GAIT PLOTS")
    print("=" * 60)

    left_df  = load_and_smooth(LEFT_CSV)
    right_df = load_and_smooth(RIGHT_CSV)
    print(f"Loaded: Left={len(left_df)}, Right={len(right_df)} samples")

    t_l, lpx, lpy = estimate_position(left_df)
    t_r, rpx, rpy = estimate_position(right_df)
    print(f"Position estimated. Ranges (m):")
    print(f"  Left  X:[{lpx.min():.4f},{lpx.max():.4f}] Y:[{lpy.min():.4f},{lpy.max():.4f}]")
    print(f"  Right X:[{rpx.min():.4f},{rpx.max():.4f}] Y:[{rpy.min():.4f},{rpy.max():.4f}]")

    # Phase analysis
    ls = signal.detrend(left_df["gy"].values)
    rs = signal.detrend(right_df["gy"].values)
    ls, rs = ls/(np.std(ls)+1e-8), rs/(np.std(rs)+1e-8)
    corr = signal.correlate(ls, rs, mode="full")
    lags = signal.correlation_lags(len(ls), len(rs), mode="full")
    best_lag = lags[np.argmax(corr)]
    auto_c = signal.correlate(ls, ls, mode="full")
    mid = len(auto_c)//2
    pks, _ = find_peaks(auto_c[mid:], distance=20)
    cycle_s = pks[0] if len(pks) > 0 else len(ls)//3
    phase_deg = (best_lag / cycle_s * 360) % 360
    print(f"Phase shift: {phase_deg:.1f}°")

    print("\nGenerating paper-style figures...")
    plot_fig5_style(lpx, lpy, rpx, rpy, left_df=left_df)
    plot_fig7_style(lpx, lpy, rpx, rpy)
    plot_bilateral_research(lpx, lpy, rpx, rpy, phase_deg)
    plot_sensor_signals(left_df, right_df)

    print(f"\nAll plots saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
