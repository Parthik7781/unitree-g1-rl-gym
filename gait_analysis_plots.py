"""
================================================================================
GAIT ANALYSIS — RESEARCH-STYLE PLOTS
================================================================================

Loads left_ankle.csv and right_ankle.csv, performs:
  1. Noise smoothing (Savitzky-Golay)
  2. Position estimation via double integration + ZUPT drift correction
  3. Phase analysis (cross-correlation + peak detection)
  4. Generates 7 publication-ready plots

Dependencies: pandas, numpy, matplotlib, scipy
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import signal
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path

# Handle scipy version compatibility
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "validation_gait_data" / "data"
OUTPUT_DIR = Path(__file__).parent / "validation_gait_data" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT_CSV   = DATA_DIR / "left_ankle.csv"
RIGHT_CSV  = DATA_DIR / "right_ankle.csv"

SAMPLE_RATE = 100.0          # Hz  (dt = 0.01 s)
DT          = 1.0 / SAMPLE_RATE

# Plot palette
CLR_LEFT  = "#1f77b4"        # blue
CLR_RIGHT = "#ff7f0e"        # orange
DPI       = 200


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & SMOOTH
# ─────────────────────────────────────────────────────────────────────────────
def load_and_smooth(csv_path: Path) -> pd.DataFrame:
    """Load CSV, normalise time to start from 0, apply Savitzky-Golay smoothing."""
    df = pd.read_csv(csv_path)

    # Normalise time
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]

    # Light Savitzky-Golay smoothing on accel & gyro columns
    cols_to_smooth = ["ax", "ay", "az", "gx", "gy", "gz"]
    win = 11          # window length (must be odd)
    poly = 3          # polynomial order
    for c in cols_to_smooth:
        if c in df.columns:
            df[c] = savgol_filter(df[c].values, window_length=win, polyorder=poly)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — POSITION ESTIMATION  (double integration + ZUPT)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_position(df: pd.DataFrame):
    """
    Integrate acceleration → velocity → position.
    Apply Zero Velocity Update (ZUPT) + high-pass drift correction.
    NO artificial loop closing.
    """
    time  = df["timestamp"].values
    ax    = df["ax"].values.copy()
    ay    = df["ay"].values.copy()
    az    = df["az"].values.copy()
    roll  = df["roll"].values if "roll" in df.columns else np.zeros(len(df))
    pitch = df["pitch"].values if "pitch" in df.columns else np.zeros(len(df))

    # --- 2a. Remove gravity component ---------------------------------
    g = 9.81
    grav_x =  g * np.sin(pitch)
    grav_y = -g * np.sin(roll) * np.cos(pitch)
    grav_z =  g * np.cos(roll) * np.cos(pitch)

    ax_clean = ax - grav_x
    ay_clean = ay - grav_y

    # --- 2b. Band-pass filter (keep gait frequencies, reject DC drift) --
    #     Typical gait energy: 0.5 – 6 Hz
    sos_bp = signal.butter(4, [0.4, 8.0], btype="band", fs=SAMPLE_RATE, output="sos")
    ax_filt = signal.sosfiltfilt(sos_bp, ax_clean)
    ay_filt = signal.sosfiltfilt(sos_bp, ay_clean)

    # --- 2c. Detect stance phases (for ZUPT) ----------------------------
    #     Stance ≈ low acceleration magnitude windows after heel-strike peaks
    accel_mag = np.sqrt(ax_filt**2 + ay_filt**2 + (az - grav_z)**2)
    accel_mag_smooth = savgol_filter(accel_mag, 15, 3)

    # Adaptive threshold for stance
    stance_thresh = np.percentile(accel_mag_smooth, 30)
    is_stance = accel_mag_smooth < stance_thresh

    # --- 2d. First integration: accel → velocity -----------------------
    vx = cumtrapz(ax_filt, dx=DT, initial=0)
    vy = cumtrapz(ay_filt, dx=DT, initial=0)

    # ZUPT: force velocity → 0 during stance windows
    vx[is_stance] = 0.0
    vy[is_stance] = 0.0

    # Linear drift removal on velocity (high-pass effect)
    vx = signal.detrend(vx, type="linear")
    vy = signal.detrend(vy, type="linear")

    # Smooth velocity
    vx = savgol_filter(vx, 15, 3)
    vy = savgol_filter(vy, 15, 3)

    # --- 2e. Second integration: velocity → position -------------------
    px = cumtrapz(vx, dx=DT, initial=0)
    py = cumtrapz(vy, dx=DT, initial=0)

    # High-pass drift correction on position (remove slow wander)
    sos_hp = signal.butter(3, 0.3, btype="high", fs=SAMPLE_RATE, output="sos")
    px = signal.sosfiltfilt(sos_hp, px)
    py = signal.sosfiltfilt(sos_hp, py)

    # Final light smoothing
    px = savgol_filter(px, 21, 3)
    py = savgol_filter(py, 21, 3)

    return time, px, py


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PHASE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def compute_phase_shift(left_signal: np.ndarray, right_signal: np.ndarray):
    """
    Compute phase difference between left and right using:
      • Cross-correlation (time-lag approach)
      • Peak detection
    Returns dict with lag_samples, phase_deg, etc.
    """
    # Normalise
    ls = signal.detrend(left_signal)
    rs = signal.detrend(right_signal)
    ls = ls / (np.std(ls) + 1e-8)
    rs = rs / (np.std(rs) + 1e-8)

    # Cross-correlation
    corr = signal.correlate(ls, rs, mode="full")
    lags = signal.correlation_lags(len(ls), len(rs), mode="full")
    best_lag = lags[np.argmax(corr)]

    # Estimate gait cycle length from auto-correlation peaks
    auto_corr = signal.correlate(ls, ls, mode="full")
    auto_lags = signal.correlation_lags(len(ls), len(ls), mode="full")
    mid = len(auto_lags) // 2
    auto_pos = auto_corr[mid:]          # positive lags only
    peaks, _ = find_peaks(auto_pos, distance=20)
    if len(peaks) > 0:
        cycle_samples = peaks[0]        # first recurrence = cycle length
    else:
        cycle_samples = len(ls) // 3    # fallback

    phase_deg = (best_lag / cycle_samples) * 360.0
    # Normalise to [0, 360)
    phase_deg = phase_deg % 360

    return {
        "lag_samples":   best_lag,
        "cycle_samples": cycle_samples,
        "phase_deg":     phase_deg,
        "correlation":   corr,
        "lags":          lags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Apply consistent research styling to an axes."""
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


# ── (1) Acceleration Plots ───────────────────────────────────────────────────
def plot_acceleration(left_df, right_df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=DPI, sharex=True)
    t_l = left_df["timestamp"].values
    t_r = right_df["timestamp"].values

    for i, (col, label) in enumerate([("ax", "Ax"), ("ay", "Ay"), ("az", "Az")]):
        axes[i].plot(t_l, left_df[col],  color=CLR_LEFT,  lw=1.5, alpha=0.85, label="Left")
        axes[i].plot(t_r, right_df[col], color=CLR_RIGHT, lw=1.5, alpha=0.85, label="Right")
        _style_ax(axes[i], ylabel=f"{label} (m/s²)",
                  title=f"{label} — Acceleration vs Time")
        axes[i].legend(fontsize=10, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / "acceleration_signals.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── (2) Gyroscope Phase Plot ─────────────────────────────────────────────────
def plot_gyroscope_phase(left_df, right_df, phase_info):
    fig, ax = plt.subplots(figsize=(14, 5), dpi=DPI)
    t_l = left_df["timestamp"].values
    t_r = right_df["timestamp"].values

    ax.plot(t_l, left_df["gy"],  color=CLR_LEFT,  lw=1.8, alpha=0.85, label="Left (Gy)")
    ax.plot(t_r, right_df["gy"], color=CLR_RIGHT, lw=1.8, alpha=0.85, label="Right (Gy)")

    # Annotate phase shift
    phase_text = f"Phase shift ≈ {phase_info['phase_deg']:.1f}°"
    ax.text(0.02, 0.95, phase_text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

    _style_ax(ax, xlabel="Time (s)", ylabel="Angular velocity (rad/s)",
              title="Gyroscope (Gy) — Phase Shift Between Left & Right")
    ax.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    path = OUTPUT_DIR / "gyroscope_phase_signals.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── (3) Position vs Time ─────────────────────────────────────────────────────
def plot_position_vs_time(time, px, py, leg: str):
    clr = CLR_LEFT if leg == "left" else CLR_RIGHT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), dpi=DPI, sharex=True)

    ax1.plot(time, px, color=clr, lw=2, alpha=0.85)
    ax1.fill_between(time, px, alpha=0.15, color=clr)
    _style_ax(ax1, ylabel="X Position (m)",
              title=f"{leg.capitalize()} Ankle — X Position vs Time")

    ax2.plot(time, py, color=clr, lw=2, alpha=0.85)
    ax2.fill_between(time, py, alpha=0.15, color=clr)
    _style_ax(ax2, xlabel="Time (s)", ylabel="Y Position (m)",
              title=f"{leg.capitalize()} Ankle — Y Position vs Time")

    plt.tight_layout()
    path = OUTPUT_DIR / f"position_vs_time_{leg}.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── (4) Tear-Drop Trajectory (single leg) ────────────────────────────────────
def plot_tear_drop(px, py, leg: str):
    clr = CLR_LEFT if leg == "left" else CLR_RIGHT
    fig, ax = plt.subplots(figsize=(9, 9), dpi=DPI)

    ax.plot(px, py, color=clr, lw=2.5, alpha=0.8, label="Foot path")

    # Start (circle) & End (square)
    ax.scatter(px[0],  py[0],  color="green", s=180, marker="o", zorder=5,
               edgecolors="black", linewidths=1.5, label="Start")
    ax.scatter(px[-1], py[-1], color="red",   s=180, marker="s", zorder=5,
               edgecolors="black", linewidths=1.5, label="End")

    # Directional arrows
    step = max(1, len(px) // 10)
    for i in range(0, len(px) - step, step):
        dx = px[i + step] - px[i]
        dy = py[i + step] - py[i]
        mag = np.hypot(dx, dy)
        if mag > 1e-5:
            ax.annotate("", xy=(px[i] + dx * 0.55, py[i] + dy * 0.55),
                         xytext=(px[i], py[i]),
                         arrowprops=dict(arrowstyle="->", color=clr, lw=1.5, alpha=0.5))

    _style_ax(ax, xlabel="X Position (m)", ylabel="Y Position (m)",
              title=f"{leg.capitalize()} Ankle — Tear-Drop Trajectory")
    ax.legend(fontsize=11, loc="best")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    path = OUTPUT_DIR / f"tear_drop_{leg}_trajectory.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── (5) Bilateral Gait Comparison ────────────────────────────────────────────
def plot_bilateral_comparison(lpx, lpy, rpx, rpy, phase_info):
    fig, ax = plt.subplots(figsize=(11, 9), dpi=DPI)

    ax.plot(lpx, lpy, color=CLR_LEFT,  lw=2.5, alpha=0.8, label="Left Ankle")
    ax.plot(rpx, rpy, color=CLR_RIGHT, lw=2.5, alpha=0.8, label="Right Ankle")

    # Start & End markers
    for px, py, clr, tag in [
        (lpx, lpy, CLR_LEFT,  "Left"),
        (rpx, rpy, CLR_RIGHT, "Right"),
    ]:
        ax.scatter(px[0],  py[0],  color=clr, s=150, marker="o", zorder=5,
                   edgecolors="black", linewidths=1.3, label=f"{tag} Start")
        ax.scatter(px[-1], py[-1], color=clr, s=150, marker="s", zorder=5,
                   edgecolors="black", linewidths=1.3, label=f"{tag} End")

    # Phase annotation
    phase_text = f"Natural phase shift ≈ {phase_info['phase_deg']:.1f}°"
    ax.text(0.02, 0.97, phase_text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

    _style_ax(ax, xlabel="X Position (m)", ylabel="Y Position (m)",
              title="Bilateral Gait Comparison")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    path = OUTPUT_DIR / "bilateral_gait_comparison.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  GAIT ANALYSIS — RESEARCH-STYLE PLOTS")
    print("=" * 70)

    # ── Load & smooth ────────────────────────────────────────────────────
    print("\n[1/5] Loading and smoothing data …")
    left_df  = load_and_smooth(LEFT_CSV)
    right_df = load_and_smooth(RIGHT_CSV)
    print(f"      Left  samples : {len(left_df)}")
    print(f"      Right samples : {len(right_df)}")

    # ── Position estimation ──────────────────────────────────────────────
    print("[2/5] Estimating position (double integration + ZUPT) …")
    t_l, lpx, lpy = estimate_position(left_df)
    t_r, rpx, rpy = estimate_position(right_df)
    print(f"      Left  position range — X: [{lpx.min():.4f}, {lpx.max():.4f}]  "
          f"Y: [{lpy.min():.4f}, {lpy.max():.4f}]")
    print(f"      Right position range — X: [{rpx.min():.4f}, {rpx.max():.4f}]  "
          f"Y: [{rpy.min():.4f}, {rpy.max():.4f}]")

    # ── Phase analysis ───────────────────────────────────────────────────
    print("[3/5] Computing phase shift …")
    # Use gyroscope pitch (gy) — most indicative of gait swing
    phase_info = compute_phase_shift(left_df["gy"].values, right_df["gy"].values)
    print(f"      Cross-corr lag  : {phase_info['lag_samples']} samples")
    print(f"      Cycle length    : {phase_info['cycle_samples']} samples")
    print(f"      Phase shift     : {phase_info['phase_deg']:.1f}°")

    # ── Generate plots ───────────────────────────────────────────────────
    print("[4/5] Generating plots …")

    # (1) Acceleration
    plot_acceleration(left_df, right_df)

    # (2) Gyroscope phase
    plot_gyroscope_phase(left_df, right_df, phase_info)

    # (3) Position vs time
    plot_position_vs_time(t_l, lpx, lpy, "left")
    plot_position_vs_time(t_r, rpx, rpy, "right")

    # (4) Tear-drop trajectories
    plot_tear_drop(lpx, lpy, "left")
    plot_tear_drop(rpx, rpy, "right")

    # (5) Bilateral comparison
    plot_bilateral_comparison(lpx, lpy, rpx, rpy, phase_info)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n[5/5] Done!  All plots saved to:")
    print(f"      {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
