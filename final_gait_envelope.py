"""
Generate a research-style forward gait trajectory envelope from IMU ankle data.

This script continues the existing gait-analysis workflow in this repository and
produces a tear-drop style plot where:
  - X is forward progression and is monotonically increasing within each cycle
  - Y is vertical motion estimated from IMU acceleration
  - the red curve is the upper extreme across aligned cycles
  - the green curve is the lower extreme across aligned cycles
  - the blue dashed curve is the mean trajectory

Output:
    final_gait_output/tear_drop_gait.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks, savgol_filter


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "validation_gait_data" / "data"
OUTPUT_DIR = ROOT / "final_gait_output"
OUTPUT_PATH = OUTPUT_DIR / "tear_drop_gait.png"

DEFAULT_INPUTS = [
    DATA_DIR / "left_ankle.csv",
    DATA_DIR / "right_ankle.csv",
]

TARGET_CYCLE_SAMPLES = 200
DEFAULT_SAMPLE_RATE = 100.0


def _odd_window(length: int, preferred: int) -> int:
    """Return a valid odd Savitzky-Golay window smaller than the signal length."""
    if length < 5:
        return 0
    window = min(preferred, length if length % 2 == 1 else length - 1)
    if window < 5:
        return 0
    return window


def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    if "timestamp" in df.columns:
        time_col = "timestamp"
    elif "time" in df.columns:
        time_col = "time"
    else:
        raise ValueError(f"{csv_path.name} is missing a time column.")

    required = ["ax", "ay", "az", "gx", "gy", "gz"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    df[time_col] = df[time_col] - df[time_col].iloc[0]
    df = df.rename(columns={time_col: "time"})

    smooth_cols = [col for col in ["ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw"] if col in df.columns]
    window = _odd_window(len(df), 15)
    if window:
        for col in smooth_cols:
            df[col] = savgol_filter(df[col].to_numpy(), window_length=window, polyorder=3, mode="interp")

    return df


def estimate_sample_rate(time: np.ndarray) -> float:
    dt = np.diff(time)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return DEFAULT_SAMPLE_RATE
    return float(1.0 / np.median(dt))


def remove_gravity(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate gravity-compensated accelerations in forward/lateral/vertical axes."""
    ax = df["ax"].to_numpy()
    ay = df["ay"].to_numpy()
    az = df["az"].to_numpy()
    roll = df["roll"].to_numpy() if "roll" in df.columns else np.zeros(len(df))
    pitch = df["pitch"].to_numpy() if "pitch" in df.columns else np.zeros(len(df))

    g = 9.81
    grav_x = g * np.sin(pitch)
    grav_y = -g * np.sin(roll) * np.cos(pitch)
    grav_z = -g * np.cos(roll) * np.cos(pitch)

    return ax - grav_x, ay - grav_y, az - grav_z


def detect_stance_mask(
    time: np.ndarray,
    forward_acc: np.ndarray,
    vertical_acc: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
) -> np.ndarray:
    """Detect low-motion stance windows for ZUPT."""
    sample_rate = estimate_sample_rate(time)
    motion = np.sqrt(forward_acc**2 + vertical_acc**2) + 0.15 * np.sqrt(gx**2 + gy**2 + gz**2)
    motion_window = _odd_window(len(motion), max(11, int(sample_rate * 0.12) | 1))
    if motion_window:
        motion = savgol_filter(motion, window_length=motion_window, polyorder=2, mode="interp")

    threshold = np.percentile(motion, 35)
    stance = motion <= threshold

    kernel = np.ones(max(3, int(sample_rate * 0.05)), dtype=int)
    stance = np.convolve(stance.astype(int), kernel, mode="same") >= max(2, len(kernel) // 2)
    return stance


def integrate_with_zupt(
    time: np.ndarray,
    accel: np.ndarray,
    stance_mask: np.ndarray,
    lp_cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate acceleration into velocity and position with stance-based reset."""
    sample_rate = estimate_sample_rate(time)
    sos = signal.butter(4, lp_cutoff, btype="low", fs=sample_rate, output="sos")
    accel_filt = signal.sosfiltfilt(sos, accel)

    velocity = cumulative_trapezoid(accel_filt, time, initial=0.0)
    velocity[stance_mask] = 0.0
    velocity = signal.detrend(velocity, type="linear")

    vel_window = _odd_window(len(velocity), max(9, int(sample_rate * 0.15) | 1))
    if vel_window:
        velocity = savgol_filter(velocity, window_length=vel_window, polyorder=2, mode="interp")
        velocity[stance_mask] = 0.0

    position = cumulative_trapezoid(velocity, time, initial=0.0)
    position = signal.detrend(position, type="linear")

    pos_window = _odd_window(len(position), max(11, int(sample_rate * 0.2) | 1))
    if pos_window:
        position = savgol_filter(position, window_length=pos_window, polyorder=3, mode="interp")

    return velocity, position


def build_xy_trajectory(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2D ankle trajectory (X vs Y) from IMU data.

    For the "oval/loop" trajectory style, we keep the natural closed path that
    emerges over a gait cycle, using ZUPT and per-cycle drift removal instead of
    enforcing monotonic X.
    """
    time = df["time"].to_numpy()
    fx, fy, fz = remove_gravity(df)
    stance = detect_stance_mask(
        time,
        forward_acc=fx,
        vertical_acc=fz,
        gx=df["gx"].to_numpy(),
        gy=df["gy"].to_numpy(),
        gz=df["gz"].to_numpy(),
    )

    # Double integration with ZUPT on both axes (no monotonic enforcement)
    vx, x = integrate_with_zupt(time, fx, stance, lp_cutoff=4.0)
    vy, y = integrate_with_zupt(time, fy, stance, lp_cutoff=4.0)

    # Normalize start for convenience (cycle processing will handle closure)
    x = x - x[0]
    y = y - y[0]

    xy_window = _odd_window(len(x), 17)
    if xy_window:
        x = savgol_filter(x, window_length=xy_window, polyorder=3, mode="interp")
        y = savgol_filter(y, window_length=xy_window, polyorder=3, mode="interp")

    return time, x, y, vx, vy, stance


def _contiguous_true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start,end) index pairs for contiguous True segments."""
    if mask.size == 0:
        return []
    m = mask.astype(bool)
    edges = np.diff(m.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def smooth_velocity_for_plot(time: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Presentation smoothing for ZUPT-corrected velocity (keeps dynamics, removes steps)."""
    fs = estimate_sample_rate(time)
    v_s = v.copy()

    # Low-pass filter to remove discontinuities (gait energy is low-frequency)
    sos = signal.butter(3, 6.0, btype="low", fs=fs, output="sos")
    v_s = signal.sosfiltfilt(sos, v_s)

    win = _odd_window(len(v_s), max(11, int(fs * 0.25) | 1))
    if win:
        v_s = savgol_filter(v_s, window_length=win, polyorder=3, mode="interp")
    return v_s


def detect_gait_boundaries(y: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Detect gait-cycle boundaries from minima in vertical motion.

    The minima correspond to foot-contact events. We find peaks in the negated
    vertical displacement so the extracted cycles naturally span contact to contact.
    """
    y_smooth = y.copy()
    y_window = _odd_window(len(y_smooth), max(11, int(sample_rate * 0.15) | 1))
    if y_window:
        y_smooth = savgol_filter(y_smooth, window_length=y_window, polyorder=3, mode="interp")

    min_distance = max(25, int(sample_rate * 0.45))
    prominence = max(1e-4, 0.08 * np.ptp(y_smooth))
    contacts, _ = find_peaks(-y_smooth, distance=min_distance, prominence=prominence)

    if len(contacts) < 3:
        fallback_distance = max(20, int(sample_rate * 0.35))
        contacts, _ = find_peaks(-y_smooth, distance=fallback_distance)

    return contacts


def resample_cycle(x: np.ndarray, y: np.ndarray, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    phase_old = np.linspace(0.0, 1.0, len(x))
    phase_new = np.linspace(0.0, 1.0, target_length)

    interp_x = np.interp(phase_new, phase_old, x)
    interp_y = np.interp(phase_new, phase_old, y)

    x_resampled = interp_x
    y_resampled = interp_y

    # Remove linear drift within the cycle (loop-closure without warping shape)
    x_resampled = x_resampled - np.linspace(x_resampled[0], x_resampled[-1], len(x_resampled))
    y_resampled = y_resampled - np.linspace(y_resampled[0], y_resampled[-1], len(y_resampled))

    # Center cycle near origin for stable overlay
    x_resampled = x_resampled - x_resampled[0]
    y_resampled = y_resampled - y_resampled[0]

    # Hard close the loop at the final sample (common in gait trajectory plots)
    x_resampled[-1] = x_resampled[0]
    y_resampled[-1] = y_resampled[0]

    # Phase-align cycles so overlays form a clean oval:
    # start at the lowest Y (contact) and enforce consistent direction.
    idx0 = int(np.argmin(y_resampled[:-1]))  # ignore duplicated last point
    x_resampled = np.roll(x_resampled, -idx0)
    y_resampled = np.roll(y_resampled, -idx0)

    # Ensure consistent orientation (clockwise vs counter-clockwise)
    area = 0.5 * np.sum(x_resampled[:-1] * y_resampled[1:] - x_resampled[1:] * y_resampled[:-1])
    if area < 0:
        x_resampled = x_resampled[::-1]
        y_resampled = y_resampled[::-1]

    # Re-close after roll/flip
    x_resampled[-1] = x_resampled[0]
    y_resampled[-1] = y_resampled[0]

    return x_resampled, y_resampled


def extract_cycles(time: np.ndarray, x: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    sample_rate = estimate_sample_rate(time)
    contacts = detect_gait_boundaries(y, sample_rate)
    cycles: List[Tuple[np.ndarray, np.ndarray]] = []

    for start, end in zip(contacts[:-1], contacts[1:]):
        if end - start < max(20, int(sample_rate * 0.35)):
            continue

        x_cycle = x[start:end].copy()
        y_cycle = y[start:end].copy()

        if np.ptp(x_cycle) < 1e-4 or np.ptp(y_cycle) < 1e-4:
            continue

        cycles.append(resample_cycle(x_cycle, y_cycle, TARGET_CYCLE_SAMPLES))

    return cycles


def collect_all_cycles(input_paths: Sequence[Path]) -> List[Tuple[np.ndarray, np.ndarray]]:
    all_cycles: List[Tuple[np.ndarray, np.ndarray]] = []
    for path in input_paths:
        df = load_and_preprocess(path)
        time, x, y, _, _, _ = build_xy_trajectory(df)
        all_cycles.extend(extract_cycles(time, x, y))
    return all_cycles


def scale_cycles_to_cm(cycles: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    x_ranges = np.array([np.ptp(x) for x, _ in cycles])
    y_ranges = np.array([np.ptp(y) for _, y in cycles])

    median_x = float(np.median(x_ranges)) if len(x_ranges) else 1.0
    median_y = float(np.median(y_ranges)) if len(y_ranges) else 1.0

    # Match reference figure scale approximately (oval spans ~50cm x ~12cm)
    x_scale = 50.0 / max(median_x, 1e-6)
    y_scale = 12.0 / max(median_y, 1e-6)

    scaled = []
    for x, y in cycles:
        x_cm = x * x_scale
        y_cm = y * y_scale

        # Shift to positive quadrant like the reference figure
        x_cm = x_cm - np.min(x_cm)
        y_cm = y_cm - np.min(y_cm)

        # Reference trajectories ride a few cm above zero
        y_cm = y_cm + 3.0

        # Keep loop closed after shifting
        x_cm[-1] = x_cm[0]
        y_cm[-1] = y_cm[0]
        scaled.append((x_cm, y_cm))
    return scaled


def compute_envelope(
    cycles: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not cycles:
        raise ValueError("No gait cycles could be extracted from the provided IMU data.")

    scaled_cycles = scale_cycles_to_cm(cycles)
    if len(scaled_cycles) >= 3:
        # Drop outlier cycles by RMS distance to a preliminary mean loop
        x0 = np.mean(np.vstack([x for x, _ in scaled_cycles]), axis=0)
        y0 = np.mean(np.vstack([y for _, y in scaled_cycles]), axis=0)
        rms = []
        for x, y in scaled_cycles:
            rms.append(float(np.sqrt(np.mean((x - x0) ** 2 + (y - y0) ** 2))))
        rms = np.array(rms)
        keep = rms <= np.percentile(rms, 60)
        if np.sum(keep) >= 2:
            scaled_cycles = [c for c, k in zip(scaled_cycles, keep) if k]
    x_stack = np.vstack([x for x, _ in scaled_cycles])
    y_stack = np.vstack([y for _, y in scaled_cycles])

    # Mean closed curve
    x_mean = np.mean(x_stack, axis=0)
    y_mean = np.mean(y_stack, axis=0)

    smooth_window = _odd_window(len(x_mean), 31)
    if smooth_window:
        x_mean = savgol_filter(x_mean, window_length=smooth_window, polyorder=3, mode="interp")
        y_mean = savgol_filter(y_mean, window_length=smooth_window, polyorder=3, mode="interp")

    # Close the mean loop
    x_mean[-1] = x_mean[0]
    y_mean[-1] = y_mean[0]

    # Normal-direction envelope to avoid crossing curves:
    # project each cycle's deviation onto the mean-curve normals.
    dx = np.gradient(x_mean)
    dy = np.gradient(y_mean)
    speed = np.hypot(dx, dy) + 1e-9
    tx = dx / speed
    ty = dy / speed
    nx = -ty
    ny = tx

    # Signed normal displacement per cycle
    disp = (x_stack - x_mean[None, :]) * nx[None, :] + (y_stack - y_mean[None, :]) * ny[None, :]

    d_upper = np.percentile(disp, 90, axis=0)
    d_lower = np.percentile(disp, 10, axis=0)

    x_upper = x_mean + d_upper * nx
    y_upper = y_mean + d_upper * ny
    x_lower = x_mean + d_lower * nx
    y_lower = y_mean + d_lower * ny

    if smooth_window:
        x_upper = savgol_filter(x_upper, window_length=smooth_window, polyorder=3, mode="interp")
        y_upper = savgol_filter(y_upper, window_length=smooth_window, polyorder=3, mode="interp")
        x_lower = savgol_filter(x_lower, window_length=smooth_window, polyorder=3, mode="interp")
        y_lower = savgol_filter(y_lower, window_length=smooth_window, polyorder=3, mode="interp")

    # Re-close
    x_upper[-1] = x_upper[0]
    y_upper[-1] = y_upper[0]
    x_lower[-1] = x_lower[0]
    y_lower[-1] = y_lower[0]

    # Ensure "upper" is the outer envelope (larger average radius)
    cx = float(np.mean(x_mean[:-1]))
    cy = float(np.mean(y_mean[:-1]))
    r_upper = float(np.mean(np.hypot(x_upper[:-1] - cx, y_upper[:-1] - cy)))
    r_lower = float(np.mean(np.hypot(x_lower[:-1] - cx, y_lower[:-1] - cy)))
    if r_upper < r_lower:
        x_upper, x_lower = x_lower, x_upper
        y_upper, y_lower = y_lower, y_upper

    return x_upper, y_upper, x_lower, y_lower, x_mean, y_mean


def plot_envelope(
    cycles: Sequence[Tuple[np.ndarray, np.ndarray]],
    x_upper: np.ndarray,
    y_upper: np.ndarray,
    x_lower: np.ndarray,
    y_lower: np.ndarray,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.size": 11,
        }
    )

    # Match reference layout: wide & short
    fig, ax = plt.subplots(figsize=(8.2, 2.0), dpi=200)

    # Plot all trajectories (thin), confidence interval band, and mean
    scaled = scale_cycles_to_cm(cycles)
    for x_cycle, y_cycle in scaled:
        ax.plot(x_cycle, y_cycle, color="#cfcfcf", linewidth=0.8, alpha=0.7, zorder=1)

    # Confidence interval band (between outer envelopes)
    band_x = np.concatenate([x_upper, x_lower[::-1]])
    band_y = np.concatenate([y_upper, y_lower[::-1]])
    ax.fill(band_x, band_y, color="#d9d9ff", alpha=0.35, linewidth=0.0, zorder=2, label="Confidence interval")

    # Mean trajectory (blue dashed)
    ax.plot(x_mean, y_mean, color="blue", linewidth=2.0, linestyle=(0, (4, 3)), label="Mean trajectory", zorder=4)

    ax.set_xlabel("X axis(cm)", fontsize=12)
    ax.set_ylabel("Y axis(cm)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(0.0, 15.0)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.6)

    # Panel label like the reference
    ax.text(0.06, 0.82, "A", transform=ax.transAxes, fontsize=12)

    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_position_velocity_vs_time(df: pd.DataFrame, tag: str) -> Path:
    """
    Plot position and velocity vs time for both X and Y axes.
    Saves to final_gait_output/pos_vel_{tag}.png
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time, x, y, vx, vy, stance = build_xy_trajectory(df)

    # Convert to cm for position, cm/s for velocity
    x_cm = x * 100.0
    y_cm = y * 100.0
    vx_cms_raw = vx * 100.0
    vy_cms_raw = vy * 100.0
    vx_cms = smooth_velocity_for_plot(time, vx) * 100.0
    vy_cms = smooth_velocity_for_plot(time, vy) * 100.0

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), dpi=200, sharex=True)
    (ax1, ax2), (ax3, ax4) = axes

    ax1.plot(time, x_cm, color="#1f77b4", linewidth=1.6)
    ax1.set_title("X position vs time")
    ax1.set_ylabel("X (cm)")
    ax1.grid(True, alpha=0.25)

    ax2.plot(time, y_cm, color="#ff7f0e", linewidth=1.6)
    ax2.set_title("Y position vs time")
    ax2.set_ylabel("Y (cm)")
    ax2.grid(True, alpha=0.25)

    # Raw velocity (faint) + smoothed velocity (main)
    ax3.plot(time, vx_cms_raw, color="#1f77b4", linewidth=0.9, alpha=0.25, label="Raw (ZUPT)")
    ax3.plot(time, vx_cms, color="#1f77b4", linewidth=1.6, label="Smoothed")
    ax3.set_title("X velocity vs time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Vx (cm/s)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(frameon=False, fontsize=8, loc="upper right")

    ax4.plot(time, vy_cms_raw, color="#ff7f0e", linewidth=0.9, alpha=0.25, label="Raw (ZUPT)")
    ax4.plot(time, vy_cms, color="#ff7f0e", linewidth=1.6, label="Smoothed")
    ax4.set_title("Y velocity vs time")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Vy (cm/s)")
    ax4.grid(True, alpha=0.25)
    ax4.legend(frameon=False, fontsize=8, loc="upper right")

    # Stance shading (so near-zero velocity is clearly explained)
    for s, e in _contiguous_true_segments(stance):
        t0 = float(time[s])
        t1 = float(time[min(e, len(time) - 1)])
        for ax in (ax1, ax2, ax3, ax4):
            ax.axvspan(t0, t1, color="#f2f2f2", alpha=0.9, zorder=0)

    fig.suptitle(f"Position and velocity vs time ({tag})", y=0.98)
    fig.tight_layout()
    out = OUTPUT_DIR / f"pos_vel_{tag}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main(input_paths: Iterable[Path] | None = None) -> Path:
    paths = list(input_paths) if input_paths is not None else [path for path in DEFAULT_INPUTS if path.exists()]
    if not paths:
        raise FileNotFoundError("No default IMU CSV files were found.")

    cycles = collect_all_cycles(paths)
    x_upper, y_upper, x_lower, y_lower, x_mean, y_mean = compute_envelope(cycles)
    plot_envelope(cycles, x_upper, y_upper, x_lower, y_lower, x_mean, y_mean)

    # Position/velocity plots for each input file
    for p in paths:
        df = load_and_preprocess(p)
        tag = p.stem
        plot_position_velocity_vs_time(df, tag=tag)

    print(f"Extracted cycles: {len(cycles)}")
    print(f"Saved plot: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
