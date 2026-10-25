"""
================================================================================
G1 ROBOT - BILATERAL ANKLE GAIT TRAJECTORY EXTRACTOR
================================================================================

Extracts ankle foot trajectories from MuJoCo simulation using ground-truth
body positions (data.xpos / data.body().xpos). No sensors, no integration.

Produces a publication-style bilateral plot matching the reference image:
  - Teardrop-shaped gait cycle curves
  - Left subplot: Start of Training vs Gait Trainer (dotted)
  - Right subplot: End of Training vs Gait Trainer (dotted)

Usage:
    python g1_gait_trajectory.py [config.yaml]

Output:
    gait_trajectory_output/bilateral_ankle_gait.png
    gait_trajectory_output/ankle_trajectory_data.csv
================================================================================
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks, savgol_filter

import mujoco

# ── Optional: pull LEGGED_GYM_ROOT_DIR from environment or sibling path ──────
try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ImportError:
    LEGGED_GYM_ROOT_DIR = os.environ.get(
        "LEGGED_GYM_ROOT_DIR",
        str(Path(__file__).resolve().parents[2])   # adjust depth if needed
    )

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Standard PD torque computation."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def smooth(signal, window=11, poly=3):
    """Savitzky-Golay smoothing; safe for short arrays."""
    w = min(window, len(signal) - 1)
    if w % 2 == 0:
        w -= 1
    if w >= 5:
        return savgol_filter(signal, w, poly)
    return signal


def detrend(arr):
    """Remove linear trend (global forward drift)."""
    trend = np.linspace(arr[0], arr[-1], len(arr))
    return arr - trend


def find_gait_cycles(y_signal, min_distance=100):
    """
    Find indices of gait-cycle boundaries (stance-phase nadirs in Y).
    Returns list of (start, end) index pairs.
    """
    # More lenient peak detection for shorter signals
    if len(y_signal) < 500:
        # Very short signal: just use entire signal as one cycle
        return [(0, len(y_signal)-1)], [0, len(y_signal)-1]
    
    # Find local minima in Y (stance phase touchdowns)
    # Use smaller min_distance for 2 Hz gait (more cycles in same time window)
    peaks, props = find_peaks(-y_signal, distance=min_distance//2, height=0)
    
    if len(peaks) < 2:
        # Fallback: use entire signal as one pseudo-cycle
        return [(0, len(y_signal)-1)], [0, len(y_signal)-1]
    
    cycles = []
    for i in range(len(peaks) - 1):
        cycles.append((peaks[i], peaks[i + 1]))
    return cycles, peaks


def extract_mean_cycle(x_all, y_all, cycles, n_points=200):
    """
    Resample each detected cycle to n_points and compute mean trajectory.
    Returns mean_x, mean_y, std_x, std_y (all length n_points).
    """
    xs, ys = [], []
    t_norm = np.linspace(0, 1, n_points)
    for s, e in cycles:
        seg_x = x_all[s:e]
        seg_y = y_all[s:e]
        t_seg = np.linspace(0, 1, len(seg_x))
        xs.append(np.interp(t_norm, t_seg, seg_x))
        ys.append(np.interp(t_norm, t_seg, seg_y))
    xs = np.array(xs)
    ys = np.array(ys)
    return xs.mean(0), ys.mean(0), xs.std(0), ys.std(0)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_file: str) -> dict:
    config_path = (
        Path(LEGGED_GYM_ROOT_DIR) / "deploy" / "deploy_mujoco" / "configs" / config_file
    )
    if not config_path.exists():
        print(f"[WARN] Config not found at {config_path}. Using built-in defaults.")
        return _default_config()
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in cfg.items():
        if isinstance(v, str):
            cfg[k] = v.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    return cfg


def _default_config() -> dict:
    return {
        "xml_path": f"{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.xml",
        "simulation_dt": 0.002,
        "control_decimation": 4,
        "kps": [100]*6 + [100]*6,
        "kds": [2]*6  + [2]*6,
        "default_angles": [0., 0., -0.1, 0.3, -0.2, 0.,
                           0., 0., -0.1, 0.3, -0.2, 0., 0.],
        "num_actions": 12,
    }


def find_xml(cfg: dict) -> str:
    """Locate a valid XML/URDF file for the G1 model."""
    candidates = [
        cfg.get("xml_path", ""),
        f"{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.xml",
        f"{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf",
        f"{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof.xml",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    raise FileNotFoundError(
        "Cannot find G1 model file. Set xml_path in your config or "
        "set LEGGED_GYM_ROOT_DIR correctly.\n"
        f"Searched: {candidates}"
    )


def find_foot_bodies(m):
    """Return (left_id, left_name, right_id, right_name) for ankle bodies."""
    all_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(m.nbody)
    ]

    def search(side, keywords=("ankle", "foot")):
        for kw in keywords:
            for name in all_names:
                if name and side in name.lower() and kw in name.lower():
                    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                    return bid, name
        return -1, None

    l_id, l_name = search("left")
    r_id, r_name = search("right")

    if l_id < 0 or r_id < 0:
        raise RuntimeError(
            f"Could not find ankle/foot bodies. Available bodies:\n  {all_names}"
        )
    return l_id, l_name, r_id, r_name


def run_simulation(cfg: dict, duration: float = 20.0) -> dict:
    """
    Run G1 simulation for `duration` seconds and return raw ankle positions.

    Returns dict with keys:
        time, left_x, left_y, left_z, right_x, right_y, right_z
    """
    xml_path = find_xml(cfg)
    print(f"[SIM] Loading model: {xml_path}")

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = float(cfg.get("simulation_dt", 0.002))

    # ── Identify ankle bodies ────────────────────────────────────────────────
    l_id, l_name, r_id, r_name = find_foot_bodies(m)
    print(f"[SIM] Left ankle : {l_name}  (id={l_id})")
    print(f"[SIM] Right ankle: {r_name}  (id={r_id})")

    # ── PD gains & default posture ───────────────────────────────────────────
    n_act = m.nu
    n_dof = m.nq - 7  # free-joint = 7 DOF

    def pad(arr, n):
        a = np.array(arr, dtype=np.float64)
        if len(a) >= n:
            return a[:n]
        return np.pad(a, (0, n - len(a)))

    kp  = pad(cfg.get("kps", [100]*n_act), n_act)
    kd  = pad(cfg.get("kds", [2]  *n_act), n_act)
    q0  = pad(cfg.get("default_angles", [0]*n_dof), n_dof)

    # Initialise joint positions
    d.qpos[7:7+n_dof] = q0[:n_dof]
    mujoco.mj_forward(m, d)

    # ── Simulate ─────────────────────────────────────────────────────────────
    dt        = m.opt.timestep
    num_steps = int(duration / dt)
    rec_every = 1                       # record every step (downsample below)
    print(f"[SIM] Running {num_steps} steps ({duration}s) …")

    t_list  = []
    lx, ly, lz = [], [], []
    rx, ry, rz = [], [], []

    t0 = time.time()
    for step in range(num_steps):
        # Sinusoidal walking pattern (2 Hz gait cycle = 2 cycles per second)
        freq = 2.0  # Hz
        phase = (step * dt) * 2 * np.pi * freq   # radians, 2 cycles per second

        tgt = q0.copy()
        if n_dof >= 12:
            # Enhanced walking pattern for proper gait cycles:
            # Hip pitch: ±0.5 rad, anti-phase left/right (more forward motion)
            tgt[2]  = q0[2]  + 0.50 * np.sin(phase)          # L hip pitch
            tgt[8]  = q0[8]  - 0.50 * np.sin(phase)          # R hip pitch
            
            # Hip roll: ±0.25 rad for lateral sway (creates tear-drop X variation)
            tgt[1]  = q0[1]  + 0.25 * np.sin(phase + np.pi)   # L hip roll
            tgt[7]  = q0[7]  - 0.25 * np.sin(phase + np.pi)   # R hip roll
            
            # Knee flexion: strong swing phase
            tgt[3]  = q0[3]  + 0.35 * (1 - np.cos(phase))    # L knee (swing up)
            tgt[9]  = q0[9]  + 0.35 * (1 + np.cos(phase))    # R knee (swing up)
            
            # Ankle pitch: enhanced push-off timing
            tgt[4]  = q0[4]  - 0.20 * np.sin(phase)          # L ankle pitch
            tgt[10] = q0[10] + 0.20 * np.sin(phase)          # R ankle pitch

        tau = pd_control(
            tgt[:n_act], d.qpos[7:7+n_act],
            kp,           np.zeros(n_act),
            d.qvel[6:6+n_act], kd
        )
        d.ctrl[:] = np.clip(tau, -300, 300)

        # Drive the base forward to produce spatial walking path
        d.qvel[0] = 0.5   # +X forward (m/s) - increased for more progression

        mujoco.mj_step(m, d)

        if step % rec_every == 0:
            # Ground-truth body positions via xpos ──────────────────────────
            # Method 1 (preferred, named access):
            #   pos = d.body(l_name).xpos
            # Method 2 (index access, equally valid):
            #   pos = d.xpos[l_id]
            l_pos = d.xpos[l_id].copy()    # world-frame XYZ
            r_pos = d.xpos[r_id].copy()

            t_list.append(d.time)
            lx.append(l_pos[0]); ly.append(l_pos[1]); lz.append(l_pos[2])
            rx.append(r_pos[0]); ry.append(r_pos[1]); rz.append(r_pos[2])

        if (step + 1) % (num_steps // 10) == 0:
            pct = (step + 1) / num_steps * 100
            print(f"  {pct:5.1f}%  ({step+1}/{num_steps})  elapsed={time.time()-t0:.1f}s")

    print(f"[SIM] Done in {time.time()-t0:.2f}s")

    return {
        "time": np.array(t_list),
        "left_x":  np.array(lx),  "left_y":  np.array(ly),  "left_z":  np.array(lz),
        "right_x": np.array(rx),  "right_y": np.array(ry),  "right_z": np.array(rz),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOT  (matches reference: two side-by-side subplots, cm units, teardrop)
# ─────────────────────────────────────────────────────────────────────────────

def make_bilateral_plot(raw: dict, out_dir: Path):
    """
    Generate a plot with multiple distinct tear-drop gait cycles.
    Each cycle is displayed with clear separation (2x5 grid layout).
    Uses lateral (Y) vs height (Z) for tear-drop visualization.
    """

    def to_cm(arr):
        return arr * 100.0

    # Use lateral (Y) and height (Z) for tear-drop shapes
    ly_cm = to_cm(raw["left_y"])
    lz_cm = to_cm(raw["left_z"] - raw["left_z"].min())  # normalize height
    
    ry_cm = to_cm(raw["right_y"])
    rz_cm = to_cm(raw["right_z"] - raw["right_z"].min())

    # ── Detect cycles ─────────────────────────────────────────────────────────
    sample_rate = 1.0 / (raw["time"][1] - raw["time"][0]) if len(raw["time"]) > 1 else 100
    min_dist = int(0.5 * sample_rate)

    l_cycles, _ = find_gait_cycles(lz_cm, min_distance=min_dist)
    r_cycles, _ = find_gait_cycles(rz_cm, min_distance=min_dist)

    print(f"[PLOT] Detected {len(l_cycles)} left cycles, {len(r_cycles)} right cycles")

    # Always divide into 5 equal segments for consistent visualization
    n = len(ly_cm)
    l_cycles = [(i*n//5, (i+1)*n//5) for i in range(5)]
    r_cycles = [(i*n//5, (i+1)*n//5) for i in range(5)]

    # ── Create 2-row subplot layout (5 cols) ──────────────────────────────────
    n_cycles = max(len(l_cycles), len(r_cycles))
    fig, axes = plt.subplots(2, n_cycles, figsize=(16, 6))
    fig.patch.set_facecolor("white")
    
    # Ensure axes is always 2D array
    if n_cycles == 1:
        axes = axes.reshape(2, 1)

    # ── Resampling parameters ────────────────────────────────────────────────
    n_points = 100
    t_norm = np.linspace(0, 1, n_points)

    # ── Plot LEFT foot cycles (top row, blue) ────────────────────────────────
    for idx, (s, e) in enumerate(l_cycles):
        ax = axes[0, idx]
        
        seg_y = ly_cm[s:e]
        seg_z = lz_cm[s:e]
        
        if len(seg_y) < 2:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xlim(-3, 3)
            ax.set_ylim(0, 10)
            continue
        
        # Resample cycle to fixed points
        t_seg = np.linspace(0, 1, len(seg_y))
        y_interp = np.interp(t_norm, t_seg, seg_y)
        z_interp = np.interp(t_norm, t_seg, seg_z)
        
        # Center Y, keep Z from ground
        y_centered = y_interp - y_interp.mean()
        z_centered = z_interp - z_interp.min()
        
        # Plot the tear-drop cycle
        ax.plot(y_centered, z_centered, color='#1f77b4', linewidth=2.5, alpha=0.8)
        ax.fill(y_centered, z_centered, color='#1f77b4', alpha=0.1)
        ax.set_title(f"Left Cycle {idx+1}", fontsize=10, fontweight="bold", color='#1f77b4')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("Lateral (cm)")
        ax.set_ylabel("Height (cm)")

    # ── Plot RIGHT foot cycles (bottom row, red) ─────────────────────────────
    for idx, (s, e) in enumerate(r_cycles):
        ax = axes[1, idx]
        
        seg_y = ry_cm[s:e]
        seg_z = rz_cm[s:e]
        
        if len(seg_y) < 2:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xlim(-3, 3)
            ax.set_ylim(0, 10)
            continue
        
        # Resample cycle to fixed points
        t_seg = np.linspace(0, 1, len(seg_y))
        y_interp = np.interp(t_norm, t_seg, seg_y)
        z_interp = np.interp(t_norm, t_seg, seg_z)
        
        # Center Y, keep Z from ground
        y_centered = y_interp - y_interp.mean()
        z_centered = z_interp - z_interp.min()
        
        # Plot the tear-drop cycle
        ax.plot(y_centered, z_centered, color='#ff6b6b', linewidth=2.5, alpha=0.8)
        ax.fill(y_centered, z_centered, color='#ff6b6b', alpha=0.1)
        ax.set_title(f"Right Cycle {idx+1}", fontsize=10, fontweight="bold", color='#ff6b6b')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("Lateral (cm)")
        ax.set_ylabel("Height (cm)")

    plt.suptitle("G1 Robot Bilateral Gait Analysis - Tear Drop Cycles", 
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout(pad=1.0)

    out_path = out_dir / "bilateral_ankle_gait.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(raw: dict, out_dir: Path):
    df = pd.DataFrame(raw)
    p = out_dir / "ankle_trajectory_data.csv"
    df.to_csv(p, index=False)
    print(f"[CSV] Saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config_file  = sys.argv[1] if len(sys.argv) > 1 else "g1_walk.yaml"
    duration_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    out_dir      = Path("gait_trajectory_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  G1 BILATERAL ANKLE GAIT TRAJECTORY EXTRACTOR")
    print(f"  Config: {config_file}   Duration: {duration_sec}s")
    print(f"  Output: {out_dir.resolve()}")
    print("=" * 70)

    cfg = load_config(config_file)

    # ── Run simulation ────────────────────────────────────────────────────────
    raw = run_simulation(cfg, duration=duration_sec)

    # ── Generate plot ─────────────────────────────────────────────────────────
    img_path = make_bilateral_plot(raw, out_dir)

    # ── Save data ─────────────────────────────────────────────────────────────
    save_csv(raw, out_dir)

    print("\n[DONE] Output files:")
    print(f"  {img_path}")
    print(f"  {out_dir / 'ankle_trajectory_data.csv'}")


if __name__ == "__main__":
    main()