"""
IMU Data Collection at 3 Step Lengths for Unitree G1 Robot
===========================================================
Runs three sequential 1-minute trials at different step lengths (small,
medium, large) by varying the action_scale parameter. action_scale directly
controls how far each joint moves per step:
  - Larger action_scale  => longer strides
  - Smaller action_scale => shorter shuffling steps

Forward velocity command is kept constant (from g1.yaml cmd_init) so that
the cadence and speed remain comparable between trials.

Step length configurations:
  Small  : action_scale = 0.15  (~40% smaller than default)
  Medium : action_scale = 0.25  (default from g1.yaml)
  Large  : action_scale = 0.38  (~52% larger than default)

Output CSVs saved to `imu_data_collect/` folder:
  - small_step_imu_data.csv
  - medium_step_imu_data.csv
  - large_step_imu_data.csv

Usage:
    cd c:\\unitree_rl_gym-main\\unitree_rl_gym-main
    python deploy/deploy_mujoco/deploy_imu_step_lengths.py g1.yaml
"""

import time
import sys
import os
from pathlib import Path

# Add parent directories to path so we can import legged_gym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import csv


# ─── Step length configurations (action_scale values) ────────────────────────
# Kept within ±20% of the trained default (0.25) to avoid destabilising the
# policy. Larger deviations cause the robot to fall within ~10 seconds.
STEP_CONFIGS = {
    "small":  {"action_scale": 0.20, "label": "Small Steps  (action_scale=0.20, -20% from default)"},
    "medium": {"action_scale": 0.25, "label": "Medium Steps (action_scale=0.25, default)"},
    "large":  {"action_scale": 0.30, "label": "Large Steps  (action_scale=0.30, +20% from default)"},
}

TRIAL_DURATION = 60.0  # seconds per trial

CSV_HEADERS = [
    'Time(s)',
    'Gyro_X(rad/s)', 'Gyro_Y(rad/s)', 'Gyro_Z(rad/s)',
    'Accel_X(m/s2)', 'Accel_Y(m/s2)', 'Accel_Z(m/s2)',
    'Gravity_X', 'Gravity_Y', 'Gravity_Z',
    'LinVel_X(m/s)', 'LinVel_Y(m/s)', 'LinVel_Z(m/s)',
    'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z',
    'Pos_X(m)', 'Pos_Y(m)', 'Pos_Z(m)',
    'Left_Knee(rad)', 'Right_Knee(rad)',
    'Action_Scale',
]

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",   "left_elbow_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",  "right_elbow_joint",
]


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_step_trial(step_name, step_cfg, config, output_dir):
    """
    Run a single 1-minute simulation at a given step length (action_scale)
    and save IMU data to CSV.
    """
    action_scale = step_cfg["action_scale"]
    label        = step_cfg["label"]

    print(f"\n{'='*60}")
    print(f"  Trial : {label}")
    print(f"  Duration : {TRIAL_DURATION}s")
    print(f"{'='*60}\n")

    # ── Parse config ──────────────────────────────────────────────────────────
    policy_path    = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path       = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    simulation_dt      = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps            = np.array(config["kps"],            dtype=np.float32)
    kds            = np.array(config["kds"],            dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale  = config["ang_vel_scale"]
    dof_pos_scale  = config["dof_pos_scale"]
    dof_vel_scale  = config["dof_vel_scale"]
    cmd_scale      = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions    = config["num_actions"]
    num_obs        = config["num_obs"]
    cmd            = np.array(config["cmd_init"], dtype=np.float32)
    arm_target_angles = np.array(config.get("arm_target_angles", [0.0]*8), dtype=np.float32)
    arm_kp = float(config.get("arm_kp", 80.0))
    arm_kd = float(config.get("arm_kd", 2.0))

    # ── Context ───────────────────────────────────────────────────────────────
    action         = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs            = np.zeros(num_obs,     dtype=np.float32)
    counter        = 0
    prev_lin_vel   = np.zeros(3)
    control_dt     = simulation_dt * control_decimation
    data_rows      = 0

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = output_dir / f"{step_name}_step_imu_data.csv"
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(CSV_HEADERS)
    csv_file.flush()

    # ── Load model + policy ───────────────────────────────────────────────────
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    arm_ctrl_map = []
    for idx, jname in enumerate(ARM_JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    jname)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if jid >= 0 and aid >= 0:
            arm_ctrl_map.append((idx, aid,
                                 int(m.jnt_qposadr[jid]),
                                 int(m.jnt_dofadr[jid])))

    policy = torch.jit.load(policy_path)

    # ── Simulation loop ───────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < TRIAL_DURATION:
            step_start = time.time()

            # Leg torques
            d.ctrl[:] = 0.0
            tau_legs = pd_control(
                target_dof_pos,
                d.qpos[7: 7 + num_actions], kps,
                np.zeros_like(kds),
                d.qvel[6: 6 + num_actions], kds,
            )
            d.ctrl[:num_actions] = tau_legs

            # Arm torques
            for tidx, aid, qpa, qva in arm_ctrl_map:
                tau = (arm_target_angles[tidx] - d.qpos[qpa]) * arm_kp \
                    + (0.0 - d.qvel[qva]) * arm_kd
                d.ctrl[aid] = tau

            mujoco.mj_step(m, d)
            counter += 1

            if counter % control_decimation == 0:
                qj    = d.qpos[7: 7 + num_actions]
                dqj   = d.qvel[6: 6 + num_actions]
                quat  = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj_sc  = (qj - default_angles) * dof_pos_scale
                dqj_sc = dqj * dof_vel_scale
                grav   = get_gravity_orientation(quat)
                om_sc  = omega * ang_vel_scale

                period    = 0.8
                count     = counter * simulation_dt
                phase     = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3]                                           = om_sc
                obs[3:6]                                          = grav
                obs[6:9]                                          = cmd * cmd_scale
                obs[9: 9 + num_actions]                           = qj_sc
                obs[9 + num_actions: 9 + 2 * num_actions]         = dqj_sc
                obs[9 + 2 * num_actions: 9 + 3 * num_actions]     = action
                obs[9 + 3 * num_actions: 9 + 3 * num_actions + 2] = [sin_phase, cos_phase]

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action     = policy(obs_tensor).detach().numpy().squeeze()

                # Apply the trial-specific action_scale (controls step length)
                target_dof_pos = action * action_scale + default_angles

                # ── IMU data ──────────────────────────────────────────────────
                current_time = counter * simulation_dt
                base_lin_vel = d.qvel[0:3]
                base_pos     = d.qpos[0:3]
                accel        = (base_lin_vel - prev_lin_vel) / control_dt
                prev_lin_vel = base_lin_vel.copy()
                left_knee    = float(d.qpos[7 + 3])
                right_knee   = float(d.qpos[7 + 9])

                writer.writerow([
                    f"{current_time:.6f}",
                    f"{omega[0]:.6f}", f"{omega[1]:.6f}", f"{omega[2]:.6f}",
                    f"{accel[0]:.6f}", f"{accel[1]:.6f}", f"{accel[2]:.6f}",
                    f"{grav[0]:.6f}",  f"{grav[1]:.6f}",  f"{grav[2]:.6f}",
                    f"{base_lin_vel[0]:.6f}", f"{base_lin_vel[1]:.6f}", f"{base_lin_vel[2]:.6f}",
                    f"{quat[0]:.6f}", f"{quat[1]:.6f}", f"{quat[2]:.6f}", f"{quat[3]:.6f}",
                    f"{base_pos[0]:.6f}", f"{base_pos[1]:.6f}", f"{base_pos[2]:.6f}",
                    f"{left_knee:.6f}", f"{right_knee:.6f}",
                    f"{action_scale:.4f}",
                ])
                data_rows += 1

                if data_rows % 100 == 0:
                    csv_file.flush()

                # Progress every ~5s sim time
                if abs(current_time % 5.0) < control_dt:
                    print(
                        f"  [{label}] t={current_time:.1f}s | "
                        f"LinVel=({base_lin_vel[0]:.3f},{base_lin_vel[1]:.3f},{base_lin_vel[2]:.3f}) | "
                        f"Knee L={left_knee:.3f} R={right_knee:.3f} | "
                        f"Rows={data_rows}",
                        flush=True
                    )

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    csv_file.close()
    print(f"\n  [OK] Trial complete: {label}")
    print(f"       Rows collected : {data_rows}")
    print(f"       Saved to       : {csv_path}")
    return data_rows


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect IMU data at 3 step lengths for G1 robot"
    )
    parser.add_argument(
        "config_file", type=str,
        help="config file name in the config folder (e.g., g1.yaml)"
    )
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Output directory
    output_dir = Path(LEGGED_GYM_ROOT_DIR) / "imu_data_collect"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  IMU DATA COLLECTION -- 3 STEP LENGTHS")
    print("=" * 60)
    print(f"  Config     : {args.config_file}")
    print(f"  Output dir : {output_dir}")
    print(f"  Trials     : Small -> Medium -> Large steps")
    print(f"  Duration   : {TRIAL_DURATION}s each")
    print("=" * 60)

    results = {}
    for step_name, step_cfg in STEP_CONFIGS.items():
        rows = run_step_trial(step_name, step_cfg, config, output_dir)
        results[step_name] = rows

    # Final summary
    print("\n" + "=" * 60)
    print("  ALL TRIALS COMPLETE -- SUMMARY")
    print("=" * 60)
    for step_name, step_cfg in STEP_CONFIGS.items():
        csv_path = output_dir / f"{step_name}_step_imu_data.csv"
        print(f"  {step_cfg['label']}")
        print(f"    -> {csv_path}  (Rows: {results[step_name]})")
    print("=" * 60)
    print(f"\n  All CSV files saved to: {output_dir}\n")
