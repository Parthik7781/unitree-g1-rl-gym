"""
IMU Data Collection at 3 Walking Speeds for Unitree G1 Robot
=============================================================
Runs the G1 robot simulation at three walking speeds (slow, normal, fast),
each for 1 minute, and collects comprehensive IMU data into CSV files
stored in the `imu_data_collect/` folder.

Usage:
    cd c:\\unitree_rl_gym-main\\unitree_rl_gym-main
    python deploy/deploy_mujoco/deploy_imu_3speeds.py g1.yaml
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


# ─── Walking speed configurations ────────────────────────────────────────────
SPEED_CONFIGS = {
    "slow":   {"forward_vel": 0.25, "label": "Slow Walking (0.25 m/s cmd)"},
    "normal": {"forward_vel": 0.50, "label": "Normal Walking (0.50 m/s cmd)"},
    "fast":   {"forward_vel": 1.00, "label": "Fast Walking (1.00 m/s cmd)"},
}

# Duration for each speed trial (seconds)
TRIAL_DURATION = 60.0

# CSV column headers for IMU data
CSV_HEADERS = [
    'Time(s)',
    'Gyro_X(rad/s)', 'Gyro_Y(rad/s)', 'Gyro_Z(rad/s)',
    'Accel_X(m/s2)', 'Accel_Y(m/s2)', 'Accel_Z(m/s2)',
    'Gravity_X', 'Gravity_Y', 'Gravity_Z',
    'LinVel_X(m/s)', 'LinVel_Y(m/s)', 'LinVel_Z(m/s)',
    'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z',
    'Pos_X(m)', 'Pos_Y(m)', 'Pos_Z(m)',
]


def get_gravity_orientation(quaternion):
    """Compute gravity vector in body frame from quaternion."""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


# Arm joint names for the G1 model
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


def run_speed_trial(speed_name, speed_config, config, output_dir):
    """
    Run a single 1-minute simulation trial at the specified speed and
    collect IMU data into a CSV file.

    Parameters
    ----------
    speed_name : str
        Name key (e.g., "slow", "normal", "fast")
    speed_config : dict
        Contains 'forward_vel' and 'label'
    config : dict
        Parsed YAML configuration
    output_dir : Path
        Directory to save CSV output
    """
    forward_vel = speed_config["forward_vel"]
    label = speed_config["label"]

    print(f"\n{'='*60}")
    print(f"  Starting trial: {label}")
    print(f"  Duration: {TRIAL_DURATION}s | Forward velocity cmd: {forward_vel}")
    print(f"{'='*60}\n")

    # ── Parse config ──────────────────────────────────────────────────────────
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    # Set the walking command with the trial's forward velocity
    cmd = np.array([forward_vel, 0.0, 0.0], dtype=np.float32)

    arm_target_angles = np.array(
        config.get("arm_target_angles", [0.0] * 8), dtype=np.float32
    )
    arm_kp = float(config.get("arm_kp", 80.0))
    arm_kd = float(config.get("arm_kd", 2.0))

    # ── Context variables ─────────────────────────────────────────────────────
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Previous linear velocity for acceleration estimation
    prev_lin_vel = np.zeros(3)

    # ── CSV file setup ────────────────────────────────────────────────────────
    csv_path = output_dir / f"{speed_name}_imu_data.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADERS)
    csv_file.flush()

    # ── Load MuJoCo model ─────────────────────────────────────────────────────
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Build arm control mapping
    arm_ctrl_map = []
    for idx, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if joint_id >= 0 and actuator_id >= 0:
            arm_ctrl_map.append((
                idx, actuator_id,
                int(m.jnt_qposadr[joint_id]),
                int(m.jnt_dofadr[joint_id])
            ))

    if len(arm_ctrl_map) == 0:
        print("  Arm joints not found — running leg-only control.")
    else:
        print(f"  Arm control active on {len(arm_ctrl_map)} joints.")

    # ── Load policy ───────────────────────────────────────────────────────────
    policy = torch.jit.load(policy_path)

    # ── Data counters ─────────────────────────────────────────────────────────
    data_rows = 0
    control_dt = simulation_dt * control_decimation

    # ── Simulation loop ───────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < TRIAL_DURATION:
            step_start = time.time()

            # ── Leg torques ───────────────────────────────────────────────────
            d.ctrl[:] = 0.0
            tau_legs = pd_control(
                target_dof_pos,
                d.qpos[7: 7 + num_actions],
                kps,
                np.zeros_like(kds),
                d.qvel[6: 6 + num_actions],
                kds,
            )
            d.ctrl[:num_actions] = tau_legs

            # ── Arm torques ───────────────────────────────────────────────────
            for target_idx, actuator_id, qpos_adr, qvel_adr in arm_ctrl_map:
                q = d.qpos[qpos_adr]
                dq = d.qvel[qvel_adr]
                tq = arm_target_angles[target_idx]
                tau = (tq - q) * arm_kp + (0.0 - dq) * arm_kd
                d.ctrl[actuator_id] = tau

            # ── Step physics ──────────────────────────────────────────────────
            mujoco.mj_step(m, d)
            counter += 1

            # ── Policy + IMU logging (every control_decimation steps) ─────────
            if counter % control_decimation == 0:
                # Build observation for the policy
                qj = d.qpos[7: 7 + num_actions]
                dqj = d.qvel[6: 6 + num_actions]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]   # angular velocity (raw)

                qj_scaled = (qj - default_angles) * dof_pos_scale
                dqj_scaled = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega_scaled = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega_scaled
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9: 9 + num_actions] = qj_scaled
                obs[9 + num_actions: 9 + 2 * num_actions] = dqj_scaled
                obs[9 + 2 * num_actions: 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions: 9 + 3 * num_actions + 2] = np.array(
                    [sin_phase, cos_phase]
                )

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

                # ── Collect IMU data ──────────────────────────────────────────
                current_time = counter * simulation_dt
                base_lin_vel = d.qvel[0:3]          # linear velocity
                base_pos = d.qpos[0:3]              # position

                # Estimate linear acceleration from velocity change
                accel = (base_lin_vel - prev_lin_vel) / control_dt
                prev_lin_vel = base_lin_vel.copy()

                # Write IMU row
                csv_writer.writerow([
                    f"{current_time:.6f}",
                    # Gyroscope (angular velocity, raw)
                    f"{omega[0]:.6f}", f"{omega[1]:.6f}", f"{omega[2]:.6f}",
                    # Accelerometer (estimated)
                    f"{accel[0]:.6f}", f"{accel[1]:.6f}", f"{accel[2]:.6f}",
                    # Gravity orientation
                    f"{gravity_orientation[0]:.6f}",
                    f"{gravity_orientation[1]:.6f}",
                    f"{gravity_orientation[2]:.6f}",
                    # Linear velocity
                    f"{base_lin_vel[0]:.6f}",
                    f"{base_lin_vel[1]:.6f}",
                    f"{base_lin_vel[2]:.6f}",
                    # Quaternion orientation
                    f"{quat[0]:.6f}", f"{quat[1]:.6f}",
                    f"{quat[2]:.6f}", f"{quat[3]:.6f}",
                    # Position
                    f"{base_pos[0]:.6f}", f"{base_pos[1]:.6f}",
                    f"{base_pos[2]:.6f}",
                ])
                data_rows += 1

                # Flush every 100 rows for safety
                if data_rows % 100 == 0:
                    csv_file.flush()

                # Console progress every 5 seconds of sim time
                if abs(current_time % 5.0) < control_dt:
                    print(
                        f"  [{label}] t={current_time:.1f}s | "
                        f"LinVel=({base_lin_vel[0]:.3f}, {base_lin_vel[1]:.3f}, {base_lin_vel[2]:.3f}) | "
                        f"Gyro=({omega[0]:.3f}, {omega[1]:.3f}, {omega[2]:.3f}) | "
                        f"Rows={data_rows}",
                        flush=True
                    )

            # ── Viewer sync + time keeping ────────────────────────────────────
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # ── Close CSV ─────────────────────────────────────────────────────────────
    csv_file.close()
    print(f"\n  [OK] Trial complete: {label}")
    print(f"    Rows collected: {data_rows}")
    print(f"    Saved to: {csv_path}")

    return data_rows


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect IMU data at 3 walking speeds for G1 robot"
    )
    parser.add_argument(
        "config_file", type=str,
        help="config file name in the config folder (e.g., g1.yaml)"
    )
    args = parser.parse_args()

    # Load YAML config once
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create output directory
    output_dir = Path(LEGGED_GYM_ROOT_DIR) / "imu_data_collect"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  IMU DATA COLLECTION -- UNITREE G1 ROBOT")
    print("=" * 60)
    print(f"  Config: {args.config_file}")
    print(f"  Output: {output_dir}")
    print(f"  Speeds: {', '.join(sc['label'] for sc in SPEED_CONFIGS.values())}")
    print(f"  Trial duration: {TRIAL_DURATION}s each")
    print("=" * 60)

    # Run all three speed trials sequentially
    results = {}
    for speed_name, speed_config in SPEED_CONFIGS.items():
        rows = run_speed_trial(speed_name, speed_config, config, output_dir)
        results[speed_name] = rows

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL TRIALS COMPLETE -- SUMMARY")
    print("=" * 60)
    for speed_name, speed_config in SPEED_CONFIGS.items():
        csv_path = output_dir / f"{speed_name}_imu_data.csv"
        print(f"  {speed_config['label']:>35s}  ->  {csv_path}")
        print(f"  {'':>35s}     Rows: {results[speed_name]}")
    print("=" * 60)
    print(f"\n  All CSV files saved to: {output_dir}\n")
