"""
IMU Data Collection with BOTH Legs Restricted for Unitree G1 Robot
===================================================================
Runs a single 1-minute trial with both legs restricted (reduced ROM).
Restriction is applied by scaling all leg joint excursions around the
default pose using rom_scale (default = 0.95, ~5% reduction).
A very mild restriction is used so the robot stays stable for the full
60 seconds (the policy was not trained with restrictions).

Output CSV saved to `imu_data_collect/both_restricted_imu_data.csv`

Usage:
    cd c:\\unitree_rl_gym-main\\unitree_rl_gym-main
    python deploy/deploy_mujoco/deploy_imu_both_restricted.py g1.yaml

Optional args:
    --rom_scale 0.95       ROM scale factor (0.0=fully locked, 1.0=no restriction)
    --restrict_percent 5   Alternative: restriction as percentage (overrides rom_scale)
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


# ─── All 12 leg DOF (both legs) ──────────────────────────────────────────────
# 0=L_hip_pitch, 1=L_hip_roll, 2=L_hip_yaw, 3=L_knee, 4=L_ankle_pitch, 5=L_ankle_roll
# 6=R_hip_pitch, 7=R_hip_roll, 8=R_hip_yaw, 9=R_knee,10=R_ankle_pitch,11=R_ankle_roll
BOTH_LEG_INDICES = list(range(12))

TRIAL_DURATION = 60.0  # seconds

CSV_HEADERS = [
    'Time(s)',
    'Gyro_X(rad/s)', 'Gyro_Y(rad/s)', 'Gyro_Z(rad/s)',
    'Accel_X(m/s2)', 'Accel_Y(m/s2)', 'Accel_Z(m/s2)',
    'Gravity_X', 'Gravity_Y', 'Gravity_Z',
    'LinVel_X(m/s)', 'LinVel_Y(m/s)', 'LinVel_Z(m/s)',
    'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z',
    'Pos_X(m)', 'Pos_Y(m)', 'Pos_Z(m)',
    'Left_Knee(rad)', 'Right_Knee(rad)',
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


def apply_restriction(target_dof_pos, default_angles, rom_scale, restricted_indices):
    """Scale excursion of specified joints around their default pose."""
    restricted = target_dof_pos.copy()
    for idx in restricted_indices:
        delta = restricted[idx] - default_angles[idx]
        restricted[idx] = default_angles[idx] + rom_scale * delta
    return restricted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect IMU data with BOTH legs restricted for G1 robot"
    )
    parser.add_argument(
        "config_file", type=str,
        help="config file name in the config folder (e.g., g1.yaml)"
    )
    parser.add_argument(
        "--rom_scale", type=float, default=0.95,
        help="ROM scale factor for both legs (default=0.95 => ~5%% restriction; raise if robot falls)"
    )
    parser.add_argument(
        "--restrict_percent", type=float, default=None,
        help="Restriction as a percentage e.g. 5 means 5%% ROM reduction (overrides --rom_scale)"
    )
    args = parser.parse_args()

    if args.restrict_percent is not None:
        args.rom_scale = 1.0 - (args.restrict_percent / 100.0)

    restriction_pct = round((1 - args.rom_scale) * 100)

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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
    action_scale   = config["action_scale"]
    cmd_scale      = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions    = config["num_actions"]
    num_obs        = config["num_obs"]
    cmd            = np.array(config["cmd_init"], dtype=np.float32)
    arm_target_angles = np.array(config.get("arm_target_angles", [0.0] * 8), dtype=np.float32)
    arm_kp = float(config.get("arm_kp", 80.0))
    arm_kd = float(config.get("arm_kd", 2.0))

    # ── Output dir + CSV ─────────────────────────────────────────────────────
    output_dir = Path(LEGGED_GYM_ROOT_DIR) / "imu_data_collect"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "both_restricted_imu_data.csv"

    print("\n" + "=" * 60)
    print("  IMU DATA COLLECTION -- BOTH LEGS RESTRICTED")
    print("=" * 60)
    print(f"  Config      : {args.config_file}")
    print(f"  Output      : {csv_path}")
    print(f"  ROM scale   : {args.rom_scale}  (~{restriction_pct}% restriction on both legs)")
    print(f"  Duration    : {TRIAL_DURATION}s")
    print("=" * 60 + "\n")

    # ── Context variables ─────────────────────────────────────────────────────
    action         = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs            = np.zeros(num_obs, dtype=np.float32)
    counter        = 0
    prev_lin_vel   = np.zeros(3)
    control_dt     = simulation_dt * control_decimation
    data_rows      = 0

    # ── Open CSV ──────────────────────────────────────────────────────────────
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(CSV_HEADERS)
    csv_file.flush()

    # ── Load model + policy ───────────────────────────────────────────────────
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    ARM_JOINT_NAMES = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",   "left_elbow_joint",
        "right_shoulder_pitch_joint","right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",  "right_elbow_joint",
    ]
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

                obs[:3]                                          = om_sc
                obs[3:6]                                         = grav
                obs[6:9]                                         = cmd * cmd_scale
                obs[9: 9 + num_actions]                          = qj_sc
                obs[9 + num_actions: 9 + 2 * num_actions]        = dqj_sc
                obs[9 + 2 * num_actions: 9 + 3 * num_actions]    = action
                obs[9 + 3 * num_actions: 9 + 3 * num_actions + 2]= [sin_phase, cos_phase]

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action     = policy(obs_tensor).detach().numpy().squeeze()

                # Apply restriction to BOTH legs
                unconstrained  = action * action_scale + default_angles
                target_dof_pos = apply_restriction(
                    unconstrained, default_angles, args.rom_scale, BOTH_LEG_INDICES
                )

                # IMU data
                current_time = counter * simulation_dt
                base_lin_vel = d.qvel[0:3]
                base_pos     = d.qpos[0:3]
                accel        = (base_lin_vel - prev_lin_vel) / control_dt
                prev_lin_vel = base_lin_vel.copy()

                left_knee  = float(d.qpos[7 + 3])
                right_knee = float(d.qpos[7 + 9])

                writer.writerow([
                    f"{current_time:.6f}",
                    f"{omega[0]:.6f}", f"{omega[1]:.6f}", f"{omega[2]:.6f}",
                    f"{accel[0]:.6f}", f"{accel[1]:.6f}", f"{accel[2]:.6f}",
                    f"{grav[0]:.6f}",  f"{grav[1]:.6f}",  f"{grav[2]:.6f}",
                    f"{base_lin_vel[0]:.6f}", f"{base_lin_vel[1]:.6f}", f"{base_lin_vel[2]:.6f}",
                    f"{quat[0]:.6f}", f"{quat[1]:.6f}", f"{quat[2]:.6f}", f"{quat[3]:.6f}",
                    f"{base_pos[0]:.6f}", f"{base_pos[1]:.6f}", f"{base_pos[2]:.6f}",
                    f"{left_knee:.6f}", f"{right_knee:.6f}",
                ])
                data_rows += 1

                if data_rows % 100 == 0:
                    csv_file.flush()

                # Progress every ~5 sim seconds
                if abs(current_time % 5.0) < control_dt:
                    print(
                        f"  [Both Restricted] t={current_time:.1f}s | "
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

    print("\n" + "=" * 60)
    print("  TRIAL COMPLETE")
    print("=" * 60)
    print(f"  Condition   : Both legs restricted (~{restriction_pct}% ROM reduction)")
    print(f"  Rows        : {data_rows}")
    print(f"  Saved to    : {csv_path}")
    print("=" * 60 + "\n")
