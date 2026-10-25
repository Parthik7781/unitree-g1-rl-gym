"""
IMU Data Collection with Restricted Leg Walking for Unitree G1 Robot
=====================================================================
Runs two sequential 1-minute trials:
  Trial 1: Left leg restricted (reduced ROM), right leg normal
  Trial 2: Right leg restricted (reduced ROM), left leg normal

Restriction is applied by scaling all left/right leg joint excursions
around the default pose using rom_scale (default = 0.85, ~15% reduction).
A mild restriction is used so the robot remains stable (the policy was not
trained with restrictions, so large values cause falls).

Output CSVs are saved to `imu_data_collect/` folder:
  - left_restricted_imu_data.csv
  - right_restricted_imu_data.csv

Usage:
    cd c:\\unitree_rl_gym-main\\unitree_rl_gym-main
    python deploy/deploy_mujoco/deploy_imu_restricted_legs.py g1.yaml

Optional args:
    --rom_scale 0.85       ROM scale factor (0.0=fully locked, 1.0=no restriction)
    --restrict_percent 15  Alternative: restriction as percentage (overrides rom_scale)
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


# ─── G1 12-DOF leg joint index layout ────────────────────────────────────────
# Index  Joint
# ─────────────────────────────────────────────────
#   0    left_hip_pitch_joint
#   1    left_hip_roll_joint
#   2    left_hip_yaw_joint
#   3    left_knee_joint
#   4    left_ankle_pitch_joint
#   5    left_ankle_roll_joint
#   6    right_hip_pitch_joint
#   7    right_hip_roll_joint
#   8    right_hip_yaw_joint
#   9    right_knee_joint
#  10    right_ankle_pitch_joint
#  11    right_ankle_roll_joint

LEFT_LEG_INDICES  = [0, 1, 2, 3, 4, 5]   # all 6 left-leg DOF
RIGHT_LEG_INDICES = [6, 7, 8, 9, 10, 11]  # all 6 right-leg DOF

# Duration for each trial (seconds)
TRIAL_DURATION = 60.0

# CSV column headers
CSV_HEADERS = [
    'Time(s)',
    'Gyro_X(rad/s)', 'Gyro_Y(rad/s)', 'Gyro_Z(rad/s)',
    'Accel_X(m/s2)', 'Accel_Y(m/s2)', 'Accel_Z(m/s2)',
    'Gravity_X', 'Gravity_Y', 'Gravity_Z',
    'LinVel_X(m/s)', 'LinVel_Y(m/s)', 'LinVel_Z(m/s)',
    'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z',
    'Pos_X(m)', 'Pos_Y(m)', 'Pos_Z(m)',
    # Knee angles for quick diagnosis
    'Left_Knee(rad)', 'Right_Knee(rad)',
]

# Trial definitions: (csv_name, label, restricted_indices)
TRIAL_CONFIGS = [
    {
        "csv_name":          "left_restricted_imu_data.csv",
        "label":             "Left Leg Restricted / Right Leg Normal",
        "restricted_indices": LEFT_LEG_INDICES,
    },
    {
        "csv_name":          "right_restricted_imu_data.csv",
        "label":             "Right Leg Restricted / Left Leg Normal",
        "restricted_indices": RIGHT_LEG_INDICES,
    },
]


def get_gravity_orientation(quaternion):
    """Compute gravity vector in body frame from quaternion."""
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def apply_restriction(target_dof_pos, default_angles, rom_scale, restricted_indices):
    """
    Scale excursion of specified joints around their default pose.
    A rom_scale of 0.55 means the joint can only move 55% as far from
    its default angle as the policy intended (45% ROM reduction).
    """
    restricted = target_dof_pos.copy()
    for idx in restricted_indices:
        delta = restricted[idx] - default_angles[idx]
        restricted[idx] = default_angles[idx] + rom_scale * delta
    return restricted


def run_trial(trial_cfg, config, output_dir, rom_scale):
    """
    Run a single 1-minute simulation trial with one leg restricted
    and collect IMU data into a CSV file.
    """
    label             = trial_cfg["label"]
    csv_name          = trial_cfg["csv_name"]
    restricted_indices = trial_cfg["restricted_indices"]

    print(f"\n{'='*60}")
    print(f"  Trial : {label}")
    print(f"  ROM scale: {rom_scale}  (restriction ~{round((1-rom_scale)*100)}%)")
    print(f"  Duration : {TRIAL_DURATION}s")
    print(f"{'='*60}\n")

    # ── Parse config ──────────────────────────────────────────────────────────
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path    = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

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

    arm_target_angles = np.array(
        config.get("arm_target_angles", [0.0] * 8), dtype=np.float32
    )
    arm_kp = float(config.get("arm_kp", 80.0))
    arm_kd = float(config.get("arm_kd", 2.0))

    # ── Context variables ─────────────────────────────────────────────────────
    action         = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs            = np.zeros(num_obs, dtype=np.float32)
    counter        = 0
    prev_lin_vel   = np.zeros(3)
    control_dt     = simulation_dt * control_decimation

    # ── CSV setup ─────────────────────────────────────────────────────────────
    csv_path = output_dir / csv_name
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(CSV_HEADERS)
    csv_file.flush()

    # ── Load model + policy ───────────────────────────────────────────────────
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Build arm control mapping (same as deploy_mujoco.py)
    ARM_JOINT_NAMES = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",   "left_elbow_joint",
        "right_shoulder_pitch_joint","right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",  "right_elbow_joint",
    ]
    arm_ctrl_map = []
    for idx, jname in enumerate(ARM_JOINT_NAMES):
        jid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    jname)
        aid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if jid >= 0 and aid >= 0:
            arm_ctrl_map.append((idx, aid,
                                 int(m.jnt_qposadr[jid]),
                                 int(m.jnt_dofadr[jid])))

    policy = torch.jit.load(policy_path)
    data_rows = 0

    # ── Simulation loop ───────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < TRIAL_DURATION:
            step_start = time.time()

            # Leg torques
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

            # Arm torques
            for tidx, aid, qpa, qva in arm_ctrl_map:
                tau = (arm_target_angles[tidx] - d.qpos[qpa]) * arm_kp \
                    + (0.0 - d.qvel[qva]) * arm_kd
                d.ctrl[aid] = tau

            mujoco.mj_step(m, d)
            counter += 1

            if counter % control_decimation == 0:
                # Build observation
                qj   = d.qpos[7: 7 + num_actions]
                dqj  = d.qvel[6: 6 + num_actions]
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

                obs[:3]                                      = om_sc
                obs[3:6]                                     = grav
                obs[6:9]                                     = cmd * cmd_scale
                obs[9: 9 + num_actions]                      = qj_sc
                obs[9 + num_actions: 9 + 2 * num_actions]    = dqj_sc
                obs[9 + 2 * num_actions: 9 + 3 * num_actions]= action
                obs[9 + 3 * num_actions: 9 + 3 * num_actions + 2] = [sin_phase, cos_phase]

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action     = policy(obs_tensor).detach().numpy().squeeze()

                # Apply restriction to the specified leg
                unconstrained   = action * action_scale + default_angles
                target_dof_pos  = apply_restriction(
                    unconstrained, default_angles, rom_scale, restricted_indices
                )

                # ── Collect IMU data ──────────────────────────────────────────
                current_time = counter * simulation_dt
                base_lin_vel = d.qvel[0:3]
                base_pos     = d.qpos[0:3]
                accel        = (base_lin_vel - prev_lin_vel) / control_dt
                prev_lin_vel = base_lin_vel.copy()

                left_knee  = float(d.qpos[7 + 3])   # index 3 in qpos leg block
                right_knee = float(d.qpos[7 + 9])   # index 9 in qpos leg block

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

                # Progress print every ~5s of sim time
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
        description="Collect IMU data with restricted left/right leg walking for G1 robot"
    )
    parser.add_argument(
        "config_file", type=str,
        help="config file name in the config folder (e.g., g1.yaml)"
    )
    parser.add_argument(
        "--rom_scale", type=float, default=0.95,
        help="ROM scale factor for restricted leg (default=0.95 => ~5%% restriction; raise if robot falls)"
    )
    parser.add_argument(
        "--restrict_percent", type=float, default=None,
        help="Restriction as a percentage e.g. 5 means 5%% ROM reduction (overrides --rom_scale)"
    )
    args = parser.parse_args()

    # Allow percent-based override
    if args.restrict_percent is not None:
        args.rom_scale = 1.0 - (args.restrict_percent / 100.0)

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create output directory
    output_dir = Path(LEGGED_GYM_ROOT_DIR) / "imu_data_collect"
    output_dir.mkdir(parents=True, exist_ok=True)

    restriction_pct = round((1 - args.rom_scale) * 100)

    print("\n" + "=" * 60)
    print("  IMU DATA COLLECTION -- RESTRICTED LEG WALKING")
    print("=" * 60)
    print(f"  Config      : {args.config_file}")
    print(f"  Output dir  : {output_dir}")
    print(f"  ROM scale   : {args.rom_scale}  (~{restriction_pct}% restriction)")
    print(f"  Trials      : Left restricted, then Right restricted")
    print(f"  Duration    : {TRIAL_DURATION}s each")
    print("=" * 60)

    results = {}
    for trial_cfg in TRIAL_CONFIGS:
        rows = run_trial(trial_cfg, config, output_dir, args.rom_scale)
        results[trial_cfg["csv_name"]] = rows

    # Final summary
    print("\n" + "=" * 60)
    print("  ALL TRIALS COMPLETE -- SUMMARY")
    print("=" * 60)
    for trial_cfg in TRIAL_CONFIGS:
        csv_path = output_dir / trial_cfg["csv_name"]
        print(f"  {trial_cfg['label']}")
        print(f"    -> {csv_path}  (Rows: {results[trial_cfg['csv_name']]})")
    print("=" * 60)
    print(f"\n  All CSV files saved to: {output_dir}\n")
