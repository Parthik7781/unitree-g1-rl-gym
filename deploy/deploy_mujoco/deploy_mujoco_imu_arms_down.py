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


def get_gravity_orientation(quaternion):
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


def get_imu_data(model, data, body_name):
    """Extract IMU data (acceleration and angular velocity) from a body"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    body_vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, True)

    lin_vel = body_vel[0:3]
    ang_vel = body_vel[3:6]

    return lin_vel, ang_vel


def estimate_acceleration(prev_lin_vel, curr_lin_vel, dt):
    """Estimate acceleration from velocity change"""
    return (curr_lin_vel - prev_lin_vel) / dt


# ─────────────────────────────────────────────────────────────────────────────
# G1 Robot Arm Joint Layout (indices relative to arm DOF block, starting at
# qpos[19] / qvel[18] when the model has 12 leg + 8 arm DOF):
#
#   Index  |  Joint
#   ──────────────────────────────────────
#     0    |  Left  Shoulder Pitch   (+ve = forward swing)
#     1    |  Left  Shoulder Roll    (+ve = away from body)
#     2    |  Left  Shoulder Yaw     (+ve = internal rotation)
#     3    |  Left  Elbow Pitch      (+ve = flex / bend)
#     4    |  Right Shoulder Pitch   (+ve = forward swing)
#     5    |  Right Shoulder Roll    (-ve = away from body, opposite sign)
#     6    |  Right Shoulder Yaw     (+ve = internal rotation)
#     7    |  Right Elbow Pitch      (+ve = flex / bend)
#
# "Arms Down – straight" target:
#   Both arms hang vertically along the sides of the torso.
#   Shoulder Pitch  = 0.0   (no forward/backward swing)
#   Shoulder Roll   = 0.0   (arms close to body)
#   Shoulder Yaw    = 0.0   (neutral rotation)
#   Elbow           = 0.0   (fully extended, straight arm)
# ─────────────────────────────────────────────────────────────────────────────

# Target arm joint angles for "Arms Down – straight" posture
# Order: [L_sh_pitch, L_sh_roll, L_sh_yaw, L_elbow,
#          R_sh_pitch, R_sh_roll, R_sh_yaw, R_elbow]
ARMS_DOWN_TARGET = np.array(
    [0.0,  0.0,  0.0,  0.0,   # Left arm  – straight down
     0.0,  0.0,  0.0,  0.0],  # Right arm – straight down
    dtype=np.float32
)

# PD gains for arm joints (tune if the arms oscillate or are too stiff)
ARM_KP = 80.0    # Position gain  [N·m / rad]
ARM_KD = 2.0     # Damping gain   [N·m·s / rad]
NUM_ARM_DOF = 8  # 4 per arm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="config file name in the config folder")
    parser.add_argument("--arm_posture", type=str, default="P2_Arms_Down",
                        help="Arm posture (fixed to P2_Arms_Down in this version)")
    args = parser.parse_args()

    config_file  = args.config_file
    arm_posture  = "P2_Arms_Down"   # Hard-coded for this experiment

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path        = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path           = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        simulation_duration = config["simulation_duration"]
        simulation_dt      = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps                = np.array(config["kps"],            dtype=np.float32)
        kds                = np.array(config["kds"],            dtype=np.float32)
        default_angles     = np.array(config["default_angles"], dtype=np.float32)
        ang_vel_scale      = config["ang_vel_scale"]
        dof_pos_scale      = config["dof_pos_scale"]
        dof_vel_scale      = config["dof_vel_scale"]
        action_scale       = config["action_scale"]
        cmd_scale          = np.array(config["cmd_scale"],      dtype=np.float32)
        num_actions        = config["num_actions"]
        num_obs            = config["num_obs"]
        cmd                = np.array(config["cmd_init"],       dtype=np.float32)

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = Path("Experimental_graphs") / arm_posture
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Gait Analysis: {arm_posture}")
    print(f"Posture      : Both arms straight down (P2)")
    print(f"Output dir   : {output_dir}")
    print(f"{'='*60}\n")

    # ── Context variables ─────────────────────────────────────────────────────
    action         = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs            = np.zeros(num_obs,     dtype=np.float32)
    counter        = 0

    # ── CSV files ─────────────────────────────────────────────────────────────
    trajectory_file   = open(output_dir / "trajectory.csv",    'w', newline='')
    trajectory_writer = csv.writer(trajectory_file)
    trajectory_writer.writerow(['Time(s)', 'X_Pos(m)', 'Y_Pos(m)'])
    trajectory_file.flush()

    left_imu_file   = open(output_dir / "left_hand_imu.csv",   'w', newline='')
    left_imu_writer = csv.writer(left_imu_file)
    left_imu_writer.writerow(['Time(s)', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    left_imu_file.flush()

    right_imu_file   = open(output_dir / "right_hand_imu.csv", 'w', newline='')
    right_imu_writer = csv.writer(right_imu_file)
    right_imu_writer.writerow(['Time(s)', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    right_imu_file.flush()

    # NEW: arm joint angle log so you can verify the arms are truly down
    arm_angle_file   = open(output_dir / "arm_joint_angles.csv", 'w', newline='')
    arm_angle_writer = csv.writer(arm_angle_file)
    arm_angle_writer.writerow([
        'Time(s)',
        'L_ShoulderPitch(rad)', 'L_ShoulderRoll(rad)',
        'L_ShoulderYaw(rad)',   'L_Elbow(rad)',
        'R_ShoulderPitch(rad)', 'R_ShoulderRoll(rad)',
        'R_ShoulderYaw(rad)',   'R_Elbow(rad)'
    ])
    arm_angle_file.flush()

    # ── Load MuJoCo model & policy ────────────────────────────────────────────
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)

    # Detect whether the loaded XML has arm DOF
    n_dof        = len(d.qpos) - 7          # total DOF (exclude floating base)
    has_arm_dof  = n_dof > 12               # True when model includes arm joints
    n_arm_dof    = n_dof - 12 if has_arm_dof else 0

    print(f"Model DOF      : {n_dof} total")
    if has_arm_dof:
        print(f"  Leg DOF      : 12")
        print(f"  Arm DOF      : {n_arm_dof}")
        if n_arm_dof >= NUM_ARM_DOF:
            print(f"  Arm control  : ENABLED – targeting 'Arms Down' posture")
        else:
            print(f"  Arm control  : PARTIAL – model has fewer than {NUM_ARM_DOF} arm DOF")
    else:
        print(f"  Leg DOF      : 12 (arms are passive / not in this XML)")
        print(f"  Arm control  : DISABLED – switch to a full-body XML to enable")
    print()

    # Clamp to however many arm DOF are actually available
    active_arm_dof    = min(n_arm_dof, NUM_ARM_DOF)
    arms_down_target  = ARMS_DOWN_TARGET[:active_arm_dof]
    arm_kps           = np.full(active_arm_dof, ARM_KP, dtype=np.float32)
    arm_kds           = np.full(active_arm_dof, ARM_KD, dtype=np.float32)

    # ── Previous velocity buffers for acceleration estimation ─────────────────
    prev_left_lin_vel  = np.zeros(3)
    prev_right_lin_vel = np.zeros(3)

    # ── Simulation loop ───────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # ── 1. Leg torques from policy ────────────────────────────────────
            tau_legs = pd_control(
                target_dof_pos,
                d.qpos[7:19],
                kps,
                np.zeros_like(kds),
                d.qvel[6:18],
                kds
            )
            d.ctrl[:12] = tau_legs

            # ── 2. Arm torques – hold "Arms Down" posture ─────────────────────
            # Both left and right arms are commanded to hang straight down.
            # qpos[19 : 19+active_arm_dof] holds the arm joint positions.
            # qvel[18 : 18+active_arm_dof] holds the arm joint velocities.
            if has_arm_dof and active_arm_dof > 0:
                tau_arms = pd_control(
                    arms_down_target,              # target: all zeros → straight down
                    d.qpos[19 : 19 + active_arm_dof],
                    arm_kps,
                    np.zeros(active_arm_dof, dtype=np.float32),
                    d.qvel[18 : 18 + active_arm_dof],
                    arm_kds
                )
                d.ctrl[12 : 12 + active_arm_dof] = tau_arms

            mujoco.mj_step(m, d)
            counter += 1

            # ── 3. Policy + data logging (every control_decimation steps) ─────
            if counter % control_decimation == 0:

                # Build observation (leg DOF only, as the policy was trained on)
                qj   = d.qpos[7:19]
                dqj  = d.qvel[6:18]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj   = (qj - default_angles) * dof_pos_scale
                dqj  = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega_scaled = omega * ang_vel_scale

                period    = 0.8
                count     = counter * simulation_dt
                phase     = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3]                                    = omega_scaled
                obs[3:6]                                   = gravity_orientation
                obs[6:9]                                   = cmd * cmd_scale
                obs[9 : 9 + num_actions]                   = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action     = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

                current_time = counter * simulation_dt

                # ── Trajectory log ────────────────────────────────────────────
                robot_x = d.xpos[0, 0]
                robot_y = d.xpos[0, 1]
                trajectory_writer.writerow([
                    f"{current_time:.6f}",
                    f"{robot_x:.6f}",
                    f"{robot_y:.6f}"
                ])
                trajectory_file.flush()

                # ── Arm joint angle log ───────────────────────────────────────
                if has_arm_dof and active_arm_dof > 0:
                    arm_q = d.qpos[19 : 19 + active_arm_dof]
                    # Pad to 8 values if fewer DOF exist
                    arm_q_padded = np.zeros(NUM_ARM_DOF)
                    arm_q_padded[:active_arm_dof] = arm_q
                    arm_angle_writer.writerow([
                        f"{current_time:.6f}",
                        f"{arm_q_padded[0]:.6f}",   # L shoulder pitch
                        f"{arm_q_padded[1]:.6f}",   # L shoulder roll
                        f"{arm_q_padded[2]:.6f}",   # L shoulder yaw
                        f"{arm_q_padded[3]:.6f}",   # L elbow
                        f"{arm_q_padded[4]:.6f}",   # R shoulder pitch
                        f"{arm_q_padded[5]:.6f}",   # R shoulder roll
                        f"{arm_q_padded[6]:.6f}",   # R shoulder yaw
                        f"{arm_q_padded[7]:.6f}",   # R elbow
                    ])
                    arm_angle_file.flush()

                # ── Left hand IMU ─────────────────────────────────────────────
                try:
                    left_lin_vel, left_ang_vel = get_imu_data(
                        m, d, "left_wrist_roll_rubber_hand"
                    )
                    left_acc = estimate_acceleration(
                        prev_left_lin_vel, left_lin_vel,
                        simulation_dt * control_decimation
                    )
                    prev_left_lin_vel = left_lin_vel.copy()

                    left_imu_writer.writerow([
                        f"{current_time:.6f}",
                        f"{left_acc[0]:.6f}",
                        f"{left_acc[1]:.6f}",
                        f"{left_acc[2]:.6f}",
                        f"{left_ang_vel[0]:.6f}",
                        f"{left_ang_vel[1]:.6f}",
                        f"{left_ang_vel[2]:.6f}"
                    ])
                    left_imu_file.flush()
                except Exception:
                    pass

                # ── Right hand IMU ────────────────────────────────────────────
                try:
                    right_lin_vel, right_ang_vel = get_imu_data(
                        m, d, "right_wrist_roll_rubber_hand"
                    )
                    right_acc = estimate_acceleration(
                        prev_right_lin_vel, right_lin_vel,
                        simulation_dt * control_decimation
                    )
                    prev_right_lin_vel = right_lin_vel.copy()

                    right_imu_writer.writerow([
                        f"{current_time:.6f}",
                        f"{right_acc[0]:.6f}",
                        f"{right_acc[1]:.6f}",
                        f"{right_acc[2]:.6f}",
                        f"{right_ang_vel[0]:.6f}",
                        f"{right_ang_vel[1]:.6f}",
                        f"{right_ang_vel[2]:.6f}"
                    ])
                    right_imu_file.flush()
                except Exception:
                    pass

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # ── Close all files ───────────────────────────────────────────────────────
    trajectory_file.close()
    left_imu_file.close()
    right_imu_file.close()
    arm_angle_file.close()

    print(f"\n{'='*60}")
    print(f"Experiment completed : {arm_posture}")
    print(f"Posture              : Both arms straight down")
    print(f"Trajectory data      : {output_dir}/trajectory.csv")
    print(f"Left hand IMU        : {output_dir}/left_hand_imu.csv")
    print(f"Right hand IMU       : {output_dir}/right_hand_imu.csv")
    print(f"Arm joint angles     : {output_dir}/arm_joint_angles.csv")
    print(f"{'='*60}\n")