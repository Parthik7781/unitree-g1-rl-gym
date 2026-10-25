import time
import sys
import os

# Add parent directories to path so we can import legged_gym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import csv

from legged_gym import LEGGED_GYM_ROOT_DIR


LEFT_KNEE_IDX = 3
RIGHT_KNEE_IDX = 9

# Optional left-leg joints if you also want hip/ankle restriction.
LEFT_HIP_PITCH_IDX = 0
LEFT_ANKLE_PITCH_IDX = 4
RIGHT_HIP_PITCH_IDX = 6
RIGHT_ANKLE_PITCH_IDX = 10


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


def apply_leg_restriction(
    target_dof_pos,
    default_angles,
    rom_scale,
    restrict_left_hip_ankle=False,
    bilateral=False,
):
    """Scale selected leg-joint excursions around default pose.

    When bilateral is False, only left knee is restricted (optionally left hip/ankle pitch).
    When bilateral is True, both knees are restricted (optionally both hip/ankle pitch).
    """
    restricted = target_dof_pos.copy()

    restricted_indices = [LEFT_KNEE_IDX]
    if bilateral:
        restricted_indices.append(RIGHT_KNEE_IDX)

    if restrict_left_hip_ankle:
        restricted_indices.extend([LEFT_HIP_PITCH_IDX, LEFT_ANKLE_PITCH_IDX])
        if bilateral:
            restricted_indices.extend([RIGHT_HIP_PITCH_IDX, RIGHT_ANKLE_PITCH_IDX])

    for idx in restricted_indices:
        delta = restricted[idx] - default_angles[idx]
        restricted[idx] = default_angles[idx] + rom_scale * delta

    return restricted


def save_single_knee_plot(times, left_knee_rad, right_knee_rad, output_path):
    left_deg = np.rad2deg(np.array(left_knee_rad))
    right_deg = np.rad2deg(np.array(right_knee_rad))

    plt.figure(figsize=(11, 6))
    plt.plot(times, left_deg, color="tab:red", linewidth=2.0, label="Left Knee")
    plt.plot(times, right_deg, color="tab:blue", linewidth=2.0, label="Right Knee")
    plt.title("Left vs Right Knee Joint Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (degrees)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_symmetry_index(left_vals_deg, right_vals_deg):
    left = np.array(left_vals_deg, dtype=np.float64)
    right = np.array(right_vals_deg, dtype=np.float64)
    denom = 0.5 * (left + right)
    denom = np.where(np.abs(denom) < 1e-8, np.nan, denom)
    si = (np.abs(left - right) / np.abs(denom)) * 100.0
    return si


def save_symmetry_plot(times, symmetry_index, output_path):
    plt.figure(figsize=(11, 6))
    plt.plot(times, symmetry_index, color="tab:green", linewidth=2.0, label="Symmetry Index")
    plt.title("Knee Joint Symmetry Index vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Symmetry Index (%)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_knee_data_csv(times, left_knee_rad, right_knee_rad, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "left_knee_deg", "right_knee_deg"])
        for t, l_rad, r_rad in zip(times, left_knee_rad, right_knee_rad):
            writer.writerow([f"{t:.6f}", f"{np.rad2deg(l_rad):.6f}", f"{np.rad2deg(r_rad):.6f}"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument(
        "--rom_scale",
        type=float,
        default=0.55,
        help="Left-leg ROM scaling factor around default pose (0.55 ~= 45 percent reduction)",
    )
    parser.add_argument(
        "--restrict_percent",
        type=float,
        default=None,
        help="Restriction percent. Example: 25 means 25 percent ROM reduction (equivalent rom_scale=0.75)",
    )
    parser.add_argument(
        "--restrict_left_hip_ankle",
        action="store_true",
        help="Also restrict left hip pitch and left ankle pitch with the same ROM scale",
    )
    parser.add_argument(
        "--bilateral",
        action="store_true",
        help="Apply restriction to both legs instead of left leg only",
    )
    parser.add_argument(
        "--plot_file",
        type=str,
        default="left_vs_right_knee_angle.png",
        help="Output file for the single overlay plot",
    )
    parser.add_argument(
        "--si_plot_file",
        type=str,
        default="knee_symmetry_index.png",
        help="Output file for the knee symmetry index plot",
    )
    parser.add_argument(
        "--knee_data_file",
        type=str,
        default="knee_data.csv",
        help="Output CSV file for time and left/right knee angles",
    )
    args = parser.parse_args()

    if args.restrict_percent is not None:
        args.rom_scale = 1.0 - (args.restrict_percent / 100.0)

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
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

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    times = []
    left_knee_angles = []
    right_knee_angles = []

    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            tau = pd_control(target_dof_pos, d.qpos[7:19], kps, np.zeros_like(kds), d.qvel[6:18], kds)
            d.ctrl[:len(tau)] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:19]
                dqj = d.qvel[6:18]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()

                unconstrained_target = action * action_scale + default_angles
                target_dof_pos = apply_leg_restriction(
                    unconstrained_target,
                    default_angles,
                    rom_scale=args.rom_scale,
                    restrict_left_hip_ankle=args.restrict_left_hip_ankle,
                    bilateral=args.bilateral,
                )

                current_time = counter * simulation_dt
                times.append(current_time)
                left_knee_angles.append(float(d.qpos[7 + LEFT_KNEE_IDX]))
                right_knee_angles.append(float(d.qpos[7 + RIGHT_KNEE_IDX]))

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    save_single_knee_plot(times, left_knee_angles, right_knee_angles, args.plot_file)
    left_deg = np.rad2deg(np.array(left_knee_angles))
    right_deg = np.rad2deg(np.array(right_knee_angles))
    si = compute_symmetry_index(left_deg, right_deg)
    save_symmetry_plot(times, si, args.si_plot_file)

    finite_si = si[np.isfinite(si)]
    if finite_si.size > 0:
        print(f"Mean Symmetry Index: {np.mean(finite_si):.3f}%")
    save_knee_data_csv(times, left_knee_angles, right_knee_angles, args.knee_data_file)
    print(f"Saved plot: {args.plot_file}")
    print(f"Saved plot: {args.si_plot_file}")
    print(f"Saved knee data: {args.knee_data_file}")
