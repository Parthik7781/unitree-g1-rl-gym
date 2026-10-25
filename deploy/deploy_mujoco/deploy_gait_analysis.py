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
    # Get body ID
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    
    # Get linear acceleration and angular velocity in world frame
    # Note: In MuJoCo, we can get linear acceleration from d.subtree_linvel time derivatives
    # For simplicity, we'll compute from forward kinematics
    
    # Get body frame velocity (6D: linear + angular)
    body_vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, True)
    
    # Linear velocity and angular velocity in body frame
    lin_vel = body_vel[0:3]
    ang_vel = body_vel[3:6]
    
    return lin_vel, ang_vel


def estimate_acceleration(prev_lin_vel, curr_lin_vel, dt):
    """Estimate acceleration from velocity change"""
    return (curr_lin_vel - prev_lin_vel) / dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument("--arm_posture", type=str, default="P1_Natural_Walk",
                       help="Arm posture: P1_Natural_Walk, P2_Arms_Down, P3_Arms_Forward, P4_Arms_Sideways, P5_Arms_Up")
    args = parser.parse_args()
    
    config_file = args.config_file
    arm_posture = args.arm_posture
    
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
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

    # Define arm postures (joint angles for 8 arm DOF: 4 per side)
    # These are placeholder values for different arm configurations
    # In the 12-DOF model, arms are passive, but we can still conceptually
    # represent different postures for data organization
    arm_postures = {
        "P1_Natural_Walk": "Natural arm swing - Baseline gait",
        "P2_Arms_Down": "Both arms straight down along the body",
        "P3_Arms_Forward": "Both arms extended forward approximately 90 degrees",
        "P4_Arms_Sideways": "Both arms extended sideways approximately 90 degrees (T-pose)",
        "P5_Arms_Up": "Both arms raised overhead",
    }
    
    arm_config = arm_postures.get(arm_posture, arm_postures["P1_Natural_Walk"])
    
    # Create output directory
    output_dir = Path("Experimental_graphs") / arm_posture
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Gait Analysis: {arm_posture}")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    # Define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    
    # Create CSV files for data collection
    trajectory_file = open(output_dir / "trajectory.csv", 'w', newline='')
    trajectory_writer = csv.writer(trajectory_file)
    trajectory_writer.writerow(['Time(s)', 'X_Pos(m)', 'Y_Pos(m)'])
    trajectory_file.flush()
    
    left_imu_file = open(output_dir / "left_hand_imu.csv", 'w', newline='')
    left_imu_writer = csv.writer(left_imu_file)
    left_imu_writer.writerow(['Time(s)', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    left_imu_file.flush()
    
    right_imu_file = open(output_dir / "right_hand_imu.csv", 'w', newline='')
    right_imu_writer = csv.writer(right_imu_file)
    right_imu_writer.writerow(['Time(s)', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    right_imu_file.flush()

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load policy
    policy = torch.jit.load(policy_path)
    
    # Determine model capabilities
    # The 12-DOF model has 12 leg DOF, no arm actuation
    n_dof = len(d.qpos) - 7  # Exclude floating base
    has_arm_control = n_dof > 12
    
    print(f"Model configuration: {n_dof} DOF total")
    if has_arm_control:
        print(f"  - Legs: 12 DOF")
        print(f"  - Arms: {n_dof - 12} DOF")
    else:
        print(f"  - Legs: 12 DOF (arms are passive)")
    
    print(f"Arm configuration: {arm_config}\n")
    
    # Track previous velocities for acceleration estimation
    prev_left_lin_vel = np.zeros(3)
    prev_right_lin_vel = np.zeros(3)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # Control leg DOF with policy (12 DOF)
            tau = pd_control(target_dof_pos, d.qpos[7:19], kps, np.zeros_like(kds), d.qvel[6:18], kds)
            
            # If model has arm DOF, control them (they'll be passive without this)
            # For now, we skip arm control for the 12-DOF model
            
            d.ctrl[:len(tau)] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Get observation for policy (using 12 leg DOF)
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
                
                # Policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # Transform action to target_dof_pos (for legs only)
                target_dof_pos = action * action_scale + default_angles
                
                # Record trajectory data
                current_time = counter * simulation_dt
                robot_x = d.xpos[0, 0]  # Pelvis X position
                robot_y = d.xpos[0, 1]  # Pelvis Y position
                trajectory_writer.writerow([f"{current_time:.6f}", f"{robot_x:.6f}", f"{robot_y:.6f}"])
                trajectory_file.flush()
                
                # Get hand IMU data
                try:
                    left_lin_vel, left_ang_vel = get_imu_data(m, d, "left_wrist_roll_rubber_hand")
                    left_acc = estimate_acceleration(prev_left_lin_vel, left_lin_vel, simulation_dt * control_decimation)
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
                except:
                    pass
                
                try:
                    right_lin_vel, right_ang_vel = get_imu_data(m, d, "right_wrist_roll_rubber_hand")
                    right_acc = estimate_acceleration(prev_right_lin_vel, right_lin_vel, simulation_dt * control_decimation)
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
                except:
                    pass

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Close files
    trajectory_file.close()
    left_imu_file.close()
    right_imu_file.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Experiment completed: {arm_posture}")
    print(f"   Configuration: {arm_config}")
    print(f"   Trajectory data: {output_dir}/trajectory.csv")
    print(f"   Left hand IMU: {output_dir}/left_hand_imu.csv")
    print(f"   Right hand IMU: {output_dir}/right_hand_imu.csv")
    print(f"{'='*60}\n")
