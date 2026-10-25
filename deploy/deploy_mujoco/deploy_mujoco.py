import time
import sys
import os

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


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
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
        arm_target_angles = np.array(
            config.get("arm_target_angles", [0.0] * 8), dtype=np.float32
        )
        arm_kp = float(config.get("arm_kp", 80.0))
        arm_kd = float(config.get("arm_kd", 2.0))

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    
    # Create sensor data file
    sensor_file_path = "upper_body_sensor_data.csv"
    sensor_file = open(sensor_file_path, 'w', newline='')
    sensor_writer = csv.writer(sensor_file)
    sensor_writer.writerow(['Step', 'Time(s)', 'Torso_Pos(rad)', 'Torso_Vel(rad/s)', 
                           'IMU_AngVel_X', 'IMU_AngVel_Y', 'IMU_AngVel_Z',
                           'Gravity_X', 'Gravity_Y', 'Gravity_Z',
                           'BaseLinVel_X', 'BaseLinVel_Y', 'BaseLinVel_Z'])
    sensor_file.flush()

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Build explicit arm control mapping by joint/actuator names.
    # This avoids index-based mistakes (e.g., accidentally commanding waist joints).
    arm_ctrl_map = []
    for idx, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if joint_id >= 0 and actuator_id >= 0:
            arm_ctrl_map.append((idx, actuator_id, int(m.jnt_qposadr[joint_id]), int(m.jnt_dofadr[joint_id])))

    if len(arm_ctrl_map) == 0:
        print("Arm joints not found in this model. Running leg-only control.")
    else:
        print(f"Arm control active on {len(arm_ctrl_map)} joints (shoulders/elbows only).")

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            d.ctrl[:] = 0.0
            tau_legs = pd_control(
                target_dof_pos,
                d.qpos[7 : 7 + num_actions],
                kps,
                np.zeros_like(kds),
                d.qvel[6 : 6 + num_actions],
                kds,
            )
            d.ctrl[:num_actions] = tau_legs

            # Hold both arms down by targeting shoulder/elbow joints only.
            for target_idx, actuator_id, qpos_adr, qvel_adr in arm_ctrl_map:
                q = d.qpos[qpos_adr]
                dq = d.qvel[qvel_adr]
                tq = arm_target_angles[target_idx]
                tau = (tq - q) * arm_kp + (0.0 - dq) * arm_kd
                d.ctrl[actuator_id] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7 : 7 + num_actions]
                dqj = d.qvel[6 : 6 + num_actions]
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
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
                
                # Write sensor readings every 5 steps to CSV file
                if counter % (control_decimation * 5) == 0:
                    torso_pos = d.qpos[6]
                    torso_vel = d.qvel[5]
                    imu_ang_vel = omega / ang_vel_scale
                    base_lin_vel = d.qvel[0:3]
                    
                    sensor_writer.writerow([
                        counter, 
                        counter * simulation_dt,
                        f"{torso_pos:.6f}",
                        f"{torso_vel:.6f}",
                        f"{imu_ang_vel[0]:.6f}",
                        f"{imu_ang_vel[1]:.6f}",
                        f"{imu_ang_vel[2]:.6f}",
                        f"{gravity_orientation[0]:.6f}",
                        f"{gravity_orientation[1]:.6f}",
                        f"{gravity_orientation[2]:.6f}",
                        f"{base_lin_vel[0]:.6f}",
                        f"{base_lin_vel[1]:.6f}",
                        f"{base_lin_vel[2]:.6f}"
                    ])
                    sensor_file.flush()
                    
                    # Also print to console
                    print(f"[Step {counter}] Torso: {torso_pos:.4f} rad | IMU Ang Vel: [{imu_ang_vel[0]:.4f}, {imu_ang_vel[1]:.4f}, {imu_ang_vel[2]:.4f}]", flush=True)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Close the sensor data file
    sensor_file.close()
    print(f"\nSensor data saved to: {sensor_file_path}")
