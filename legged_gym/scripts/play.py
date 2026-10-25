import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

# IMU Sensor data structure for G1 lower body
class G1IMUSensor:
    def __init__(self):
        # G1 joint indices for lower body
        self.left_knee_idx = None
        self.right_knee_idx = None
        self.left_thigh_idx = None  # hip_pitch
        self.right_thigh_idx = None  # hip_pitch
        
    def get_sensor_readings(self, env, env_id=0):
        """Extract IMU and joint sensor readings from knee and thigh"""
        readings = {}
        
        # Base IMU readings (angular velocity and gravity)
        readings['imu_ang_vel'] = env.base_ang_vel[env_id].cpu().numpy()
        readings['projected_gravity'] = env.projected_gravity[env_id].cpu().numpy()
        readings['base_lin_vel'] = env.base_lin_vel[env_id].cpu().numpy()
        readings['base_ang_vel_rad'] = env.base_ang_vel[env_id].cpu().numpy()
        
        # Joint-specific readings - Knee joints (indices 3, 9 in 12DOF G1)
        left_knee_pos = env.dof_pos[env_id, 3].cpu().numpy()
        left_knee_vel = env.dof_vel[env_id, 3].cpu().numpy()
        right_knee_pos = env.dof_pos[env_id, 9].cpu().numpy()
        right_knee_vel = env.dof_vel[env_id, 9].cpu().numpy()
        
        # Joint-specific readings - Thigh joints (hip_pitch - indices 2, 8)
        left_thigh_pos = env.dof_pos[env_id, 2].cpu().numpy()
        left_thigh_vel = env.dof_vel[env_id, 2].cpu().numpy()
        right_thigh_pos = env.dof_pos[env_id, 8].cpu().numpy()
        right_thigh_vel = env.dof_vel[env_id, 8].cpu().numpy()
        
        readings['knee_left'] = {'position': left_knee_pos, 'velocity': left_knee_vel}
        readings['knee_right'] = {'position': right_knee_pos, 'velocity': right_knee_vel}
        readings['thigh_left'] = {'position': left_thigh_pos, 'velocity': left_thigh_vel}
        readings['thigh_right'] = {'position': right_thigh_pos, 'velocity': right_thigh_vel}
        
        return readings
    
    def print_readings(self, readings, step=0):
        """Pretty print sensor readings"""
        print(f"\n=== G1 IMU Sensor Readings (Step: {step}) ===")
        print(f"Base IMU Angular Velocity (rad/s): {readings['imu_ang_vel']}")
        print(f"Base IMU Gravity Vector: {readings['projected_gravity']}")
        print(f"Base Linear Velocity (m/s): {readings['base_lin_vel']}")
        print(f"\n--- Lower Body Knee Sensors ---")
        print(f"Left Knee  - Pos: {readings['knee_left']['position']:.4f} rad, Vel: {readings['knee_left']['velocity']:.4f} rad/s")
        print(f"Right Knee - Pos: {readings['knee_right']['position']:.4f} rad, Vel: {readings['knee_right']['velocity']:.4f} rad/s")
        print(f"\n--- Lower Body Thigh Sensors ---")
        print(f"Left Thigh  - Pos: {readings['thigh_left']['position']:.4f} rad, Vel: {readings['thigh_left']['velocity']:.4f} rad/s")
        print(f"Right Thigh - Pos: {readings['thigh_right']['position']:.4f} rad, Vel: {readings['thigh_right']['velocity']:.4f} rad/s")


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Initialize G1 IMU sensor
    imu_sensor = G1IMUSensor()
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # Capture IMU sensor readings every 10 steps
        if i % 10 == 0:
            readings = imu_sensor.get_sensor_readings(env, env_id=0)
            imu_sensor.print_readings(readings, step=i)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
