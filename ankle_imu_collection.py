"""
================================================================================
ANKLE IMU SENSOR DATA COLLECTION MODULE
================================================================================

This module simulates and collects IMU sensor data from ankle (foot joint area)
sensors on both legs of the Unitree G1 robot during gait cycles.

The system collects:
- Acceleration (ax, ay, az)
- Gyroscope/Angular Velocity (gx, gy, gz)
- Orientation (roll, pitch, yaw)
- Timestamps

Data is synchronized across both ankles and stored with consistent sampling.

Author: Gait Analysis System
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict


class AnkleIMUSensor:
    """
    Simulates an ankle IMU sensor with realistic gait data.
    
    This generates synthetic but physically plausible ankle sensor data
    based on typical bipedal gait patterns during walking.
    """
    
    def __init__(self, leg_side: str, sample_rate: float = 100.0, seed: int = None):
        """
        Initialize ankle IMU sensor.
        
        Args:
            leg_side (str): 'left' or 'right' - which leg this sensor is on
            sample_rate (float): Sampling frequency in Hz (default 100 Hz)
            seed (int): Random seed for reproducible data generation
        """
        if leg_side.lower() not in ['left', 'right']:
            raise ValueError("leg_side must be 'left' or 'right'")
        
        self.leg_side = leg_side.lower()
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Gait cycle parameters (typical human-like walking)
        self.gait_cycle_duration = 1.0  # seconds (approximately 60 steps/min)
        self.stride_frequency = 1.0 / self.gait_cycle_duration  # Hz
        
        # Phase offset: right leg is 180 degrees out of phase with left leg
        # during normal bipedal gait (alternating steps)
        self.phase_offset = np.pi if self.leg_side == 'right' else 0.0
        
    def generate_gait_data(self, duration: float, start_time: float = 0.0) -> pd.DataFrame:
        """
        Generate synthetic ankle IMU data for the specified duration.
        
        The generated data represents a realistic gait cycle with:
        - Periodic motion patterns
        - Acceleration and gyroscope signals reflecting ankle motion
        - Orientation changes during swing and stance phases
        
        Args:
            duration (float): Duration of data collection in seconds
            start_time (float): Starting time for timestamps
            
        Returns:
            pd.DataFrame: Sensor data with columns:
                - timestamp: absolute time in seconds
                - sample_num: sequential sample number
                - ax, ay, az: linear acceleration (m/s^2)
                - gx, gy, gz: angular velocity (rad/s)
                - roll, pitch, yaw: orientation (radians)
        """
        num_samples = int(duration * self.sample_rate)
        timestamps = np.linspace(start_time, start_time + duration, num_samples, endpoint=False)
        sample_nums = np.arange(num_samples)
        
        # Compute phase of gait cycle for each sample
        # For right leg: start at opposite phase (π radians = 180°)
        gait_phase = (2 * np.pi * self.stride_frequency * timestamps + self.phase_offset) % (2 * np.pi)
        
        # =====================================================================
        # STANCE PHASE vs SWING PHASE
        # =====================================================================
        # Stance phase: 0 to pi (foot in contact with ground)
        # Swing phase: pi to 2*pi (foot in swing/flight)
        
        is_stance = gait_phase < np.pi
        
        # =====================================================================
        # LINEAR ACCELERATION (ankle) - REALISTIC ASYMMETRIC GAIT PATTERN
        # =====================================================================
        # Creates asymmetric acceleration that integrates to tear-drop shape
        # Key: Different accelerations during stance vs swing phases
        # This asymmetry is essential for tear-drop formation
        
        # ANTERIOR-POSTERIOR (X) ACCELERATION
        # Swing phase: strong forward acceleration (2-3 m/s^2)
        # Stance phase: gradual deceleration and push-off (0.5-1 m/s^2)
        ax = np.zeros(num_samples)
        ax[is_stance] = 0.8 * np.sin(gait_phase[is_stance])  # Slow during stance
        ax[~is_stance] = 2.5 * np.sin(np.pi - gait_phase[~is_stance])  # Fast forward during swing
        
        # MEDIAL-LATERAL (Y) ACCELERATION (vertical lift)
        # Foot clearance during swing phase creates characteristic lift
        # Use sharper pulse during swing for asymmetry
        ay = np.zeros(num_samples)
        swing_phase_norm = ((gait_phase[~is_stance] - np.pi) % (2 * np.pi))
        ay[~is_stance] = 3.0 * np.sin(swing_phase_norm)  # Strong lift during swing
        ay[is_stance] = 0.3 * np.cos(gait_phase[is_stance])  # Minimal during stance
        
        # VERTICAL (Z) ACCELERATION
        # Gravity constant (-9.81) + gait modulation
        # Impact during early stance, lighter during swing
        az = -9.81 * np.ones(num_samples)
        az[is_stance] += 2.0 * np.sin(gait_phase[is_stance])  # Impact and push-off
        az[~is_stance] += 0.5 * np.sin(gait_phase[~is_stance])  # Light oscillation in swing
        
        # Add realistic noise (moderate level for robustness)
        ax += np.random.normal(0, 0.08, num_samples)
        ay += np.random.normal(0, 0.08, num_samples)
        az += np.random.normal(0, 0.15, num_samples)
        
        # =====================================================================
        # ANGULAR VELOCITY (gyroscope)
        # =====================================================================
        # Represents ankle rotation during gait
        
        # Pitch (rotation around y-axis: dorsiflexion/plantarflexion)
        # Large during swing phase (knee drives forward)
        # Small during stance (foot mostly stable)
        gy_max = 3.0  # rad/s max angular velocity
        # Key: use phase directly without cosine modulation for proper phase shift
        gy = gy_max * np.sin(gait_phase)
        
        # Roll (rotation around x-axis: inversion/eversion)
        # Small but important for stability
        gx = 0.5 * np.sin(2 * gait_phase)
        
        # Yaw (rotation around z-axis: transverse plane rotation)
        # Very small during normal walking
        gz = 0.2 * np.sin(gait_phase)
        
        # Add realistic noise
        gx += np.random.normal(0, 0.05, num_samples)
        gy += np.random.normal(0, 0.08, num_samples)
        gz += np.random.normal(0, 0.03, num_samples)
        
        # =====================================================================
        # ORIENTATION (Euler angles)
        # =====================================================================
        # Represents the absolute orientation of the ankle
        
        # Roll: dorsiflexion angle (primary motion)
        # Range approximately -30 to +20 degrees
        roll = 0.3 * np.sin(gait_phase) + 0.1 * np.cos(2 * gait_phase)  # radians
        
        # Pitch: ankle inversion/eversion
        # Small oscillations
        pitch = 0.05 * np.sin(2 * gait_phase)
        
        # Yaw: transverse plane rotation
        # Very small, mostly due to body rotation
        yaw = 0.02 * np.sin(gait_phase)
        
        # =====================================================================
        # CREATE DATAFRAME
        # =====================================================================
        data = pd.DataFrame({
            'timestamp': timestamps,
            'sample_num': sample_nums,
            'ax': ax,
            'ay': ay,
            'az': az,
            'gx': gx,
            'gy': gy,
            'gz': gz,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'gait_phase': gait_phase,  # Internal: gait cycle phase (0 to 2*pi)
        })
        
        return data
    
    def add_measurement_noise(self, data: pd.DataFrame, noise_level: float = 0.01) -> pd.DataFrame:
        """
        Add realistic IMU measurement noise.
        
        Args:
            data (pd.DataFrame): Sensor data
            noise_level (float): Relative noise level (0.0-1.0)
            
        Returns:
            pd.DataFrame: Data with added noise
        """
        data_noisy = data.copy()
        
        # Add Gaussian noise to accelerometer
        for ax_col in ['ax', 'ay', 'az']:
            noise = np.random.normal(0, np.std(data[ax_col]) * noise_level, len(data))
            data_noisy[ax_col] += noise
        
        # Add Gaussian noise to gyroscope
        for g_col in ['gx', 'gy', 'gz']:
            noise = np.random.normal(0, np.std(data[g_col]) * noise_level * 0.5, len(data))
            data_noisy[g_col] += noise
        
        return data_noisy


class AnkleGaitDataCollector:
    """
    Collects synchronized IMU data from both ankle sensors.
    """
    
    def __init__(self, sample_rate: float = 100.0, seed: int = None):
        """
        Initialize ankle gait data collector.
        
        Args:
            sample_rate (float): Sampling frequency in Hz
            seed (int): Random seed for reproducible data generation
        """
        self.sample_rate = sample_rate
        self.seed = seed
        self.left_sensor = AnkleIMUSensor('left', sample_rate, seed=seed)
        self.right_sensor = AnkleIMUSensor('right', sample_rate, seed=seed)
        
    def collect(self, duration: float, num_gait_cycles: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect synchronized data from both ankle sensors.
        
        Args:
            duration (float): Duration per gait cycle in seconds
            num_gait_cycles (int): Number of gait cycles to collect
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (left_ankle_data, right_ankle_data)
        """
        total_duration = duration * num_gait_cycles
        
        # Generate synchronized data
        left_data = self.left_sensor.generate_gait_data(total_duration, start_time=0.0)
        right_data = self.right_sensor.generate_gait_data(total_duration, start_time=0.0)
        
        # Add realistic noise
        left_data = self.left_sensor.add_measurement_noise(left_data, noise_level=0.02)
        right_data = self.right_sensor.add_measurement_noise(right_data, noise_level=0.02)
        
        return left_data, right_data


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_ankle_data(left_data: pd.DataFrame, right_data: pd.DataFrame, output_dir: str):
    """
    Save ankle sensor data to CSV files.
    
    Args:
        left_data (pd.DataFrame): Left ankle sensor data
        right_data (pd.DataFrame): Right ankle sensor data
        output_dir (str): Output directory path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Columns to save (excluding internal gait_phase column)
    save_cols = ['timestamp', 'sample_num', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw']
    
    left_path = Path(output_dir) / 'left_ankle.csv'
    right_path = Path(output_dir) / 'right_ankle.csv'
    
    left_data[save_cols].to_csv(str(left_path), index=False)
    right_data[save_cols].to_csv(str(right_path), index=False)
    
    print(f"[DATA] Saved left ankle data: {left_path}")
    print(f"[DATA] Saved right ankle data: {right_path}")


if __name__ == "__main__":
    # Example usage
    print("Collecting ankle gait data...")
    collector = AnkleGaitDataCollector(sample_rate=100.0)
    left_data, right_data = collector.collect(duration=1.0, num_gait_cycles=3)
    
    print(f"Left ankle: {len(left_data)} samples")
    print(f"Right ankle: {len(right_data)} samples")
    print(f"\nLeft ankle data:\n{left_data.head()}")
    print(f"\nRight ankle data:\n{right_data.head()}")
