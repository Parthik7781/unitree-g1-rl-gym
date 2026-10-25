# Unitree G1 Comparative Gait Analysis System

## Overview

This system performs a comprehensive comparative analysis of the Unitree G1 humanoid robot's gait under different arm postures. The analysis collects hand IMU sensor data and trajectory information for each arm configuration.

## Arm Postures Tested

### P1_Natural_Walk
- **Configuration**: Natural arm swing
- **Description**: Baseline gait with arms in natural swinging motion
- **Biomechanics**: Normal arm-leg coordination patterns

### P2_Arms_Down
- **Configuration**: Both arms straight down along the body
- **Description**: Arms held tightly at sides
- **Biomechanics**: Tests gait without arm swing compensation

### P3_Arms_Forward
- **Configuration**: Both arms extended forward approximately 90 degrees
- **Description**: Both arms pointing forward
- **Biomechanics**: Tests forward weight redistribution

### P4_Arms_Sideways
- **Configuration**: Both arms extended sideways approximately 90 degrees (T-pose)
- **Description**: Both arms extended to sides
- **Biomechanics**: Tests lateral balance compensation

### P5_Arms_Up
- **Configuration**: Both arms raised overhead
- **Description**: Both arms raised above head
- **Biomechanics**: Tests upward weight redistribution

## Data Collection

### Hand IMU Sensors
Each experiment collects sensor data from both hands:
- **Acceleration**: ax, ay, az (m/s²)
- **Angular Velocity**: gx, gy, gz (rad/s)
- **Sampling Rate**: 50 Hz

### Robot Trajectory
- **X Position**: Global X coordinate
- **Y Position**: Global Y coordinate
- **Time**: Elapsed time in seconds
- **Sampling Rate**: 50 Hz

## Output Structure

```
Experimental_graphs/
├── P1_Natural_Walk/
│   ├── trajectory.csv              # Robot path coordinates
│   ├── left_hand_imu.csv           # Left hand sensor data
│   ├── right_hand_imu.csv          # Right hand sensor data
│   └── gait_summary.png            # Summary figure with 4 subplots
├── P2_Arms_Down/
│   ├── trajectory.csv
│   ├── left_hand_imu.csv
│   ├── right_hand_imu.csv
│   └── gait_summary.png
├── P3_Arms_Forward/
│   ├── trajectory.csv
│   ├── left_hand_imu.csv
│   ├── right_hand_imu.csv
│   └── gait_summary.png
├── P4_Arms_Sideways/
│   ├── trajectory.csv
│   ├── left_hand_imu.csv
│   ├── right_hand_imu.csv
│   └── gait_summary.png
├── P5_Arms_Up/
│   ├── trajectory.csv
│   ├── left_hand_imu.csv
│   ├── right_hand_imu.csv
│   └── gait_summary.png
├── Arm_Posture_Comparison.png      # Comprehensive comparison figure
└── GAIT_ANALYSIS_SUMMARY.txt       # Analysis findings
```

## Summary Figures

### Per-Experiment Figure (gait_summary.png)
Each experiment produces a single figure with 4 subplots:

1. **XY Trajectory**
   - Robot path in 2D space
   - Start and end point markers
   - Trajectory direction arrows

2. **Hand Acceleration Magnitude**
   - Left and right hand acceleration magnitudes over time
   - Clear distinction between hands
   - Peaks indicate high-intensity motion

3. **Hand Angular Velocity Magnitude**
   - Left and right hand rotation rates over time
   - Shows hand twisting and rotation patterns
   - Indicates rotational motion intensity

4. **Metrics Summary Panel**
   - Path length (total distance traveled)
   - Average walking speed
   - RMS and peak accelerations for both hands
   - RMS and peak angular velocities for both hands

### Comparison Figure (Arm_Posture_Comparison.png)
Single comprehensive figure comparing all 5 experiments:

1. **Combined XY Trajectories**
   - All 5 trajectories overlaid
   - Color-coded by posture
   - Shows relative path differences

2. **Acceleration Magnitude Comparison**
   - Hand acceleration over time for all postures
   - Identifies acceleration patterns
   - Reveals motion stability

3. **Angular Velocity Magnitude Comparison**
   - Hand rotation rates for all postures
   - Identifies twisting patterns
   - Shows rotational stability

4. **Comparison Table**
   - Side-by-side metrics for all 5 postures
   - Easy numerical comparison
   - Identifies extremes and patterns

## Computed Metrics

For each experiment:

- **Path Length**: Total distance traveled by robot center
- **Average Speed**: Mean walking velocity
- **RMS Acceleration**: Root mean square of hand acceleration magnitudes
- **Peak Acceleration**: Maximum instantaneous acceleration
- **RMS Angular Velocity**: Root mean square of rotation rates
- **Peak Angular Velocity**: Maximum instantaneous rotation rate

## Running the Analysis

### Single Experiment
```bash
python deploy/deploy_mujoco/deploy_gait_analysis.py g1.yaml --arm_posture P1_Natural_Walk
```

### All Experiments (Automatic)
```bash
python run_gait_analysis.py
```

### Testing System
```bash
python test_gait_analysis.py
```

## Simulation Parameters

- **Duration**: 60 seconds per experiment
- **Timestep**: 0.002 seconds (500 Hz simulation)
- **Control Frequency**: 50 Hz (0.02s control cycle)
- **Walking Speed**: 0.5 m/s forward

## Analysis Outputs

### GAIT_ANALYSIS_SUMMARY.txt
Concise text summary including:

1. Most stable motion (lowest hand acceleration)
2. Highest hand motion (highest acceleration)
3. Largest trajectory deviation (longest path)
4. Highest hand acceleration (peak values)
5. Highest hand angular velocity (peak rotation rates)
6. Detailed metrics table

## Technical Details

### IMU Data Collection
- Hand position extracted from wrist link geometry
- Acceleration estimated from velocity derivatives
- Angular velocity from body frame rotation rates

### Trajectory Data
- Extracted from pelvis (floating base) X-Y coordinates
- Represents robot center of mass motion
- Global coordinate frame

### Data Processing
- All CSV files use comma-separated format
- Floating-point values to 6 decimal places
- Includes header row with field names

## Requirements

- Python 3.8+
- MuJoCo (with Unitree G1 models)
- PyTorch (for walking policy)
- Pandas, NumPy, Matplotlib (for analysis)
- YAML (for configuration)

## Notes

- The current implementation uses the 12-DOF walking controller (legs only)
- Arm postures are represented as configuration parameters  
- Hand motion is captured from the kinematic chain (attached to torso)
- All experiments use identical walking controller parameters
- Only arm posture varies between experiments

## Future Enhancements

1. Support for 29-DOF model with active arm control
2. Per-posture arm actuation profiles
3. Real-time arm angle tracking
4. Extended analysis periods for longer-term stability
5. Step-by-step gait phase analysis
6. Force distribution analysis

---

Generated for Unitree G1 MuJoCo Simulation Project
Comparative Gait Analysis Framework
