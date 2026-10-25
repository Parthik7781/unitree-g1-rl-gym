"""
QUICK LAUNCHER - Run the complete integrated pipeline

This script executes:
1. MuJoCo simulation → collects sensor data to collected_sensor_data/run_XXX/
2. Analyzes the data → saves results to project_outputs/run_XXX/

Run with: python launch_collection.py
"""

import subprocess
import sys

print("\n" + "="*80)
print("LAUNCHING: Sensor Collection + Analysis Pipeline")
print("="*80 + "\n")

result = subprocess.run([sys.executable, "integrated_pipeline.py"])
sys.exit(result.returncode)
