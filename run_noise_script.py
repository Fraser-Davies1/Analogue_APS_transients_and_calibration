import os
import subprocess
import sys

project_dir = "/home/coder/project/Analogue_APS_transients_and_calibration"
script_path = "scripts/noise/measure_perfect_psd_v2.py"
output_png = "current_noise_sweep.png"

os.chdir(project_dir)
cmd = ["uv", "run", "python", script_path, output_png]
print(f"Executing: {' '.join(cmd)}")

result = subprocess.run(cmd, capture_output=True, text=True)

print("--- STDOUT ---")
print(result.stdout)
print("--- STDERR ---")
print(result.stderr)

if result.returncode == 0:
    print("Execution Successful")
else:
    print(f"Execution Failed with code {result.returncode}")
