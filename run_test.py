import os
import subprocess

os.chdir("/home/coder/project/Analogue_APS_transients_and_calibration")
cmd = [".venv/bin/python3", "scripts/noise/measure_perfect_psd_v2.py", "ungrounded_noise_sweep.png"]
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
except Exception as e:
    print("Error:", e)
