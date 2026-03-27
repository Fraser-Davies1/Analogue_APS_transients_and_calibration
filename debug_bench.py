import logging
from pytestlab import Bench
import os

logging.basicConfig(level=logging.DEBUG)
project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
os.chdir(project_root)

print("OPENING BENCH...")
try:
    with Bench.open("config/bench.yaml") as bench:
        print(f"Connected: {list(bench._instrument_instances.keys())}")
except Exception as e:
    print(f"ERROR: {e}")
