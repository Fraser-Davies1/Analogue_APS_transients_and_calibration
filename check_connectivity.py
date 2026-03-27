from pytestlab import Bench
import os

project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
os.chdir(project_root)

print("Attempting to connect to REAL hardware...")
try:
    with Bench.open("config/bench.yaml") as bench:
        print(f"Connected to Bench: {bench.name}")
        print(f"Osc ID: {bench.osc.id()}")
        print(f"PSU ID: {bench.psu.id()}")
        print(f"SigGen ID: {bench.siggen.id()}")
        print("CONNECTIVITY_OK")
except Exception as e:
    print(f"FAILED to connect: {e}")
