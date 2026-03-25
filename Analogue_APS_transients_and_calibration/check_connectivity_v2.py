import sys
import os
from pytestlab import Bench
import signal

def handler(signum, frame):
    raise Exception("Connection timed out!")

# Set a 10s timeout for the whole script
signal.signal(signal.SIGALRM, handler)
signal.alarm(10)

project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
os.chdir(project_root)

print("INITIATING HARDWARE PROBE...")
sys.stdout.flush()

try:
    with Bench.open("config/bench.yaml") as bench:
        print(f"BENCH_NAME: {bench.name}")
        sys.stdout.flush()
        print(f"OSC_ID: {bench.osc.id()}")
        sys.stdout.flush()
        print(f"STATUS: CONNECTIVITY_OK")
        sys.stdout.flush()
except Exception as e:
    print(f"ERROR: {e}")
    sys.stdout.flush()
