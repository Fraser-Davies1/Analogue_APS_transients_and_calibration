from pytestlab import Bench
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print(f"Current working directory: {os.getcwd()}")
print("Attempting to connect to REAL hardware...")

# Try to find a bench.yaml
bench_path = "bench.yaml"
if not os.path.exists(bench_path):
    bench_path = "config/bench.yaml"

if not os.path.exists(bench_path):
    print("Could not find bench.yaml in root or config/")
    sys.exit(1)

print(f"Using bench config: {bench_path}")

try:
    with Bench.open(bench_path) as bench:
        print(f"Connected to Bench: {bench.name}")
        print(f"Osc ID: {bench.osc.id()}")
        print(f"PSU ID: {bench.psu.id()}")
        print(f"SigGen ID: {bench.siggen.id()}")
        print("CONNECTIVITY_OK")
except Exception as e:
    print(f"FAILED to connect: {e}")
