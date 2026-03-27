from pytestlab import Bench
import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

# --- Framework Patch: Register WaveformGeneratorConfig ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass
# ---------------------------------------------------------

project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
os.chdir(project_root)

print("Opening Bench...")
try:
    with Bench.open("config/bench_sim.yaml") as bench:
        print(f"Bench Instruments: {list(bench._instrument_instances.keys())}")
        print(f"Osc ID: {bench.osc.id() if hasattr(bench, 'osc') else 'MISSING'}")
        print(f"PSU ID: {bench.psu.id() if hasattr(bench, 'psu') else 'MISSING'}")
        print(f"SigGen ID: {bench.siggen.id() if hasattr(bench, 'siggen') else 'MISSING'}")
except Exception as e:
    import traceback
    traceback.print_exc()
print("Done!")
