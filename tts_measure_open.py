import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from scipy import stats

# Framework Fix
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass

from pytestlab import Bench

def capture_tts_ramp(bench, label, duration_s=5.0):
    print(f"\n>>> INITIATING LONG CAPTURE: {label}")
    sys.stdout.flush()
    
    # Use 20mV/div for better sensitivity
    bench.osc.channel(1).setup(scale=0.02, offset=0, coupling="DC").enable()
    bench.osc.set_time_axis(scale=duration_s/10.0, position=duration_s/2.0)
    
    bench.siggen.set_output_state(1, "OFF")
    
    time.sleep(1.0) 
    print(f"    Recording {duration_s}s ramp...")
    sys.stdout.flush()
    
    bench.osc._send_command(":SINGle")
    time.sleep(duration_s + 1.0)
    
    data = bench.osc.read_channels([1])
    v = data.values["Channel 1 (V)"].to_numpy()
    t = data.values["Time (s)"].to_numpy()
    
    res = stats.linregress(t, v)
    return t, v, res.slope, res.rvalue**2

def run_open_measurement():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    
    C_integrator = 11.0e-12 
    
    try:
        with Bench.open("config/bench.yaml") as bench:
            t, v, slope, r2 = capture_tts_ramp(bench, "OPEN_AMBIENT")
            i_derived = slope * C_integrator
            
            print(f"\n>>> RESULTS (OPEN/AMBIENT)")
            print(f"    Derived Current: {i_derived*1e12:.2f} pA")
            print(f"    Linearity (R²):  {r2:.4f}")
            print(f"    Ramp Rate:       {slope*1e3:.2f} mV/s")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_open_measurement()
