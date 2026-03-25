import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from scipy import stats
from pytestlab import Bench

# Framework Fix
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass

def run_tts_with_active_reset():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    C_integrator = 11.0e-12 
    duration_s = 5.0

    print(">>> INITIATING DARK CURRENT MEASUREMENT (ACTIVE RESET)")
    sys.stdout.flush()

    try:
        with Bench.open("config/bench.yaml") as bench:
            # 1. Scope Setup
            bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="DC").enable()
            bench.osc.set_time_axis(scale=duration_s/10.0, position=duration_s/2.0)
            
            # 2. SigGen Active Reset Sequence
            print("    [1/3] Initializing SigGen CH2 (Reset Line)...")
            bench.siggen.set_function(2, "DC")
            bench.siggen.set_offset(2, 0.0)
            bench.siggen.set_output_state(2, "ON") # ACTIVE DRIVE
            time.sleep(0.5)

            print("    [2/3] Pulsing RESET HIGH (5V)...")
            bench.siggen.set_offset(2, 5.0)
            time.sleep(0.5)

            print("    [3/3] Releasing RESET LOW (0V) -> Starting Integration...")
            bench.siggen.set_offset(2, 0.0)
            
            # 3. Synchronized Capture
            bench.osc._send_command(":SINGle")
            time.sleep(duration_s + 0.5)
            
            data = bench.osc.read_channels([1])
            v = data.values["Channel 1 (V)"].to_numpy()
            t = data.values["Time (s)"].to_numpy()
            
            # 4. Analysis
            res = stats.linregress(t, v)
            i_dark = res.slope * C_integrator
            
            print(f"\n" + "="*40)
            print("      DARK CURRENT RESULTS (ACTIVE)      ")
            print("="*40)
            print(f"Derived Dark Current: {i_dark*1e12:10.2f} pA")
            print(f"Ramp Linearity (R²):  {res.rvalue**2:10.4f}")
            print(f"Measured Ramp Rate:   {res.slope*1e3:10.2f} mV/s")
            print("="*40)

            # Cleanup: Hold reset LOW (Don't turn off unless specified)
            # bench.siggen.set_output_state(2, "OFF") 

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_tts_with_active_reset()
