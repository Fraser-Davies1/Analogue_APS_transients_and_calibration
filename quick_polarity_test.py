import numpy as np
import os
import time
import sys
from pytestlab import Bench

# Framework Fix
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass

def quick_polarity_check():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    
    print(">>> INITIATING POLARITY QUICK-TEST")
    sys.stdout.flush()

    try:
        with Bench.open("config/bench.yaml") as bench:
            # 1. Prepare Scope
            bench.osc.channel(1).setup(scale=0.5, offset=0, coupling="DC").enable()
            bench.osc.set_time_axis(scale=0.1, position=0.5) 
            
            # 2. Baseline
            bench.siggen.set_output_state(1, "OFF")
            time.sleep(0.5)
            v_start = bench.osc.read_channels([1]).values["Channel 1 (V)"].mean()
            print(f"    Baseline Voltage (LED OFF): {v_start:.3f} V")
            
            # 3. Stimulus
            print("    Applying LED Stimulus...")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_frequency(1, 500)
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen.set_output_state(1, "ON")
            
            time.sleep(1.0)
            
            v_active = bench.osc.read_channels([1]).values["Channel 1 (V)"].mean()
            print(f"    Active Voltage (LED ON):   {v_active:.3f} V")
            
            delta = v_active - v_start
            print(f"\n>>> RESULTS")
            print(f"    Voltage Delta: {delta:.3f} V")
            
            if abs(delta) > 0.02:
                direction = "POSITIVE" if delta > 0 else "NEGATIVE"
                print(f"    STATUS: [OK] Photodiode is responding. Ramp is {direction}.")
            else:
                print("    STATUS: [FAIL] No significant response.")
                print("    ACTION: Flip photodiode and re-run.")

            bench.siggen.set_output_state(1, "OFF")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    quick_polarity_check()
