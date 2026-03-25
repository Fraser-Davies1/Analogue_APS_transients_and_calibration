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

def diag_integrator():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    
    with Bench.open("config/bench.yaml") as bench:
        # Check PSU
        v_psu = bench.psu.read_voltage(1)
        print(f"PSU Voltage: {v_psu:.3f} V")
        
        # Check Scope
        bench.osc.channel(1).setup(scale=0.1, offset=0, coupling="DC").enable()
        time.sleep(0.5)
        data = bench.osc.read_channels([1])
        v_mean = data.values["Channel 1 (V)"].mean()
        v_max = data.values["Channel 1 (V)"].max()
        v_min = data.values["Channel 1 (V)"].min()
        
        print(f"Scope Mean: {v_mean:.3f} V")
        print(f"Scope Max:  {v_max:.3f} V")
        print(f"Scope Min:  {v_min:.3f} V")
        
        if abs(v_mean) > 4.5:
            print("STATUS: Integrator is likely SATURATED at rail.")
        elif abs(v_mean) < 0.05:
            print("STATUS: Integrator output is near zero. Check RESET signal.")
        else:
            print("STATUS: Integrator is in linear range.")

if __name__ == "__main__":
    diag_integrator()
