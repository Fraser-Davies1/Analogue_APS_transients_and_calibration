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

def perform_hardware_reset(bench):
    """
    Pulses the reset signal to clear the integrator capacitor.
    """
    print("    >>> SENDING RESET SIGNAL (SigGen CH2)...")
    # Configure CH2 for a DC pulse
    bench.siggen.set_function(2, "DC")
    bench.siggen.set_offset(2, 5.0) # 5V Reset High
    bench.siggen.set_output_state(2, "ON")
    time.sleep(0.2)
    bench.siggen.set_output_state(2, "OFF")
    bench.siggen.set_offset(2, 0.0) # Reset Low
    print("    >>> RESET COMPLETE. Integrator floating.")

def capture_tts_with_reset(bench, label, duration_s=5.0):
    print(f"\n>>> INITIATING CAPTURE WITH RESET: {label}")
    sys.stdout.flush()
    
    # 1. Scope Setup
    bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="DC").enable()
    bench.osc.set_time_axis(scale=duration_s/10.0, position=duration_s/2.0)
    
    # 2. Reset the sensor
    perform_hardware_reset(bench)
    
    # 3. Capture Ramp
    print(f"    Recording {duration_s}s integration ramp...")
    bench.osc._send_command(":SINGle")
    time.sleep(duration_s + 0.5)
    
    data = bench.osc.read_channels([1])
    v = data.values["Channel 1 (V)"].to_numpy()
    t = data.values["Time (s)"].to_numpy()
    
    res = stats.linregress(t, v)
    return t, v, res.slope, res.rvalue**2

def run_dark_measurement_corrected():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    C_integrator = 11.0e-12 
    
    try:
        with Bench.open("config/bench.yaml") as bench:
            # Step: Dark Current (Bluetack)
            t, v, slope, r2 = capture_tts_with_reset(bench, "DARK_ENCLOSED_WITH_RESET")
            i_dark = slope * C_integrator
            
            print(f"\n" + "="*40)
            print("      DARK CURRENT RESULTS (WITH RESET)     ")
            print("="*40)
            print(f"Derived Dark Current: {i_dark*1e12:10.2f} pA")
            print(f"Ramp Linearity (R²):  {r2:10.4f}")
            print(f"Measured Ramp Rate:   {slope*1e3:10.2f} mV/s")
            print("="*40)

            plt.figure(figsize=(10, 5))
            plt.plot(t, v, color='black')
            plt.title(f"Dark Current with Hard Reset\nI_dark: {i_dark*1e12:.1f} pA")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.grid(True, alpha=0.3)
            plt.savefig("results/plots/dark_current_with_reset.png")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_dark_measurement_corrected()
