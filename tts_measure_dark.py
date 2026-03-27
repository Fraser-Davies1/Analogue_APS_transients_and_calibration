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

def capture_tts_ramp(bench, label, duration_s=10.0):
    print(f"\n>>> INITIATING DEEP DARK CAPTURE: {label}")
    sys.stdout.flush()
    
    # 1. Scope Setup: Extreme Sensitivity
    # 10mV/div, 1s/div -> 10s window
    bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="DC").enable()
    bench.osc.set_time_axis(scale=duration_s/10.0, position=duration_s/2.0)
    
    bench.siggen.set_output_state(1, "OFF")
    
    print(f"    Stabilizing for 2s...")
    time.sleep(2.0) 
    
    print(f"    Recording {duration_s}s integration ramp...")
    sys.stdout.flush()
    
    bench.osc._send_command(":SINGle")
    time.sleep(duration_s + 1.0)
    
    data = bench.osc.read_channels([1])
    v = data.values["Channel 1 (V)"].to_numpy()
    t = data.values["Time (s)"].to_numpy()
    
    res = stats.linregress(t, v)
    return t, v, res.slope, res.rvalue**2

def run_dark_measurement():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/data", exist_ok=True)
    
    C_integrator = 11.0e-12 
    
    try:
        with Bench.open("config/bench.yaml") as bench:
            t, v, slope, r2 = capture_tts_ramp(bench, "DARK_ENCLOSED_BLUETACK")
            i_dark = slope * C_integrator
            
            # Save raw data
            np.savez("results/data/tts_dark_enclosed.npz", t=t, v=v, slope=slope, current=i_dark)
            
            print(f"\n" + "="*40)
            print("      FINAL DARK CURRENT RESULTS      ")
            print("="*40)
            print(f"Derived Dark Current: {i_dark*1e12:10.2f} pA")
            print(f"Ramp Linearity (R²):  {r2:10.4f}")
            print(f"Measured Ramp Rate:   {slope*1e3:10.2f} mV/s")
            print("="*40)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(t, v, color='black', alpha=0.8)
            plt.title(f"TTS Dark Current (Bluetack Enclosed)\nDerived I_dark: {i_dark*1e12:.1f} pA")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.grid(True, alpha=0.3)
            
            plot_path = "results/plots/final_dark_current_tts.png"
            plt.savefig(plot_path)
            print(f"PLOT SAVED: {plot_path}")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_dark_measurement()
