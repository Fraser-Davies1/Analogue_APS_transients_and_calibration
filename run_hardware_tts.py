import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from scipy import stats
from pytestlab import Bench

def capture_tts_ramp(bench, label, duration_s=1.0):
    """
    Captures a single integration ramp from the oscilloscope.
    Configures the scope for high resolution and slow timebase to capture the slope.
    """
    print(f"\n>>> INITIATING CAPTURE: {label}")
    
    # 1. Scope Setup for Ramp Capture
    # 100ms/div * 10 div = 1.0s window
    bench.osc.channel(1).setup(scale=0.05, offset=0, coupling="DC").enable()
    bench.osc.set_time_axis(scale=duration_s/10.0, position=duration_s/2.0)
    
    # Ensure trigger is auto or high enough to catch the drift
    # For TTS we often just force a digitize to catch the integrator state
    time.sleep(0.5) 
    
    print(f"    Recording {duration_s}s of integration data...")
    bench.osc._send_command(":DIGitize CHANnel1")
    data = bench.osc.read_channels([1])
    
    v = data.values["Channel 1 (V)"].to_numpy()
    t = data.values["Time (s)"].to_numpy()
    
    # 2. Linear Regression to find dV/dt
    res = stats.linregress(t, v)
    
    return t, v, res.slope, res.rvalue**2

def run_tts_hardware_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 # 11pF
    
    results = {}
    
    with Bench.open("config/bench.yaml") as bench:
        # Step 1: Enclosed (Dark Current)
        input("\n[ACTION] Ensure Photodiode is ENCLOSED in Bluetack. Press [ENTER] to measure dark current...")
        t_dark, v_dark, slope_dark, r2_dark = capture_tts_ramp(bench, "DARK_ENCLOSED")
        i_dark = slope_dark * C_integrator
        results['dark'] = {'t': t_dark, 'v': v_dark, 'i': i_dark, 'r2': r2_dark}
        print(f"    Dark Current: {i_dark*1e12:.2f} pA (R²={r2_dark:.4f})")

        # Step 2: Not Enclosed (Ambient)
        input("\n[ACTION] REMOVE Bluetack / OPEN Photodiode. Press [ENTER] to measure ambient current...")
        t_open, v_open, slope_open, r2_open = capture_tts_ramp(bench, "AMBIENT_OPEN")
        i_open = slope_open * C_integrator
        results['open'] = {'t': t_open, 'v': v_open, 'i': i_open, 'r2': r2_open}
        print(f"    Ambient Current: {i_open*1e9:.2f} nA (R²={r2_open:.4f})")

    # Final Report Generation
    print("\n" + "="*50)
    print("             HARDWARE TTS AUDIT COMPLETE             ")
    print("="*50)
    print(f"Measured Dark Current:   {results['dark']['i']*1e12:10.2f} pA")
    print(f"Measured Ambient Current: {results['open']['i']*1e9:10.2f} nA")
    print(f"Signal-to-Dark Ratio:    {results['open']['i']/results['dark']['i']:10.1f}x")
    print("="*50)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(results['dark']['t'], results['dark']['v'], label=f"Dark ({results['dark']['i']*1e12:.1f} pA)", color='black')
    plt.plot(results['open']['t'], results['open']['v'], label=f"Ambient ({results['open']['i']*1e9:.1f} nA)", color='firebrick')
    plt.title("Hardware TTS Measurement: Dark vs Ambient Integration Ramps")
    plt.xlabel("Time (s)")
    plt.ylabel("Integrator Voltage (V)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = "results/plots/hardware_dark_current_tts.png"
    plt.savefig(plot_path)
    print(f"\nREPORT SAVED: {plot_path}")

if __name__ == "__main__":
    run_tts_hardware_audit()
