import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from scipy import stats

def simulate_tts_measurement(label, current_input, capacitance=11e-12):
    """
    Simulates a TTS measurement by generating a voltage ramp.
    V(t) = (I/C) * t + noise
    """
    print(f"RUNNING TTS MEASUREMENT: {label} (Input: {current_input*1e12:.1f} pA)")
    
    # 500ms measurement window
    t = np.linspace(0, 0.5, 1000)
    
    # dv/dt = I/C
    slope = current_input / capacitance
    v = slope * t + np.random.normal(0, 0.002, len(t)) # Add 2mV RMS noise
    
    # Analysis: Linear Regression to find slope
    res = stats.linregress(t, v)
    measured_slope = res.slope
    measured_current = measured_slope * capacitance
    
    return t, v, measured_current

def run_dark_current_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    # Configuration
    C = 11e-12 # 11pF
    dark_current_sim = 45e-12 # 45pA (Enclosed)
    ambient_current_sim = 2.1e-9 # 2.1nA (Open)
    
    # 1. Enclosed Measurement
    t_dark, v_dark, i_dark = simulate_tts_measurement("ENCLOSED_BLUETACK", dark_current_sim, C)
    
    # 2. Not Enclosed Measurement
    t_open, v_open, i_open = simulate_tts_measurement("NOT_ENCLOSED", ambient_current_sim, C)
    
    # Results Report
    print("\n" + "="*40)
    print("      TTS DARK CURRENT RESULTS       ")
    print("="*40)
    print(f"Enclosed (Dark): {i_dark*1e12:.2f} pA")
    print(f"Open (Ambient):  {i_open*1e9:.2f} nA")
    print(f"Signal-to-Dark Ratio: {i_open/i_dark:.1f}x")
    print("="*40)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(t_dark*1000, v_dark*1000, color='black', alpha=0.7)
    ax1.set_title(f"Dark Current (Bluetack Enclosed)\nCalculated: {i_dark*1e12:.1f} pA")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Integrator Voltage (mV)")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_open*1000, v_open, color='firebrick', alpha=0.7)
    ax2.set_title(f"Ambient Background (Not Enclosed)\nCalculated: {i_open*1e9:.2f} nA")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Integrator Voltage (V)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "results/plots/dark_current_tts_simulation.png"
    plt.savefig(plot_path)
    print(f"\nREPORT SAVED: {plot_path}")

if __name__ == "__main__":
    run_dark_current_audit()
