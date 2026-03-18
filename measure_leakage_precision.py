import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

# --- Patch for WaveformGenerator registration ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except: pass
# ------------------------------------------------

def run_precision_leakage():
    print("--- Starting Ultra-Precision Background Leakage Sweep ---")
    print("--- Strategy: High-Sensitivity Differential Averaging ---")
    
    C_INT = 11e-12 
    # 10 integration times from 0.1s to 20s
    t_ints = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
    
    with Bench.open("bench.yaml") as bench:
        # Extend timeout for 20s windows
        bench.osc._backend.set_timeout(200000) 
        
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off()

        # Initial Scope setup for Reset level
        # 5mV/div + 4.8V offset = [4.775V to 4.825V] visible window
        bench.osc.channel(1).setup(scale=0.005, offset=4.8, coupling="DC").enable()
        # Fast timebase for sampling the steady-state levels
        bench.osc.set_time_axis(scale=10e-3, position=0.0) 

        results = []

        for t in t_ints:
            print(f"\n--- integration Time: {t:.1f} s ---")
            
            # 1. RESET PHASE
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()
            time.sleep(2.0) # Ensure full reset stabilization
            
            print("  Sampling Start Level...", end="\r")
            data_start = bench.osc.read_channels(1)
            v_start = data_start.values["Channel 1 (V)"].mean()
            
            # 2. INTEGRATION PHASE
            bench.siggen.channel(1).setup_dc(offset=0.0).enable()
            print(f"  Integrating for {t}s...      ", end="\r")
            time.sleep(t)
            
            # 3. SAMPLE PHASE
            print("  Sampling End Level...  ", end="\r")
            data_end = bench.osc.read_channels(1)
            v_end = data_end.values["Channel 1 (V)"].mean()
            
            delta_v = v_start - v_end
            
            # Guard: If delta_v is negative (noise-driven), clamp to 0 for log plots
            # but keep raw for calculation
            i_pa = (C_INT * (delta_v / t)) * 1e12
            
            print(f"  V_start: {v_start:.5f}V | V_end: {v_end:.5f}V")
            print(f"  Drop: {delta_v*1000:.3f}mV | I_leak: {i_pa:.4f}pA")
            
            results.append({"t_int": t, "delta_v_mv": delta_v * 1000, "i_leak_pa": i_pa})
            
            # Re-assert Reset
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()

        # 4. Final Processing
        res_df = pl.DataFrame(results)
        res_df.write_csv("leakage_precision_results.csv")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Linear Delta V vs Time (Proves current presence)
        ax1.plot(res_df["t_int"], res_df["delta_v_mv"], 'bo-', label='Measured Drop')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Drop (mV)")
        ax1.set_title("Integration Discharge curve")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Leakage Current vs Time (Proves stability)
        ax2.plot(res_df["t_int"], res_df["i_leak_pa"], 'rs-', label='Derived I_leak')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Leakage Current (pA)")
        ax2.set_title("Leakage Current Consistency")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=-1, top=max(10, res_df["i_leak_pa"].max()*1.5))
        
        plt.tight_layout()
        plt.savefig("leakage_precision_report.png")
        print(f"\n--- Complete. Report saved: leakage_precision_report.png ---")
        print(f"Calculated Mean System Leakage: {res_df['i_leak_pa'].mean():.4f} pA")

if __name__ == "__main__":
    run_precision_leakage()
