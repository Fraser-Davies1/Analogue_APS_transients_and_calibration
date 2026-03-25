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

def run_current_time_sweep(label="Encased"):
    print(f"--- Starting Leakage Sweep: {label} Mode ---")
    C_INT = 11e-12 
    t_ints = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(60000)
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off() # LED Off for both runs

        # Scope prep
        bench.osc.channel(1).setup(scale=0.005, offset=4.8, coupling="DC").enable()
        
        results = []

        for t in t_ints:
            print(f"  T_int = {t:4.1f}s | Sampling...", end="\r")
            
            # Reset
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()
            time.sleep(1.0)
            v_start = float(bench.osc.measure_rms_voltage(1).values)
            
            # Integrate
            bench.siggen.channel(1).setup_dc(offset=0.0).enable()
            time.sleep(t)
            
            # Sample
            v_end = float(bench.osc.measure_rms_voltage(1).values)
            
            delta_v = v_start - v_end
            if delta_v < 0: delta_v = 0
            
            # Derive Current (pA)
            i_pa = (C_INT * (delta_v / t)) * 1e12
            
            results.append({"t_int": t, "delta_v_mv": delta_v * 1000, "i_pa": i_pa})
            
            # Clean up Reset
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()

        df = pl.DataFrame(results)
        df.write_csv(f"current_vs_time_{label.lower()}.csv")
        
        print(f"\n--- {label} Run Complete ---")
        return df

def plot_comparison(df1, df2=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df1["t_int"], df1["i_pa"], 'bo-', label='Encased (Blu-Tac)')
    if df2 is not None:
        plt.plot(df2["t_int"], df2["i_pa"], 'rs-', label='Open (Ambient)')
    
    plt.xlabel("Integration Time (s)")
    plt.ylabel("Derived Leakage Current (pA)")
    plt.title("Current vs. Integration Time Characterization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("current_vs_time_report.png")
    print("Combined plot saved: current_vs_time_report.png")

if __name__ == "__main__":
    df_encased = run_current_time_sweep("Encased")
    plot_comparison(df_encased)
    print("\n[ACTION REQUIRED] Please remove the Blu-Tac from the sensor.")
    print("Then type 'python3 measure_current_vs_time.py' and I will run the Open sweep.")
