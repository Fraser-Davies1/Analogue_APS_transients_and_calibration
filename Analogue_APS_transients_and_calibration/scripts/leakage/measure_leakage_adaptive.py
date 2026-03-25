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

def measure_point(bench, t):
    # Reset
    bench.siggen.channel(1).setup_dc(offset=5.0).enable()
    time.sleep(1.0)
    
    # Measure V_start (ensure we are in a safe range)
    bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
    time.sleep(0.5)
    v_start = float(bench.osc.measure_rms_voltage(1).values)
    if v_start > 10: v_start = 5.0
    
    # Integrate
    bench.siggen.channel(1).setup_dc(offset=0.0).enable()
    time.sleep(t)
    
    # Sample V_end (with auto-range check)
    v_end_raw = bench.osc.measure_rms_voltage(1).values
    v_end = float(v_end_raw.nominal_value if hasattr(v_end_raw, "nominal_value") else v_end_raw)
    
    if v_end > 10: # If clipped, try coarser
        bench.osc.channel(1).setup(scale=1.0, offset=2.5)
        time.sleep(0.5)
        v_end = float(bench.osc.measure_rms_voltage(1).values)
        
    return v_start, v_end

def run_adaptive_leakage():
    print("--- Starting Adaptive Leakage Measurement ---")
    C_INT = 11e-12 
    t_ints = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(30000)
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off()

        results = []
        for t in t_ints:
            print(f"  Testing T_int = {t}s...", end="\r")
            v1, v2 = measure_point(bench, t)
            delta_v = v1 - v2
            # Sanity check: if negative (noise), clamp to small positive or zero
            if delta_v < 0: delta_v = 0
            
            i_pa = (C_INT * (delta_v / t)) * 1e12
            results.append({"t_int": t, "i_pa": i_pa, "delta_v": delta_v})
            print(f"  T: {t:5}s | ΔV: {delta_v*1000:8.2f}mV | I: {i_pa:8.2f}pA")

        res_df = pl.DataFrame(results)
        res_df.write_csv("leakage_adaptive_results.csv")
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(res_df["t_int"], res_df["i_pa"], 'ms-', label='Leakage Current (pA)')
        plt.xlabel("Integration Time (s) [Log Scale]")
        plt.ylabel("Current (pA)")
        plt.title("APS Leakage Characterisation (Adaptive Range)")
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig("leakage_adaptive_report.png")
        print("\n--- Success! Report: leakage_adaptive_report.png ---")

if __name__ == "__main__":
    run_adaptive_leakage()
