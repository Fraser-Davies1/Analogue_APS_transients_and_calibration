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

def run_mixed_leakage():
    print("--- Starting Mixed-Mode Leakage Sweep (1ms to 10s) ---")
    C_INT = 11e-12 
    t_ints = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(150000)
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off()

        # Scope prep
        bench.osc.channel(1).setup(scale=0.005, offset=4.9, coupling="DC").enable()
        
        results = []

        for t in t_ints:
            print(f"\n--- Testing T_int: {t:.3f} s ---")
            
            # --- DIFFERENTIAL STATIC METHOD (Robust for slow times) ---
            # 1. RESET
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()
            time.sleep(1.0)
            v_start_res = bench.osc.measure_rms_voltage(1)
            v_start = float(v_start_res.values.nominal_value if hasattr(v_start_res.values, "nominal_value") else v_start_res.values)
            
            if v_start > 10.0: v_start = 4.9 # Fail-safe
            
            # 2. INTEGRATE
            bench.siggen.channel(1).setup_dc(offset=0.0).enable()
            print(f"  Integrating for {t}s...")
            time.sleep(t)
            
            # 3. SAMPLE
            v_end_res = bench.osc.measure_rms_voltage(1)
            v_end = float(v_end_res.values.nominal_value if hasattr(v_end_res.values, "nominal_value") else v_end_res.values)
            
            delta_v = v_start - v_end
            i_pa = (C_INT * (delta_v / t)) * 1e12
            
            print(f"  V_start: {v_start:.4f}V | V_end: {v_end:.4f}V")
            print(f"  Drop: {delta_v*1000:.3f}mV | I_leak: {i_pa:.3f}pA")
            
            results.append({"t_int": t, "delta_v_mv": delta_v * 1000, "i_leak_pa": i_pa})
            
            # Reset for next point
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()

        # 2. Reporting
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("leakage_mixed_results.csv")
            
            plt.figure(figsize=(10, 6))
            plt.semilogx(res_df["t_int"], res_df["i_leak_pa"], 'bo-', label='Derived I_leak (pA)')
            plt.xlabel("Integration Time (s) [Log Scale]")
            plt.ylabel("Leakage Current (pA)")
            plt.title("APS Background Leakage: 1ms to 10s Range")
            plt.grid(True, which="both", alpha=0.3)
            plt.savefig("leakage_mixed_report.png")
            print("\n--- Complete. Report: leakage_mixed_report.png ---")
        else:
            print("\n[ERROR] No data points captured.")

if __name__ == "__main__":
    run_mixed_leakage()
