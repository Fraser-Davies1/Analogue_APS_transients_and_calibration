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

def run_log_precision_leakage():
    print("--- Starting Log-Scale Precision Leakage Sweep (v2) ---")
    C_INT = 11e-12 
    t_ints = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(250000) 
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off()

        # Set to 5mV/div - the hardware limit for maximum precision
        # Offset to 4.8V (Reset level)
        bench.osc.channel(1).setup(scale=0.005, offset=4.8, coupling="DC").enable()
        bench.osc.set_time_axis(scale=10e-3, position=0.0) 

        results = []

        for t in t_ints:
            print(f"  Measuring T_int = {t:4.1f}s...", end="\r")
            
            # 1. RESET
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()
            time.sleep(2.0) 
            data_start = bench.osc.read_channels(1)
            v_start = data_start.values["Channel 1 (V)"].mean()
            
            # 2. INTEGRATE
            bench.siggen.channel(1).setup_dc(offset=0.0).enable()
            time.sleep(t)
            
            # 3. SAMPLE
            data_end = bench.osc.read_channels(1)
            v_end = data_end.values["Channel 1 (V)"].mean()
            
            delta_v = v_start - v_end
            i_eff_pa = (C_INT * (delta_v / t)) * 1e12
            
            results.append({
                "t_int_s": t,
                "v_start": v_start,
                "v_end": v_end,
                "delta_v_mv": delta_v * 1000,
                "i_calc_pa": i_eff_pa
            })
            
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()

        df = pl.DataFrame(results)
        df.write_csv("leakage_log_v2_results.csv")
        
        # Regression for Slope
        x = df["t_int_s"].to_numpy()
        y = df["delta_v_mv"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        
        real_leakage_fa = (slope / 1000.0) * C_INT * 1e15

        # Table Output
        print("\n\n--- Background Leakage Data Table (5mV/div Resolution) ---")
        print(f"{'T_int (s)':<10} | {'Drop (mV)':<12} | {'I_apparent (pA)':<15}")
        print("-" * 45)
        for row in df.iter_rows(named=True):
            print(f"{row['t_int_s']:10.1f} | {row['delta_v_mv']:12.3f} | {row['i_calc_pa']:15.4f}")
        
        print(f"\n--- Physical Parameters ---")
        print(f"Switching Pedestal:  {intercept:.3f} mV")
        print(f"True Leakage Current: {real_leakage_fa:.2f} fA")

        # Plotting
        plt.figure(figsize=(10, 7))
        # Filter for positive apparent current for log plot
        clean_df = df.filter(pl.col("i_calc_pa") > 0.0001)
        plt.loglog(clean_df["t_int_s"], clean_df["i_calc_pa"], 'bo-', label='Apparent Current (includes artifact)')
        plt.axhline(real_leakage_fa/1000.0 if real_leakage_fa > 0 else 1e-6, 
                    color='r', linestyle='--', label=f'True Leakage Floor ({real_leakage_fa:.1f} fA)')
        
        plt.xlabel("Integration Time (s) [Log Scale]")
        plt.ylabel("Current (pA) [Log Scale]")
        plt.title("WDR Background Leakage: Artifact vs. Real Current")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.savefig("leakage_loglog_v2_report.png")
        print("\nReport saved: leakage_loglog_v2_report.png")

if __name__ == "__main__":
    run_log_precision_leakage()
