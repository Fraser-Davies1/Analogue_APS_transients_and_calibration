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
    print("--- Starting Log-Scale Precision Leakage Sweep ---")
    C_INT = 11e-12 
    # 10 points across a wide range for better slope resolution
    t_ints = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(250000) 
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off()

        # High-sensitivity Zoom: 2mV/div centered on Reset level
        # This provides the maximum vertical resolution of the ADC
        bench.osc.channel(1).setup(scale=0.002, offset=4.75, coupling="DC").enable()
        bench.osc.set_time_axis(scale=10e-3, position=0.0) 

        results = []

        for t in t_ints:
            print(f"  Measuring T_int = {t:4.1f}s...", end="\r")
            
            # 1. RESET & STABILIZE
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
            # Instantaneous current calculation (includes switching artifact)
            i_eff_pa = (C_INT * (delta_v / t)) * 1e12
            
            results.append({
                "t_int_s": t,
                "v_start": v_start,
                "v_end": v_end,
                "delta_v_mv": delta_v * 1000,
                "i_calc_pa": i_eff_pa
            })
            
            # Re-reset
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()

        # 4. Data Processing
        df = pl.DataFrame(results)
        df.write_csv("leakage_log_results.csv")
        
        # Calculate the "Real" Leakage Slope via Linear Regression on Delta_V vs Time
        x = df["t_int_s"].to_numpy()
        y = df["delta_v_mv"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1) # slope in mV/s
        
        # Real Leakage = Slope * C
        real_leakage_fa = (slope / 1000.0) * C_INT * 1e15
        pedestal_mv = intercept

        # 5. Output Table
        print("\n\n--- Background Leakage Data Table ---")
        print(f"{'T_int (s)':<10} | {'Drop (mV)':<12} | {'I_apparent (pA)':<15}")
        print("-" * 45)
        for row in df.iter_rows(named=True):
            print(f"{row['t_int_s']:10.1f} | {row['delta_v_mv']:12.3f} | {row['i_calc_pa']:15.4f}")
        
        print(f"\n--- Physical Parameter Extraction ---")
        print(f"Switching Pedestal:  {pedestal_mv:.3f} mV")
        print(f"True Leakage Current: {real_leakage_fa:.2f} fA (Femto-Amps)")

        # 6. Log-Log Plotting
        plt.figure(figsize=(10, 7))
        
        # We plot the calculated current vs time on log-log
        # This highlights how the 1/t switching error fades out to reveal the real floor
        plt.loglog(df["t_int_s"], df["i_calc_pa"], 'bo-', label='Apparent Current (Total)')
        
        # Plot the "True Leakage" floor
        plt.axhline(real_leakage_fa/1000.0, color='r', linestyle='--', label=f'True Floor ({real_leakage_fa:.1f} fA)')
        
        plt.xlabel("Integration Time (s) [Log Scale]")
        plt.ylabel("Current (pA) [Log Scale]")
        plt.title("WDR Background Leakage: Apparent Current vs. Integration Time")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.savefig("leakage_loglog_report.png")
        print("\nReport saved: leakage_loglog_report.png")

if __name__ == "__main__":
    run_log_precision_leakage()
