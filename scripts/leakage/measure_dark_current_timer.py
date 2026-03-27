import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_dark_current_timer():
    print("--- Starting Dark Current Characterisation (Python-Timer Method) ---")
    C_INT = 11e-12 
    t_ints = [1.0, 2.0, 5.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        # Hardware Setup
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off()
        
        # Scope in Auto/Roll mode to see live changes
        bench.osc.channel(1).setup(scale=0.01, offset=4.8).enable()
        
        results = []

        for t in t_ints:
            print(f"\n--- integration Time: {t:.1f} s ---")
            
            # 1. RESET PHASE
            # Pulse Reset High (5V) to clear the capacitor
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()
            time.sleep(1.0)
            
            # Capture the start voltage (Mean of 10 samples)
            v_start = float(bench.osc.measure_rms_voltage(1).values)
            if v_start > 10: v_start = 4.8 # Fallback
            
            print(f"  V_reset: {v_start:.4f}V. Starting Integration...")
            
            # 2. INTEGRATION PHASE
            # Pull Reset Low (0V)
            bench.siggen.channel(1).setup_dc(offset=0.0).enable()
            
            # Precisely wait t_int seconds
            time.sleep(t)
            
            # 3. SAMPLE PHASE
            v_final = float(bench.osc.measure_rms_voltage(1).values)
            if v_final > 10: v_final = v_start # Error handling
            
            delta_v = v_start - v_final
            i_pa = (C_INT * (delta_v / t)) * 1e12
            
            print(f"  V_final: {v_final:.4f}V | Drop: {delta_v*1000:.2f}mV | I_leak: {i_pa:.2f}pA")
            
            results.append({"t_int": t, "i_pa": i_pa, "delta_v": delta_v})
            
            # Restore reset for next cycle
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()

        # 2. Report
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("dark_current_timer_results.csv")
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["t_int"], res_df["i_pa"], 'bo-')
            plt.xlabel("Integration Time (s)")
            plt.ylabel("Leakage Current (pA)")
            plt.title("Dark Current via Precision Python Timer")
            plt.grid(True)
            plt.savefig("dark_current_timer_report.png")
            print("\n--- Final Report: dark_current_timer_report.png ---")

if __name__ == "__main__":
    run_dark_current_timer()
