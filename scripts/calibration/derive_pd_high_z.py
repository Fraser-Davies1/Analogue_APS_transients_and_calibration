import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_high_z_derivation():
    print("--- Starting High-Z PD Derivation Sweep ---")
    print("--- CH1=Pixel, CH2=I_sense (220 Ohm), CH3=Reset Ref ---")
    
    R_SENSE = 220.0 
    C_INT = 11e-12

    with Bench.open("bench.yaml") as bench:
        # Hardware Setup
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            # Setup Scope - ENSURE CH2 IS 1M OHM (High-Z)
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(2).setup(scale=0.2, offset=0.0, coupling="DC").enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
            
            bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Hardware ready.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")

        results = []
        v_steps = np.linspace(2.35, 4.5, 20)

        for v in v_steps:
            print(f"  Measuring V_LED = {v:.2f}V...", end="\r")
            try:
                bench.psu.channel(2).set(voltage=v).on()
                time.sleep(0.5)
            except: pass
            
            # Capture all 3 channels
            data = bench.osc.read_channels([1, 2, 3])
            df = data.values
            
            t = df["Time (s)"].to_numpy()
            v_px = df["Channel 1 (V)"].to_numpy()
            v_i = df["Channel 2 (V)"].to_numpy()
            v_rs = df["Channel 3 (V)"].to_numpy()
            
            # 1. Measure Actual Stimulus Current (Average of whole trace)
            i_led_ma = (np.mean(v_i) / R_SENSE) * 1000.0
            
            # 2. Measure Integration Timing on CH3
            edges = np.diff((v_rs > 2.5).astype(int))
            f_pts = np.where(edges == -1)[0]
            r_pts = np.where(edges == 1)[0]
            
            if len(f_pts) > 0 and len(r_pts) > 0:
                idx_s = f_pts[0]
                idx_e = r_pts[r_pts > idx_s][0]
                
                t_int = t[idx_e] - t[idx_s]
                v_start = np.mean(v_px[max(0, idx_s-5):idx_s])
                v_end = v_px[idx_e]
                
                delta_v = v_start - v_end
                i_pd_na = (C_INT * (delta_v / t_int)) * 1e9
                
                results.append({
                    "i_led_ma": i_led_ma,
                    "i_pd_na": i_pd_na
                })

        # 3. Save and Plot
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("pd_high_z_results.csv")
            
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["i_led_ma"], res_df["i_pd_na"], 'bo-', label='Derived I_pd (nA)')
            plt.xlabel("Actual LED Current (mA)")
            plt.ylabel("Derived Photodiode Current (nA)")
            plt.title(f"Photodiode Sensitivity: Corrected for Unloaded 220 Ohm\nC_int = 11pF")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig("pd_current_high_z.png")
            print(f"\n--- Success! Data saved to pd_high_z_results.csv ---")
        else:
            print("\n[ERROR] Sweep failed to capture valid windows.")

if __name__ == "__main__":
    run_high_z_derivation()
