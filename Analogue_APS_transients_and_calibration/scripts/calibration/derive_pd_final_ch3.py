import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_automated_derivation():
    print("--- Starting Automated PD Derivation Sweep ---")
    print("--- Configuration: CH1=Pixel, CH3=Reset Ref, C_int=11pF ---")
    
    # R_eff for 220 || 50 ohm probe
    R_EFF = 40.74 
    C_INT = 11e-12

    with Bench.open("bench.yaml") as bench:
        # Hardware Setup
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(2).setup(scale=0.1, offset=0.0, coupling="DC").enable() # For I_LED
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Reset Ref
            
            bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Timing and Rails synchronized.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")

        results = []
        # Focus on the active region
        v_steps = np.linspace(2.35, 4.5, 20)

        for v in v_steps:
            print(f"  Advancing to V_LED = {v:.2f}V...", end="\r")
            try:
                # Attempt to set PSU (with retry)
                bench.psu.channel(2).set(voltage=v).on()
                time.sleep(0.5)
            except:
                pass
            
            # Capture CH1 (Pixel), CH2 (Resistor Drop), CH3 (Reset Sync)
            data = bench.osc.read_channels([1, 2, 3])
            df = data.values
            t_vec = df["Time (s)"].to_numpy()
            v_px = df["Channel 1 (V)"].to_numpy()
            v_shunt = df["Channel 2 (V)"].to_numpy()
            v_rs = df["Channel 3 (V)"].to_numpy()
            
            # 1. Calculate Real LED Current for this step
            i_led_ma = (np.mean(v_shunt) / R_EFF) * 1000.0
            
            # 2. Derive PD Current using CH3 timing
            edges = np.diff((v_rs > 2.5).astype(int))
            falls = np.where(edges == -1)[0]
            rises = np.where(edges == 1)[0]
            
            if len(falls) > 0 and len(rises) > 0:
                idx_s = falls[0]
                idx_e = rises[rises > idx_s][0]
                
                t_int = t_vec[idx_e] - t_vec[idx_s]
                v_start = np.mean(v_px[max(0, idx_s-5):idx_s])
                v_end = v_px[idx_e]
                
                delta_v = v_start - v_end
                i_pd_na = (C_INT * (delta_v / t_int)) * 1e9
                
                results.append({
                    "v_in": v,
                    "i_led_ma": i_led_ma,
                    "i_pd_na": i_pd_na
                })

        # 3. Final Report
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("pd_final_characterization.csv")
            
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["i_led_ma"], res_df["i_pd_na"], 'ms-', label='Measured I_pd')
            plt.xlabel("Actual LED Current (mA)")
            plt.ylabel("Derived Photodiode Current (nA)")
            plt.title(f"Photodiode Sensitivity Curve (C_int=11pF, CH3 Sync)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig("pd_current_final_report.png")
            print(f"\n--- Characterization Complete. Plot: pd_current_final_report.png ---")
        else:
            print("\n[ERROR] No data captured during sweep.")

if __name__ == "__main__":
    run_automated_derivation()
