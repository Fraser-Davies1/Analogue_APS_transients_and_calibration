import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_dark_current_sweep():
    print("--- Starting Dark Current Characterisation (0.5s to 10s) ---")
    C_INT = 11e-12 # 11pF
    
    # Target integration times
    t_ints = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        # 1. Hardware Prep
        # Ensure LED is OFF
        try:
            bench.psu.channel(2).off()
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            print("  PSU: VDD=5V, LED=OFF.")
        except:
            print("  [WARN] PSU control failed. Ensure LED is manually off and VDD=5V.")

        # Setup Scope
        bench.osc.channel(1).setup(scale=0.1, offset=4.8).enable() # Zoom on reset level
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Reset Ref
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        results = []

        for t in t_ints:
            print(f"\n--- integration Time: {t:.1f} s ---")
            
            # Reset Timing: Period = 2*T to allow for full reset/discharge cycle
            period = 2.0 * t
            freq = 1.0 / period
            
            try:
                bench.siggen.channel(1).setup_square(frequency=freq, amplitude=5.0, offset=2.5).enable()
                
                # Scope window to see 1.5 cycles
                # Total window = 1.5 * period. Div = total / 10.
                scope_scale = (1.5 * period) / 10.0
                bench.osc.set_time_axis(scale=scope_scale, position=period/2.0)
                
                # Wait for hardware to cycle once
                time.sleep(period * 1.5)
                
                print(f"  Capturing trace...")
                data = bench.osc.read_channels([1, 3])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 3 (V)"].to_numpy()
                
                # Edge detection to find integration window
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    idx_e = rises[rises > idx_s][0]
                    
                    actual_t_int = t_vec[idx_e] - t_vec[idx_s]
                    v_start = np.mean(v_px[max(0, idx_s-20):idx_s])
                    v_end = np.mean(v_px[max(0, idx_e-20):idx_e])
                    
                    delta_v = v_start - v_end
                    # Leakage current calculation
                    i_dark_pa = (C_INT * (delta_v / actual_t_int)) * 1e12
                    
                    results.append({
                        "t_int_s": actual_t_int,
                        "delta_v": delta_v,
                        "i_dark_pa": i_dark_pa
                    })
                    print(f"  Drop: {delta_v:.4f}V | I_dark: {i_dark_pa:.2f} pA")
                else:
                    print("  [ERROR] Reset edges not found.")
            except Exception as e:
                print(f"  [ERROR] Measurement failed: {e}")

        # 2. Save and Plot
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("dark_current_results.csv")
            
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["t_int_s"], res_df["i_dark_pa"], 'bo-', label='Measured I_dark')
            plt.xlabel("Integration Time (s)")
            plt.ylabel("Leakage Current (pA)")
            plt.title(f"Dark Current Stability (C_int = 11pF)\nMean I_dark: {res_df['i_dark_pa'].mean():.2f} pA")
            plt.grid(True, alpha=0.3)
            plt.ylim(bottom=0)
            plt.savefig("dark_current_report.png")
            print("\n--- Complete. Report: dark_current_report.png ---")
        else:
            print("\n[ERROR] No data captured.")

if __name__ == "__main__":
    run_dark_current_sweep()
