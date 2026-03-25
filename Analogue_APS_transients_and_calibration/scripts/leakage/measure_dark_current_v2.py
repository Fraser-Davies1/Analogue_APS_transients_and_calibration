import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_dark_current_v2():
    print("--- Starting Dark Current Characterisation (Robust Slow Mode) ---")
    C_INT = 11e-12 
    
    # Target integration times
    t_ints = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        # 1. Hardware Initialization
        # Set long timeout for slow transfers
        bench.osc._backend.set_timeout(40000) 
        
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off()

        # Scope Prep: Zoomed on 4.8V (The Reset Level)
        bench.osc.channel(1).setup(scale=0.05, offset=4.8).enable() 
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        results = []

        for t in t_ints:
            print(f"\n--- integration Time: {t:.1f} s ---")
            
            period = 2.0 * t
            freq = 1.0 / period
            
            # Update SigGen
            bench.siggen.channel(1).setup_square(frequency=freq, amplitude=5.0, offset=2.5).enable()
            
            # Update Scope Timebase
            # Total window = 10 Divisions. We want to see 1.2 periods.
            div_scale = (1.2 * period) / 10.0
            bench.osc.set_time_axis(scale=div_scale, position=period/2.0)
            
            # Wait for acquisition to stabilize (2 periods)
            print(f"  Waiting {period*2:.1f}s for waveform stabilization...")
            time.sleep(period * 2.0)
            
            try:
                # Capture current screen buffer (avoiding blocking DIGITIZE if possible)
                # We query trace data directly
                data = bench.osc.read_channels([1, 3])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 3 (V)"].to_numpy()
                
                # Logic: find Falling edge (Integration Start) and Rising edge (Integration Stop)
                is_high = v_rs > 2.5
                edges = np.diff(is_high.astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    # Find rising edge after the falling one
                    potential_rises = rises[rises > idx_s]
                    if len(potential_rises) > 0:
                        idx_e = potential_rises[0]
                        
                        actual_t = t_vec[idx_e] - t_vec[idx_s]
                        v_start = np.mean(v_px[max(0, idx_s-10):idx_s])
                        v_end = np.mean(v_px[max(0, idx_e-10):idx_e])
                        
                        delta_v = v_start - v_end
                        i_pa = (C_INT * (delta_v / actual_t)) * 1e12
                        
                        results.append({"t_int": actual_t, "i_pa": i_pa, "delta_v": delta_v})
                        print(f"  SUCCESS: Drop={delta_v*1000:.1f}mV, I_leak={i_pa:.2f}pA")
                        continue

                print("  [WARN] Window not detected in this frame.")
            except Exception as e:
                print(f"  [ERROR] Capture failed: {e}")

        # 2. Results
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("dark_current_v2.csv")
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["t_int"], res_df["i_pa"], 'ro-')
            plt.xlabel("Integration Time (s)")
            plt.ylabel("Leakage Current (pA)")
            plt.title("Dark Current vs. Integration Time (Stability Check)")
            plt.grid(True, alpha=0.3)
            plt.savefig("dark_current_v2_report.png")
            print("\n--- Final Report: dark_current_v2_report.png ---")
        else:
            print("\n--- Failed to capture any data points. Check CH3 connection. ---")

if __name__ == "__main__":
    run_dark_current_v2()
