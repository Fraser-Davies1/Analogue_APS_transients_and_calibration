import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_wdr_family_sweep():
    print("--- Starting WDR Family Sweep (Semilog-X) ---")
    
    R_SENSE = 220.0
    V_REF = 1.0 # Comparator threshold
    
    t_ints = [0.0005, 0.005, 0.05, 0.5, 5.0] # 0.5ms to 5s
    
    # Point distribution to ensure it completes in reasonable time
    v_fine = np.arange(2.2, 2.8, 0.02)
    v_coarse = np.arange(2.8, 4.8, 0.2)
    v_points = np.unique(np.concatenate([v_fine, v_coarse]))

    with Bench.open("bench.yaml") as bench:
        # Initial Hardware Sync
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
        bench.osc.channel(2).setup(scale=0.1, offset=0.0).enable() # I_led sense
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Reset sync
        
        master_results = []

        for t_window in t_ints:
            f = 1.0 / t_window
            print(f"\n--- Integration Time: {t_window*1000:.1f}ms ({f:.2f} Hz) ---")
            
            # Setup Timing
            bench.siggen.channel(1).setup_square(frequency=f, amplitude=5.0, offset=2.5).enable()
            # View 2 cycles
            bench.osc.set_time_axis(scale=(2*t_window)/10.0, position=t_window)
            time.sleep(1.5)

            for idx, v in enumerate(v_points):
                try:
                    bench.psu.channel(2).set(voltage=v).on()
                    # Wait for integration (longer for slow scales)
                    time.sleep(max(0.2, t_window * 1.1))
                    
                    data = bench.osc.read_channels([1, 2, 3])
                    df = data.values
                    t_vec = df["Time (s)"].to_numpy()
                    v_px = df["Channel 1 (V)"].to_numpy()
                    v_i = df["Channel 2 (V)"].to_numpy()
                    v_rs = df["Channel 3 (V)"].to_numpy()
                    
                    # 1. Ground Truth Current
                    i_led_ma = (np.mean(v_i) / R_SENSE) * 1000.0
                    
                    # 2. TTS Logic (Comparator Timing)
                    edges = np.diff((v_rs > 2.5).astype(int))
                    falls = np.where(edges == -1)[0]
                    rises = np.where(edges == 1)[0]
                    
                    if len(falls) > 0 and len(rises) > 0:
                        idx_s = falls[0]
                        idx_e = rises[rises > idx_s][0]
                        t0, v_start = t_vec[idx_s], v_px[idx_s]
                        
                        win_v = v_px[idx_s:idx_e]
                        win_t = t_vec[idx_s:idx_e]
                        
                        trigger = np.where(win_v <= V_REF)[0]
                        if len(trigger) > 0:
                            # TTS Latch
                            hit = trigger[0]
                            # Sub-sample crossing
                            v2, v1 = win_v[hit], win_v[max(0, hit-1)]
                            t2, t1 = win_t[hit], win_t[max(0, hit-1)]
                            t_cross = t1 + (V_REF-v1)*(t2-t1)/(v2-v1) if v1 != v2 else t2
                            t_sat = max(1e-9, t_cross - t0)
                            v_out = (v_start - V_REF) * (t_window / t_sat)
                        else:
                            # Direct integration
                            v_out = v_start - win_v[-1]
                        
                        master_results.append({
                            "t_int_ms": t_window * 1000,
                            "i_led_ma": i_led_ma,
                            "v_out_inferred": v_out
                        })
                        
                    if idx % 5 == 0:
                        print(f"  V_in: {v:.2f}V | I: {i_led_ma:.2f}mA | V_out: {v_out:.1f}V", end="\r")
                except: continue

        # 3. Process and Plot
        res_df = pl.DataFrame(master_results)
        res_df.write_csv("pixel_wdr_family_data.csv")
        
        plt.figure(figsize=(12, 8))
        # Group by integration time and plot raw lines
        for t in sorted(res_df["t_int_ms"].unique(), reverse=True):
            sub = res_df.filter(pl.col("t_int_ms") == t).sort("i_led_ma")
            # Filter for positive current to avoid log errors
            sub = sub.filter(pl.col("i_led_ma") > 0.01)
            plt.semilogx(sub["i_led_ma"], sub["v_out_inferred"], 'o-', markersize=3, label=f"T = {t:.1f}ms")
            
        plt.xlabel("LED Current (mA) [Log Scale]")
        plt.ylabel("Inferred Output Voltage (V) [Linear Scale]")
        plt.title("Pixel WDR Characterization: Family of Curves (TTS Method)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(title="Integration Time")
        plt.tight_layout()
        plt.savefig("pixel_wdr_semilog_family.png")
        print("\n--- Sweep Complete. Plot: pixel_wdr_semilog_family.png ---")

if __name__ == "__main__":
    run_wdr_family_sweep()
