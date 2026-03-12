import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_2d_wdr():
    print("--- Starting 2D Wide Dynamic Range (TTS) Sweep ---")
    
    # 1. Current Mapping
    iv_data = pl.read_csv("led_iv_high_res.csv")
    def get_current(v_psu):
        return np.interp(v_psu, iv_data["v_in"], iv_data["i_ma"])

    # 2. Sweep Parameters
    t_ints = [1.0, 0.1, 0.01, 0.001, 0.0001] # 1s down to 100us
    v_steps = np.linspace(2.35, 4.5, 30) # High-intensity focus

    with Bench.open("bench.yaml") as bench:
        # Hardware Setup
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.2).on()
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable()
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        master_data = []

        for t in t_ints:
            f = 1.0 / t
            print(f"\n--- Integration Time: {t*1000:.2f} ms ({f:.2f} Hz) ---")
            
            # Sync timing and scope window
            bench.siggen.channel(1).setup_square(frequency=f, amplitude=5.0, offset=2.5).enable()
            # Position scope to catch the start of integration
            scope_scale = (2 * t) / 10.0
            bench.osc.set_time_axis(scale=scope_scale, position=t)
            time.sleep(1.2)

            for v in v_steps:
                curr = get_current(v)
                print(f"  I_LED: {curr:.2f} mA", end="\r")
                
                try:
                    bench.psu.channel(2).set(voltage=v).on()
                    time.sleep(max(0.3, t * 1.1)) 
                    
                    data = bench.osc.read_channels([1, 3])
                    df = data.values
                    time_vec, v_px, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 3 (V)"].to_numpy()
                    
                    # Identify Integration Period
                    edges = np.diff((v_rs > 2.5).astype(int))
                    falls = np.where(edges == -1)[0]
                    rises = np.where(edges == 1)[0]
                    
                    if len(falls) > 0 and len(rises) > 0:
                        idx_start = falls[0]
                        idx_stop = rises[rises > idx_start][0]
                        
                        # TTS Regression Logic
                        # Extract the active ramp segment
                        ramp_v = v_px[idx_start:idx_stop]
                        ramp_t = time_vec[idx_start:idx_stop]
                        
                        # Filter for linear region (ignore noise and floor)
                        mask = (ramp_v > 1.0) & (ramp_v < 4.5)
                        
                        if np.sum(mask) > 5:
                            slope, _ = np.polyfit(ramp_t[mask], ramp_v[mask], 1)
                            inferred_v = abs(slope) * t
                        else:
                            # If integration is extremely fast, use first/last detected points
                            inferred_v = (np.mean(v_px[idx_start-5:idx_start]) - v_px[idx_stop])
                            if inferred_v < 0: inferred_v = 0
                        
                        master_data.append({
                            "t_int_ms": t * 1000,
                            "i_led_ma": curr,
                            "delta_v_inferred": inferred_v
                        })
                except:
                    continue

        # 3. Final Report
        final_df = pl.DataFrame(master_data)
        final_df.write_csv("pixel_2d_wdr_results.csv")
        
        plt.figure(figsize=(12, 8))
        for t_ms in final_df["t_int_ms"].unique():
            subset = final_df.filter(pl.col("t_int_ms") == t_ms)
            plt.plot(subset["i_led_ma"], subset["delta_v_inferred"], 'o-', label=f"t_int = {t_ms:.2f}ms")
        
        plt.yscale('log') # Log scale for Y to handle the thousands of volts
        plt.xlabel("LED Current (mA)")
        plt.ylabel("Inferred Theoretical Pixel Drop (V) [Log Scale]")
        plt.title("2D Wide Dynamic Range (WDR) Linearity\n(Slope-Based Theoretical Reconstruction)")
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        plt.savefig("pixel_2d_wdr_plot.png")
        print("\n--- 2D WDR Sweep Complete. Plot: pixel_2d_wdr_plot.png ---")

if __name__ == "__main__":
    run_2d_wdr()
