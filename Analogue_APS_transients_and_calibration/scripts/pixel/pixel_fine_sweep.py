  import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_fine_sweep():
    # Define the multi-resolution voltage points
    v_points = np.concatenate([
        np.arange(2.0, 2.3, 0.1),       # Pre-threshold (Coarse)
        np.arange(2.3, 2.601, 0.001),   # Active/Turn-on (Ultra-Fine)
        np.arange(2.7, 3.5, 0.1)        # Saturation (Coarse)
    ])

    print(f"--- Starting Multi-Resolution Sweep ({len(v_points)} points) ---")
    
    with Bench.open("bench.yaml") as bench:
        try:
            # 1. Hardware Initialization
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            # OSC: 1ms/div -> 10ms window
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
            bench.osc.set_time_axis(scale=1e-3, position=5e-3)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
        except Exception as e:
            print(f"Setup failed: {e}")
            return

        results = []
        
        for i, v in enumerate(v_points):
            # Log progress every 20 points to avoid spamming the console
            if i % 20 == 0:
                print(f"  Step {i}/{len(v_points)}: Setting V_LED to {v:.3f}V...")
            
            try:
                bench.psu.channel(2).set(voltage=v, current_limit=0.05).on()
                # Use a slightly shorter settle for the fine sweep to keep total time reasonable
                time.sleep(0.3) 
            except:
                continue
            
            # Capture and Analyze
            data = bench.osc.read_channels([1, 3])
            df = data.values
            ch1, ch3 = df["Channel 1 (V)"].to_numpy(), df["Channel 3 (V)"].to_numpy()
            
            edges = np.diff((ch3 > 2.5).astype(int))
            fall_pts, rise_pts = np.where(edges == -1)[0], np.where(edges == 1)[0]
            
            if len(fall_pts) > 0:
                t_start = fall_pts[0]
                stops = rise_pts[rise_pts > t_start]
                if len(stops) > 0:
                    t_stop = stops[0]
                    
                    # Extract values just before edges
                    v_start = np.mean(ch1[max(0, t_start-10):t_start])
                    v_end = np.mean(ch1[max(0, t_stop-10):t_stop])
                    
                    results.append({
                        "v_led_in": v,
                        "v_pixel_drop": v_start - v_end
                    })

        # Save and Analyze
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("fine_linearity_results.csv")
            
            # Calculate R-squared for the fine region (2.3V - 2.6V)
            fine_zone = res_df.filter((pl.col("v_led_in") >= 2.3) & (pl.col("v_led_in") <= 2.6))
            if len(fine_zone) > 1:
                x, y = fine_zone["v_led_in"].to_numpy(), fine_zone["v_pixel_drop"].to_numpy()
                r_sq = np.corrcoef(x, y)[0, 1]**2
                print(f"\n--- Fine Sweep Complete ---")
                print(f"Active Region Linearity (2.3V-2.6V) R²: {r_sq:.6f}")
            
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["v_led_in"], res_df["v_pixel_drop"], 'b.', markersize=2)
            plt.axvspan(2.3, 2.6, color='green', alpha=0.1, label='Fine Region')
            plt.xlabel("LED Input Voltage (V)")
            plt.ylabel("Pixel Integration Drop (V)")
            plt.title("Multi-Resolution Pixel Linearity Analysis")
            plt.grid(True, alpha=0.3)
            plt.savefig("fine_linearity_plot.png")
            print("Results saved to fine_linearity_results.csv and fine_linearity_plot.png")

if __name__ == "__main__":
    run_fine_sweep()
