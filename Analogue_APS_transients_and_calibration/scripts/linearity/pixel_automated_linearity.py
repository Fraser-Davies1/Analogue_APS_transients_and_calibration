import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_automated_test():
    print("--- Initializing Fully Automated Bench ---")
    
    with Bench.open("bench.yaml") as bench:
        # 1. Setting Rails and Logic
        # VDD = 5V, Reset = 5Vpp @ 2.5V Offset (0-5V) @ 500Hz
        print("--- Configuring Fixed Parameters (VDD=5V, Reset=5V, 500Hz) ---")
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            # OSC: 1ms/div provides 10ms (enough for multiple 2ms cycles)
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
            bench.osc.set_time_axis(scale=1e-3, position=5e-3)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Timing and Power Rails Locked.")
        except Exception as e:
            print(f"  [ERROR] Hardware initialization failed: {e}")
            return

        results = []
        v_led_sweep = np.linspace(2.3, 4.5, 20)
        
        print(f"--- Starting Automated Sweep ({len(v_led_sweep)} steps) ---")
        for v in v_led_sweep:
            print(f"  Advancing V_LED to {v:.2f}V...", end="\r")
            
            # Step 1: Set LED Voltage
            bench.psu.channel(2).set(voltage=v, current_limit=0.05).on()
            time.sleep(0.5) # Settle time
            
            # Step 2: Capture Waveforms
            data = bench.osc.read_channels([1, 3])
            df = data.values
            
            ch1 = df["Channel 1 (V)"].to_numpy() # Pixel Output
            ch3 = df["Channel 3 (V)"].to_numpy() # Reset Waveform
            
            # Step 3: Precision Edge Analysis
            # Falling edge (diff -1) = Start of integration
            # Rising edge (diff 1) = End of integration
            edges = np.diff((ch3 > 2.5).astype(int))
            fall_pts = np.where(edges == -1)[0]
            rise_pts = np.where(edges == 1)[0]
            
            if len(fall_pts) > 0:
                t_start = fall_pts[0]
                # Find the rising edge that immediately follows this falling edge
                stops = rise_pts[rise_pts > t_start]
                if len(stops) > 0:
                    t_stop = stops[0]
                    
                    # Measurement "just before" edges (average 5 samples)
                    v_start = np.mean(ch1[max(0, t_start-10):t_start])
                    v_end = np.mean(ch1[max(0, t_stop-10):t_stop])
                    delta_v = v_start - v_end
                    
                    results.append({
                        "v_led_in": v,
                        "v_pixel_start": v_start,
                        "v_pixel_end": v_end,
                        "delta_v_out": delta_v
                    })

        # 4. Final Processing and Plotting
        if not results:
            print("\n[ERROR] Test failed: No integration windows were detected.")
            return

        res_df = pl.DataFrame(results)
        res_df.write_csv("automated_pixel_report.csv")
        
        x = res_df["v_led_in"].to_numpy()
        y = res_df["delta_v_out"].to_numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'bo-', label='Pixel Response (V_start - V_end)')
        
        # Fit Line
        m, b = np.polyfit(x, y, 1)
        r_sq = np.corrcoef(x, y)[0, 1]**2
        plt.plot(x, m*x + b, 'r--', alpha=0.7, label=f'Linear Fit (R²={r_sq:.4f})')
        
        plt.xlabel("LED Input Voltage (V)")
        plt.ylabel("Pixel Integration Drop (V)")
        plt.title(f"Automated Pixel Linearity: VDD=5V, Reset=5V, 500Hz\nR² = {r_sq:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("automated_linearity_plot.png")
        
        print(f"\n--- Test Complete ---")
        print(f"R-squared: {r_sq:.4f}")
        print("Data saved to automated_pixel_report.csv")
        print("Plot saved to automated_linearity_plot.png")

if __name__ == "__main__":
    run_automated_test()
