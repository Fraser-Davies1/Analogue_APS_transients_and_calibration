import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_sweep():
    print("--- Opening Bench ---")
    with Bench.open("bench.yaml") as bench:
        # 1. Setup Rails
        print("--- Setting VDD=5V, Reset=5V (0-5V) @ 500Hz ---")
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            # Setup Scope: 1ms/div provides 10ms (5 full cycles of 500Hz)
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() # Pixel Out
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Reset Waveform
            bench.osc.set_time_axis(scale=1e-3, position=5e-3)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Hardware timing and rails set.")
        except Exception as e:
            print(f"  [WARN] Hardware setup warning: {e}")

        results = []
        # Sweeping LED voltage from 2.3V to 4.5V
        v_led_steps = np.linspace(2.3, 4.5, 20)
        
        print(f"--- Starting Sweep of PSU Channel 2 ({len(v_led_steps)} steps) ---")
        for v in v_led_steps:
            print(f"  Measuring V_LED = {v:.2f}V...", end="\r")
            try:
                bench.psu.channel(2).set(voltage=v, current_limit=0.05).on()
                time.sleep(0.5) # Allow thermal/pixel settling
            except Exception as e:
                # If PSU bridge continues to fail, we stop to avoid garbage data
                print(f"\n[ERROR] PSU Channel 2 Communication Failed: {e}")
                break
            
            # Capture full waveforms for precision analysis
            reading = bench.osc.read_channels([1, 3])
            df = reading.values
            
            v_pixel = df["Channel 1 (V)"].to_numpy()
            v_reset = df["Channel 3 (V)"].to_numpy()
            
            # Identify Integration Window
            is_high = v_reset > 2.5
            edges = np.diff(is_high.astype(int))
            falling_indices = np.where(edges == -1)[0]
            rising_indices = np.where(edges == 1)[0]
            
            if len(falling_indices) > 0 and len(rising_indices) > 0:
                idx_start = falling_indices[0] # First falling edge
                # Find the rising edge that ends this specific integration
                stops = rising_indices[rising_indices > idx_start]
                if len(stops) > 0:
                    idx_stop = stops[0]
                    
                    # Capture "just before" (average of 5 samples for noise reduction)
                    v_initial = np.mean(v_pixel[max(0, idx_start-5):idx_start])
                    v_final = np.mean(v_pixel[max(0, idx_stop-5):idx_stop])
                    
                    results.append({
                        "v_led": v,
                        "v_start": v_initial,
                        "v_end": v_final,
                        "delta_v": v_initial - v_final
                    })
        
        if not results:
            print("\n[ERROR] No valid measurement data captured.")
            return

        # 2. Data Processing and Plotting
        res_df = pl.DataFrame(results)
        res_df.write_csv("pixel_linear_sweep.csv")
        
        x = res_df["v_led"].to_numpy()
        y = res_df["delta_v"].to_numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'rs-', linewidth=2, label='Pixel Output (ΔV)')
        
        # Fit a line to check linearity
        m, b = np.polyfit(x, y, 1)
        r_sq = np.corrcoef(x, y)[0, 1]**2
        plt.plot(x, m*x + b, 'k--', alpha=0.5, label=f'Linear Fit (R²={r_sq:.4f})')
        
        plt.xlabel("LED Control Voltage (V)")
        plt.ylabel("Effective Output Voltage Drop (V)")
        plt.title(f"Pixel Linearity Sweep (Fixed 500Hz/5V Reset)\nR² = {r_sq:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("pixel_linearity_report.png")
        
        print(f"\n--- Sweep Complete ---")
        print(f"R-squared: {r_sq:.4f}")
        print("Report saved to pixel_linearity_report.png")

if __name__ == "__main__":
    run_sweep()
