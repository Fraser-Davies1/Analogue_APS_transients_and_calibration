import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pytestlab import Bench
import time

def run_refined_transient():
    print("--- Starting Refined Linearized Transients ---")
    
    # 1. Load the Calibration LUT and select specific levels
    try:
        lut = pl.read_csv("led_lin_lut.csv")
        # Target levels: 10, 20, 30, 50, 70, 90%
        target_levels = [10, 20, 30, 50, 70, 90]
        test_points = lut.filter(pl.col("light_percent").is_in(target_levels))
    except Exception as e:
        print(f"Error loading LUT: {e}")
        return

    with Bench.open("bench.yaml") as bench:
        print("--- Configuring Rails (5V) and Timing (500Hz) ---")
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            # Setup Scope for exactly 2 cycles (4ms)
            # 500us/div * 10 divisions = 5ms total window
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
            bench.osc.set_time_axis(scale=500e-6, position=2.5e-3) 
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Scope gated for 2 cycles.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")
            return

        plt.figure(figsize=(10, 6))

        for row in test_points.iter_rows(named=True):
            pct = int(row['light_percent'])
            v_led = row['v_led_control']
            
            print(f"  Capturing {pct}% Light ({v_led:.4f}V)...")
            
            # Set compensated voltage
            try:
                bench.psu.channel(2).set(voltage=v_led).on()
                time.sleep(0.8) # Thermal/Pixel stabilization
            except:
                pass
            
            # Capture waveform
            reading = bench.osc.read_channels([1, 3])
            df = reading.values
            
            # Save raw data
            df.write_csv(f"refined_transient_{pct}pct.csv")
            
            # Plot pixel ramp (CH1)
            time_ms = df["Time (s)"].to_numpy() * 1000
            pixel_v = df["Channel 1 (V)"].to_numpy()
            plt.plot(time_ms, pixel_v, label=f"Intensity {pct}%")

        # Formatting
        plt.xlabel("Time (ms)")
        plt.ylabel("Pixel Output (V)")
        plt.title("Refined Pixel Transients (2 Cycles)\nLinearized Intensity Sweep (10% to 90%)")
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("pixel_transients_refined.png")
        
        print("\n--- Refined Experiment Complete ---")
        print("Plots saved to pixel_transients_refined.png")

if __name__ == "__main__":
    run_refined_transient()
