import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pytestlab import Bench
import time

def run_transient_experiment():
    print("--- Starting Linearized Transient Experiment ---")
    
    # 1. Load the Calibration LUT
    try:
        lut = pl.read_csv("led_lin_lut.csv")
        # Select 6 levels for clear plotting: 0, 20, 40, 60, 80, 100%
        target_levels = [0, 20, 40, 60, 80, 100]
        test_points = lut.filter(pl.col("light_percent").is_in(target_levels))
    except Exception as e:
        print(f"Error loading LUT: {e}")
        return

    with Bench.open("bench.yaml") as bench:
        print("--- Configuring 5V Rails and 500Hz Reset ---")
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            # Setup Scope for 2+ full cycles
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
            bench.osc.set_time_axis(scale=1e-3, position=5e-3) # 10ms window
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Hardware ready.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")
            return

        plt.figure(figsize=(12, 7))

        for row in test_points.iter_rows(named=True):
            pct = int(row['light_percent'])
            v_led = row['v_led_control']
            
            print(f"  Capturing transient for {pct}% Light ({v_led:.4f}V)...")
            
            # Set non-linear voltage to get linear light
            try:
                bench.psu.channel(2).set(voltage=v_led).on()
                time.sleep(0.8) # Ensure settling
            except:
                pass
            
            # Capture waveform
            reading = bench.osc.read_channels([1, 3])
            df = reading.values
            
            # Save for later analysis
            df.write_csv(f"transient_{pct}pct.csv")
            
            # Plotting the pixel output (CH1)
            time_ms = df["Time (s)"].to_numpy() * 1000
            pixel_v = df["Channel 1 (V)"].to_numpy()
            
            plt.plot(time_ms, pixel_v, label=f"Intensity {pct}% ({v_led:.3f}V)")

        # 3. Formatting Plot
        plt.xlabel("Time (ms)")
        plt.ylabel("Pixel Output (V)")
        plt.title("Pixel Transient Response: Linear Light Intensity Sweep\n(VDD=5V, Reset=5V, 500Hz)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("pixel_transients_overlay.png")
        
        print("\n--- Experiment Complete ---")
        print("Waveforms saved to transient_*.csv")
        print("Overlay plot saved to pixel_transients_overlay.png")

if __name__ == "__main__":
    run_transient_experiment()
