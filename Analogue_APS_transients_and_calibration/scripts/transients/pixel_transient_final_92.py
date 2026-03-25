import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pytestlab import Bench
import time

def run_final_transient():
    print("--- Starting Final Linearized Transients (Including 92% and Ref Waveform) ---")
    
    # 1. Load the LUT and filter for target points
    lut = pl.read_csv("led_lin_lut.csv")
    test_points = lut.filter(pl.col("light_percent").is_in([10, 20, 30, 50, 70, 90]))
    
    # Construct final test list: (percent, voltage)
    final_v_list = []
    for row in test_points.iter_rows(named=True):
        final_v_list.append((row['light_percent'], row['v_led_control']))
    
    # Add the custom 92% point
    final_v_list.append((92, 2.4380)) 
    
    # Sort by percent for consistent plotting/legend
    final_v_list.sort(key=lambda x: x[0])

    with Bench.open("bench.yaml") as bench:
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
            bench.osc.set_time_axis(scale=500e-6, position=2.5e-3) 
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Hardware timing and rails set.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")
            return

        plt.figure(figsize=(12, 7))
        ref_plotted = False

        for pct, v_led in final_v_list:
            print(f"  Capturing {pct}% Light ({v_led:.4f}V)...")
            try:
                bench.psu.channel(2).set(voltage=v_led).on()
                time.sleep(0.8) # Stabilization
            except: 
                pass
            
            # Capture waveforms
            data = bench.osc.read_channels([1, 3])
            df = data.values
            time_ms = df["Time (s)"].to_numpy() * 1000
            pixel_v = df["Channel 1 (V)"].to_numpy()
            reset_v = df["Channel 3 (V)"].to_numpy()
            
            # Plot Pixel Transients
            plt.plot(time_ms, pixel_v, label=f"Pixel Output {pct}%")
            
            # Plot Reference Waveform (CH3) once
            if not ref_plotted:
                plt.plot(time_ms, reset_v, 'k--', alpha=0.4, label="Reference Reset (CH3)")
                ref_plotted = True

        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (V)")
        plt.title("Final Pixel Transients (2 Cycles)\nLinearized Intensity Sweep (10% to 92%) with Reset Reference")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("pixel_final_report_plot_92.png")
        print("\n--- Final Experiment Complete ---")
        print("Final plot saved to pixel_final_report_plot_92.png")

if __name__ == "__main__":
    run_final_transient()
