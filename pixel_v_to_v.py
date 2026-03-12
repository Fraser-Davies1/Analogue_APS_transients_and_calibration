from pytestlab import Bench, MeasurementSession
from pytestlab.measurements.steps import step
import numpy as np
import time
import polars as pl

def run_v_to_v_linearity():
    print("--- Opening Bench (Fixed 500Hz Integration) ---")
    with Bench.open("bench.yaml") as bench:
        # 1. Fixed Configuration
        print("--- Setting Fixed Reset Pulse (500Hz) ---")
        try:
            # 500Hz = 2ms period
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=3.3).enable()
            
            # Setup OSC for the 2ms window
            bench.osc.set_time_axis(scale=1.0e-3, position=2.0e-3) # 1ms/div
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
            bench.osc.channel(3).setup(scale=1.0, offset=0.0).enable()
            bench.osc.trigger.setup_edge(source="CH3", level=1.5)
            print("  [OK] Reset timing locked.")
        except Exception as e:
            print(f"  [ERROR] Timing setup failed: {e}")
            return

        print("--- Starting LED Voltage Sweep ---")
        with MeasurementSession(bench=bench, name="Pixel_V_to_V") as session:
            # Sweep LED Voltage from 2.3V to 4.0V
            session.parameter("v_led", step.linear(2.3, 4.0, 20), unit="V")
            
            @session.acquire
            def measure_v_to_v(v_led, psu, osc):
                # If PSU is manual, this will print a warning but the test will wait
                try:
                    psu.channel(2).set_voltage(v_led)
                except:
                    print(f"  [MANUAL] Please set PSU LED to {v_led}V and press Enter")
                
                time.sleep(0.8) # Allow LED and Pixel to settle
                
                # High-Precision Timing Measurement
                # We measure the peak-to-peak during the integration window
                # Alternatively, use measure_voltage_max (reset level) and min (final level)
                v_reset = osc.measure_voltage_max(1)
                v_final = osc.measure_voltage_min(1)
                delta_v = v_reset - v_final
                
                return {
                    "v_led": v_led,
                    "v_pixel_reset": v_reset,
                    "v_pixel_final": v_final,
                    "delta_v": delta_v
                }
            
            experiment = session.run(show_progress=True)
            
        # 3. Export
        filename = "v_to_v_linearity.csv"
        experiment.data.write_csv(filename)
        print(f"\n--- Results saved to {filename} ---")

if __name__ == "__main__":
    run_v_to_v_linearity()
