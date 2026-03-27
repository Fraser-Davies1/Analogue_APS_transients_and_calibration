from pytestlab import Bench, MeasurementSession
from pytestlab.measurements.steps import step
import numpy as np
import time
import polars as pl

def run_v_to_v_linearity():
    print("--- Opening Bench (V-to-V Sweep) ---")
    with Bench.open("bench.yaml") as bench:
        print("--- Setting Fixed Reset Pulse (500Hz) ---")
        try:
            # 500Hz = 2ms cycle
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=3.3).enable()
            
            # GATING: Set scope to see roughly 1.5ms of the 2ms integration
            # Scale 200us/div * 10 div = 2ms window
            bench.osc.set_time_axis(scale=200e-6, position=1.0e-3) 
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
            bench.osc.channel(3).setup(scale=1.0, offset=0.0).enable()
            bench.osc.trigger.setup_edge(source="CH3", level=1.5)
            print("  [OK] Reset timing locked and Gated.")
        except Exception as e:
            print(f"  [ERROR] Timing setup failed: {e}")
            return

        print("--- Starting LED Voltage Sweep ---")
        with MeasurementSession(bench=bench, name="Pixel_V_to_V_v2") as session:
            # Sweep LED Voltage from 2.3V to 4.5V
            session.parameter("v_led", step.linear(2.3, 4.5, 20), unit="V")
            
            @session.acquire
            def measure_v_to_v(v_led, psu, osc):
                try:
                    psu.channel(2).set_voltage(v_led)
                except: pass # Bypass PSU 500 errors
                
                time.sleep(1.0) # Settle
                
                # In the gated window, Vpp is identical to V_reset - V_final
                res = osc.measure_voltage_peak_to_peak(1)
                v_drop = res.values.nominal_value if hasattr(res.values, "nominal_value") else res.values
                
                return {
                    "v_led": v_led,
                    "v_pixel_drop": v_drop
                }
            
            experiment = session.run(show_progress=True)
            
        # Export
        filename = "v_to_v_linearity_v2.csv"
        experiment.data.write_csv(filename)
        print(f"\n--- Success! Results saved to {filename} ---")

if __name__ == "__main__":
    run_v_to_v_linearity()
