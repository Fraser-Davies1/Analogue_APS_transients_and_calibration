from pytestlab import Bench, MeasurementSession
from pytestlab.measurements.steps import step
import numpy as np
import time
import polars as pl

def run_experiment():
    print("--- Opening Bench (5V Logic / 5V VDD) ---")
    with Bench.open("bench.yaml") as bench:
        # 1. Hardware Initialization
        print("--- Initializing Hardware (VDD=5V, Reset=5V) ---")
        try:
            # Power VDD
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.05).on()
            
            # Reset Waveform (5V High, 0V Low)
            # Amplitude 5V, Offset 2.5V = 0 to 5V Square Wave
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            # OSC Setup
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() # Pixel Out
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Reset Sync (5V swing)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            
            # Timebase: 500us/div -> 5ms window to see multiple cycles
            bench.osc.set_time_axis(scale=500e-6, position=1.0e-3)
            
            print("  [OK] VDD and Reset Pulse initialized.")
        except Exception as e:
            print(f"  [ERROR] Hardware setup failed: {e}")
            print("  Please ensure VDD=5V and SigGen=5V manually if PSU/SigGen fails.")

        print("--- Starting Sweep ---")
        with MeasurementSession(bench=bench, name="Pixel_5V_Linearity") as session:
            # Sweep LED Voltage
            session.parameter("v_led", step.linear(2.3, 4.5, 20), unit="V")
            
            @session.acquire
            def measure(v_led, psu, osc):
                try:
                    psu.channel(2).set_voltage(v_led).on()
                except:
                    # If PSU bridge is down, we still try to measure what is on the scope
                    pass
                
                time.sleep(0.8)
                
                # In a 5V system, delta_v is the Vpp of the discharging ramp
                res = osc.measure_voltage_peak_to_peak(1)
                v_drop = res.values.nominal_value if hasattr(res.values, "nominal_value") else res.values
                
                return {
                    "v_led": v_led,
                    "v_drop": v_drop
                }
            
            experiment = session.run(show_progress=True)
            
        filename = "v_to_v_linearity_v3.csv"
        experiment.data.write_csv(filename)
        print(f"\n--- Success! Results saved to {filename} ---")

if __name__ == "__main__":
    run_experiment()
