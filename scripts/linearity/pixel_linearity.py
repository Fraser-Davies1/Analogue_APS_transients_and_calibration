from pytestlab import Bench, MeasurementSession
from pytestlab.measurements.steps import step
import numpy as np
import time
import polars as pl
import sys

def run_experiment():
    print("--- Opening Bench ---")
    with Bench.open("bench.yaml") as bench:
        print("--- Instrument Discovery ---")
        # Identify instruments
        for name, inst in bench._instrument_instances.items():
            try:
                print(f"  [OK] {name}: {inst.id()}")
            except Exception as e:
                print(f"  [WARN] {name} identification failed: {e}")

        # 1. Setup PSU (with manual override if reset fails)
        print("--- Configuring Power ---")
        try:
            # Try to turn on VDD and LED Bias
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.05).on() 
            bench.psu.channel(2).set(voltage=3.0, current_limit=0.02).on()
            print("  [OK] PSU Channels 1 & 2 enabled")
        except Exception as e:
            print(f"  [CRITICAL] Could not power up circuit: {e}")
            # If PSU is completely dead, we cannot proceed
            return

        # 2. Setup SigGen
        try:
            bench.siggen.channel(1).setup_square(frequency=1000, amplitude=3.3).enable()
            print("  [OK] SigGen Reset Pulse enabled")
        except Exception as e:
            print(f"  [WARN] SigGen setup failed: {e}")

        # 3. Setup Oscilloscope
        try:
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
            bench.osc.channel(3).setup(scale=1.0, offset=0.0).enable()
            bench.osc.trigger.setup_edge(source="CH3", level=1.5)
            print("  [OK] OSC Channels 1 & 3 configured")
        except Exception as e:
            print(f"  [WARN] OSC setup failed: {e}")

        print("--- Starting Sweep ---")
        with MeasurementSession(bench=bench, name="Pixel_Linearity") as session:
            # Sweeping integration time (10ms down to 0.2ms)
            session.parameter("freq", step.log(100, 5000, 15), unit="Hz")
            
            @session.acquire
            def measure(freq, siggen, osc):
                try:
                    siggen.set_frequency(1, freq)
                except: pass
                
                time.sleep(0.5)
                
                # Measure drop using Peak-to-Peak
                res = osc.measure_voltage_peak_to_peak(1)
                val = res.values
                v_drop = val.nominal_value if hasattr(val, "nominal_value") else val
                
                return {
                    "integration_time_ms": (1.0 / freq) * 1000,
                    "v_drop": v_drop
                }
            
            experiment = session.run(show_progress=True)
            
        # 4. Save results
        filename = "linearity_results.csv"
        experiment.data.write_csv(filename)
        print(f"\n--- Success! Results saved to {filename} ---")

if __name__ == "__main__":
    run_experiment()
