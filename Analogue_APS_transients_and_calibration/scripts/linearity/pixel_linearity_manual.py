from pytestlab import Bench, MeasurementSession
from pytestlab.measurements.steps import step
import numpy as np
import time
import polars as pl

def run_manual_psu_experiment():
    print("--- Opening Bench (PSU Bypass Mode) ---")
    with Bench.open("bench.yaml") as bench:
        # 1. Setup Functional Instruments
        print("--- Configuring SigGen & OSC ---")
        try:
            bench.siggen.channel(1).setup_square(frequency=1000, amplitude=3.3).enable()
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
            bench.osc.channel(3).setup(scale=1.0, offset=0.0).enable()
            bench.osc.trigger.setup_edge(source="CH3", level=1.5)
            print("  [OK] Hardware ready.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")
            return

        print("--- Starting Sweep ---")
        with MeasurementSession(bench=bench, name="Pixel_Linearity_Manual") as session:
            # Sweeping integration time (10ms down to 0.2ms)
            session.parameter("freq", step.log(100, 5000, 20), unit="Hz")
            
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
            
        # 2. Export and Summary
        filename = "linearity_results_manual.csv"
        experiment.data.write_csv(filename)
        print(f"\n--- Success! Results saved to {filename} ---")
        
        # Linearity Check
        df = experiment.data
        if not df.is_empty():
            x, y = df["integration_time_ms"].to_numpy(), df["v_drop"].to_numpy()
            # Filter out error values (9.9e37)
            mask = y < 1e30
            if mask.any():
                r_sq = np.corrcoef(x[mask], y[mask])[0, 1]**2
                print(f"Calculated Linearity R²: {r_sq:.5f}")

if __name__ == "__main__":
    run_manual_psu_experiment()
