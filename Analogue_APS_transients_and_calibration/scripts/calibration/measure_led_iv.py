import numpy as np
import polars as pl
from pytestlab import Bench, MeasurementSession
from pytestlab.measurements.steps import step
import time
import matplotlib.pyplot as plt

def run_led_iv_sweep():
    print("--- Starting LED I-V Characterization ---")
    
    with Bench.open("bench.yaml") as bench:
        try:
            # 1. Hardware Initialization
            # CH1: VDD @ 5V
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            
            # Setup OSC Channel 2 for DC measurement
            # Scale 0.1V/div provides good resolution for the resistor drop
            bench.osc.channel(2).setup(scale=0.1, offset=0.0, coupling="DC").enable()
            print("  [OK] Hardware ready.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")
            return

        with MeasurementSession(bench=bench, name="LED_IV_Sweep") as session:
            # Sweep PSU CH2 from 0.0V to 5.0V in 0.1V steps
            session.parameter("v_in_psu", step.linear(0.0, 5.0, 51), unit="V")
            
            @session.acquire
            def measure_current(v_in_psu, psu, osc):
                # Set PSU voltage
                try:
                    psu.channel(2).set(voltage=v_in_psu).on()
                    time.sleep(0.4) # Allow settling
                except: pass
                
                # Measure voltage drop across 220 ohm resistor on Scope CH2
                # We use RMS as the DC average proxy
                res = osc.measure_rms_voltage(2)
                v_res = res.values.nominal_value if hasattr(res.values, "nominal_value") else res.values
                
                # Calculate Current: I = V/R
                i_led_ma = (v_res / 220.0) * 1000.0
                
                return {
                    "v_in_psu": v_in_psu,
                    "v_resistor": v_res,
                    "i_led_ma": i_led_ma
                }
            
            experiment = session.run(show_progress=True)
            
        # 2. Save and Report
        filename = "led_iv_results.csv"
        experiment.data.write_csv(filename)
        
        # Plotting
        df = experiment.data
        x = df["v_in_psu"].to_numpy()
        y = df["i_led_ma"].to_numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'bo-', linewidth=1, markersize=4, label='Measured I-V')
        
        # Identify threshold (knee)
        # Find first index where current > 0.1mA
        active_indices = np.where(y > 0.1)[0]
        if len(active_indices) > 0:
            v_threshold = x[active_indices[0]]
            plt.axvline(v_threshold, color='r', linestyle='--', label=f'Threshold ~{v_threshold:.1f}V')
        
        plt.xlabel("Input Voltage (V)")
        plt.ylabel("LED Current (mA)")
        plt.title("LED I-V Characteristic (Measured across 220 Ohm Resistor)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("led_iv_plot.png")
        
        print(f"\n--- Sweep Complete ---")
        print(f"Results saved to {filename}")
        print("Plot saved to led_iv_plot.png")

if __name__ == "__main__":
    run_led_iv_sweep()
