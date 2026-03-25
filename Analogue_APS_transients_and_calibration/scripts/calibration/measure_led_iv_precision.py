import numpy as np
import polars as pl
from pytestlab import Bench, MeasurementSession
from pytestlab.measurements.steps import step
import time
import matplotlib.pyplot as plt

def run_led_iv_precision():
    print("--- Starting Precision LED I-V Characterization ---")
    print("--- Using Scope Average DC Value across 220 Ohm Resistor ---")
    
    with Bench.open("bench.yaml") as bench:
        try:
            # 1. Hardware Initialization
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on() # VDD
            
            # Setup OSC Channel 2
            # We want to measure the DC voltage drop across the resistor
            bench.osc.channel(2).setup(scale=0.1, offset=0.0, coupling="DC").enable()
            # Set timebase to something reasonable to get a good average (e.g. 1ms/div)
            bench.osc.set_time_axis(scale=1.0e-3, position=0.0)
            print("  [OK] Hardware initialized.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")
            return

        with MeasurementSession(bench=bench, name="LED_IV_Precision_Sweep") as session:
            # Sweep PSU CH2 from 0.0V to 5.0V in 0.1V steps
            session.parameter("v_in_psu", step.linear(0.0, 5.0, 51), unit="V")
            
            @session.acquire
            def measure_led_current(v_in_psu, psu, osc):
                # Set PSU voltage
                try:
                    psu.channel(2).set_voltage(v_in_psu).on()
                    time.sleep(0.4) # Allow settling
                except: pass
                
                # Measure Average DC Voltage on Channel 2
                # Since PyTestLab doesn't have a direct measure_average method,
                # we capture the trace and calculate the mean for maximum precision.
                reading = osc.read_channels(2)
                v_avg = reading.values["Channel 2 (V)"].mean()
                
                # Calculate Current: I = V/R (R=220)
                i_led_ma = (v_avg / 220.0) * 1000.0
                
                return {
                    "v_in_psu": v_in_psu,
                    "v_resistor_avg": v_avg,
                    "i_led_ma": i_led_ma
                }
            
            experiment = session.run(show_progress=True)
            
        # 2. Save and Report
        filename = "led_iv_precision_results.csv"
        experiment.data.write_csv(filename)
        
        # Plotting
        df = experiment.data
        x = df["v_in_psu"].to_numpy()
        y = df["i_led_ma"].to_numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'rs-', linewidth=1, markersize=4, label='Measured I-V (Scope Avg)')
        
        # Identify threshold (knee)
        active_mask = y > 0.05 # Threshold for "ON" (50uA)
        if active_mask.any():
            v_th = x[np.where(active_mask)[0][0]]
            plt.axvline(v_th, color='k', linestyle='--', label=f'Turn-on ~{v_th:.1f}V')
        
        plt.xlabel("Control Voltage V_in (V)")
        plt.ylabel("LED Current I_LED (mA)")
        plt.title("LED I-V Characteristic (Calculated from Scope DC Average)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("led_iv_precision_plot.png")
        
        print(f"\n--- Sweep Complete ---")
        print(f"Results saved to {filename}")
        print("Plot saved to led_iv_precision_plot.png")
        
        # Linearity check of the Ohmic region
        ohmic_df = df.filter(pl.col("v_in_psu") > 2.6)
        if not ohmic_df.is_empty():
            x_o = ohmic_df["v_in_psu"].to_numpy()
            y_o = ohmic_df["i_led_ma"].to_numpy()
            r_sq = np.corrcoef(x_o, y_o)[0, 1]**2
            print(f"Ohmic Region (>2.6V) Linearity R²: {r_sq:.5f}")

if __name__ == "__main__":
    run_led_iv_precision()
