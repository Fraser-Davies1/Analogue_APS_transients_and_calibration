import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def characterisation():
    print("--- Starting LED Current Characterisation ---")
    
    with Bench.open("bench.yaml") as bench:
        # 1. Setup Static Rails
        # We set a wide current limit (0.1A) once and then don't touch it.
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.psu.channel(2).set(voltage=0.0, current_limit=0.1).on()
        
        # 2. Setup Scope
        # Ensure we are in a timebase where the average is stable (e.g. 2ms/div)
        bench.osc.channel(2).setup(scale=0.2, offset=0.0, coupling="DC").enable()
        bench.osc.set_time_axis(scale=2.0e-3, position=0.0)
        time.sleep(1)

        v_in_sweep = np.arange(0.0, 5.1, 0.1)
        results = []

        print(f"--- Sweeping V_in (PSU CH2) and measuring drop on Scope CH2 ---")
        for v in v_in_sweep:
            # Set Voltage
            bench.psu.channel(2).set(voltage=v)
            time.sleep(0.5) # Settlement
            
            # Query Hardware Average DC Value from Scope
            # We use a raw query for the specific internal function
            try:
                v_drop_str = bench.osc._query(":MEASure:VAVerage? CHANnel2")
                v_drop = float(v_drop_str)
                # Filter out instrument error constants (e.g. 9.9e37)
                if v_drop > 10.0: v_drop = 0.0
            except:
                v_drop = 0.0
            
            # Calculate Current
            i_ma = (v_drop / 220.0) * 1000.0
            
            results.append({
                "v_in_psu": v,
                "v_drop_scope": v_drop,
                "i_led_ma": i_ma
            })
            
            print(f"  V_in: {v:.1f}V | V_drop: {v_drop:.3f}V | I: {i_ma:.3f}mA", end="\r")

        # 3. Data Processing
        df = pl.DataFrame(results)
        df.write_csv("led_current_results.csv")
        
        # 4. Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(df["v_in_psu"], df["i_led_ma"], 'bo-', label='I_LED (Measured via Resistor)')
        plt.xlabel("PSU Input Voltage (V)")
        plt.ylabel("LED Current (mA)")
        plt.title("LED I-V Curve: Current derived from Scope Average DC Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("led_current_plot.png")
        
        print(f"\n--- Characterisation Complete ---")
        print("Data: led_current_results.csv")
        print("Plot: led_current_plot.png")

if __name__ == "__main__":
    characterisation()
