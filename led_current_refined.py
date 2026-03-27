import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def characterisation_refined():
    print("--- Starting Refined LED Current Characterisation ---")
    
    with Bench.open("bench.yaml") as bench:
        # 1. Setup Rails
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.psu.channel(2).set(voltage=0.0, current_limit=0.1).on()
        
        # 2. Setup Scope - INCREASED SENSITIVITY
        # 50mV/div allows us to see as low as 1mV drop (~4uA)
        bench.osc.channel(2).setup(scale=0.05, offset=0.0, coupling="DC").enable()
        bench.osc.set_time_axis(scale=5.0e-3, position=0.0) # Longer time for better averaging
        time.sleep(1)

        v_in_sweep = np.arange(0.0, 5.1, 0.1)
        results = []

        print(f"--- Sweeping V_in and measuring DC Average on Scope CH2 ---")
        for v in v_in_sweep:
            bench.psu.channel(2).set(voltage=v)
            time.sleep(0.6) 
            
            # Query Hardware Average DC Value
            try:
                # We use the internal measurement function if available
                v_drop_str = bench.osc._query(":MEASure:VAVerage? CHANnel2")
                v_drop = float(v_drop_str)
                if v_drop > 10.0: v_drop = 0.0 # Error constant filter
            except:
                v_drop = 0.0
            
            i_ma = (v_drop / 220.0) * 1000.0
            
            results.append({
                "v_in": v,
                "v_drop": v_drop,
                "i_ma": i_ma
            })
            
            print(f"  V_in: {v:.1f}V | I: {i_ma:.3f}mA", end="\r")

        df = pl.DataFrame(results)
        df.write_csv("led_current_refined.csv")
        
        plt.figure(figsize=(10, 6))
        plt.plot(df["v_in"], df["i_ma"], 'ro-', markersize=3)
        plt.xlabel("Input Voltage (V)")
        plt.ylabel("LED Current (mA)")
        plt.title("Refined LED I-V Characteristic (via 220 Ohm Shunt)")
        plt.grid(True)
        plt.savefig("led_current_refined_plot.png")
        print(f"\n--- Refined Sweep Complete ---")

if __name__ == "__main__":
    characterisation_refined()
