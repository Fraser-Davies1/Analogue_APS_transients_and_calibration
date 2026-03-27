import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def characterisation_final_attempt():
    print("--- Final Attempt: Automated Scope Scaling for I-V Characterisation ---")
    
    with Bench.open("bench.yaml") as bench:
        # 1. Setup Rails
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        # Set to 4.0V (Safe high brightness) to allow scope to find the signal
        bench.psu.channel(2).set(voltage=4.0, current_limit=0.1).on()
        time.sleep(2)
        
        # 2. Scope Scaling
        print("  Finding signal on Scope CH2...")
        bench.osc.channel(2).enable()
        bench.osc.auto_scale()
        time.sleep(2)

        v_in_sweep = np.arange(0.0, 5.1, 0.2) # Coarser steps for speed
        results = []

        print(f"--- Sweeping ---")
        for v in v_in_sweep:
            bench.psu.channel(2).set(voltage=v)
            time.sleep(0.5) 
            
            try:
                # We use the RMS measurement as it's more robust to noise
                res = bench.osc.measure_rms_voltage(2)
                v_drop = res.values.nominal_value if hasattr(res.values, "nominal_value") else res.values
                if v_drop > 10.0: v_drop = 0.0 
            except:
                v_drop = 0.0
            
            i_ma = (v_drop / 220.0) * 1000.0
            results.append({"v_in": v, "i_ma": i_ma})
            print(f"  V_in: {v:.1f}V | I: {i_ma:.3f}mA", end="\r")

        pl.DataFrame(results).write_csv("led_iv_final.csv")
        print("\n--- Complete. Check led_iv_final.csv ---")

if __name__ == "__main__":
    characterisation_final_attempt()
