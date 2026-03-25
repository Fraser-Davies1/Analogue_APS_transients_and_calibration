import numpy as np
import polars as pl
from pytestlab import Bench
import time

def rerun_led_iv():
    print("--- Restoring LED I-V Data (100 Points) ---")
    R_SENSE = (220 * 50) / (220 + 50) # 40.74 Ohm effective
    
    with Bench.open("bench.yaml") as bench:
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.2).on()
        bench.osc.channel(2).setup(scale=0.1, offset=0.0, coupling="DC").enable()
        bench.osc.auto_scale()
        
        v_sweep = np.linspace(0.0, 5.0, 100)
        results = []

        for v in v_sweep:
            bench.psu.channel(2).set(voltage=v)
            time.sleep(0.3)
            try:
                res = bench.osc.measure_rms_voltage(2)
                v_drop = res.values.nominal_value if hasattr(res.values, "nominal_value") else res.values
                if v_drop > 10.0: v_drop = 0.0
                i_ma = (v_drop / R_SENSE) * 1000.0
                results.append({"v_in": v, "i_ma": i_ma})
            except: continue
            print(f"  V_in: {v:.2f}V | I: {i_ma:.2f}mA", end="\r")

        pl.DataFrame(results).write_csv("led_iv_high_res.csv")
        print("\n--- LED I-V Data Restored ---")

if __name__ == "__main__":
    rerun_led_iv()
