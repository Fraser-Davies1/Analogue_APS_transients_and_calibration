import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def characterisation_high_res():
    print("--- Starting High-Resolution LED I-V (100 Points) ---")
    print("--- Compensating for 50 Ohm Parallel Probe Loading ---")
    
    # R_total = 220 || 50
    R_SENSE = (220 * 50) / (220 + 50) 
    
    with Bench.open("bench.yaml") as bench:
        # 1. Initialization
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.2).on()
        bench.psu.channel(2).set(voltage=3.0, current_limit=0.2).on()
        
        # 2. Initial Auto-Scale
        bench.osc.channel(2).setup(scale=0.1, offset=0.0, coupling="DC").enable()
        bench.osc.auto_scale()
        time.sleep(2)

        v_sweep = np.linspace(0.0, 5.0, 100)
        results = []

        print(f"--- Sweeping 100 points ---")
        for v in v_sweep:
            bench.psu.channel(2).set(voltage=v)
            time.sleep(0.3) 
            
            try:
                # Capture RMS as proxy for DC Average
                res = bench.osc.measure_rms_voltage(2)
                v_drop = res.values.nominal_value if hasattr(res.values, "nominal_value") else res.values
                if v_drop > 10.0: v_drop = 0.0
                
                # Check for clipping
                # Note: get_channel_axis returns [scale, offset]
                axis = bench.osc.get_channel_axis(2)
                current_scale = axis[0]
                if v_drop > (current_scale * 3.2):
                    new_scale = current_scale * 2
                    bench.osc.channel(2).setup(scale=new_scale)
                    time.sleep(0.4)
                    v_drop = bench.osc.measure_rms_voltage(2).values
                    if hasattr(v_drop, "nominal_value"): v_drop = v_drop.nominal_value
                
                # Current through the LED string
                i_ma = (v_drop / R_SENSE) * 1000.0
                results.append({"v_in": v, "v_drop": v_drop, "i_ma": i_ma})
            except:
                continue
            
            print(f"  Step: {v:.2f}V | I: {i_ma:.2f}mA", end="\r")

        df = pl.DataFrame(results)
        df.write_csv("led_iv_high_res.csv")
        
        plt.figure(figsize=(10, 6))
        plt.plot(df["v_in"], df["i_ma"], 'b.', markersize=2, label='Data')
        plt.xlabel("Control Voltage (V)")
        plt.ylabel("Actual LED Current (mA)")
        plt.title(f"High-Res LED I-V (Corrected for 50 Ohm Load)\nR_eff = {R_SENSE:.2f} Ohm")
        plt.grid(True, alpha=0.3)
        plt.savefig("led_iv_high_res_plot.png")
        print(f"\n--- Characterisation Complete ---")

if __name__ == "__main__":
    characterisation_high_res()
