import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_validation():
    print("--- Starting Linearized Light Validation ---")
    
    # 1. Load the LUT
    try:
        lut = pl.read_csv("led_lin_lut.csv")
    except:
        print("Error: led_lin_lut.csv not found. Run calibrate_led_np.py first.")
        return

    with Bench.open("bench.yaml") as bench:
        # Setup Rails
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
        bench.osc.set_time_axis(scale=1e-3, position=5e-3)
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        results = []
        
        for row in lut.iter_rows(named=True):
            percent = row['light_percent']
            v_set = row['v_led_control']
            
            print(f"  Targeting {percent}% Light (Setting PSU to {v_set:.4f}V)...")
            
            bench.psu.channel(2).set(voltage=v_set).on()
            time.sleep(0.6) # Stable settling
            
            # Precision analysis
            data = bench.osc.read_channels([1, 3])
            df = data.values
            ch1, ch3 = df["Channel 1 (V)"].to_numpy(), df["Channel 3 (V)"].to_numpy()
            
            edges = np.diff((ch3 > 2.5).astype(int))
            f_idx = np.where(edges == -1)[0][0]
            r_pts = np.where(edges == 1)[0]
            r_idx = r_pts[r_pts > f_idx][0]
            
            v_start = np.mean(ch1[f_idx-10:f_idx])
            v_end = np.mean(ch1[r_idx-10:r_idx])
            
            results.append({
                "target_percent": percent,
                "actual_drop": v_start - v_end,
                "v_psu_used": v_set
            })

        # Final Analysis
        res_df = pl.DataFrame(results)
        x = res_df["target_percent"].to_numpy()
        y = res_df["actual_drop"].to_numpy()
        
        r_sq = np.corrcoef(x, y)[0, 1]**2
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'gs-', label=f'Linearized Response (R²={r_sq:.5f})')
        plt.xlabel("Target Light Intensity (%)")
        plt.ylabel("Measured Pixel Drop (V)")
        plt.title("Validation: Linearized Light Output via LUT Compensation")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("validation_plot.png")
        
        print(f"\n--- Validation Complete ---")
        print(f"System Linearity after Compensation: R² = {r_sq:.5f}")
        if r_sq > 0.99:
            print("RESULT: SUCCESS. The LED non-linearity has been successfully accounted for.")
        else:
            print("RESULT: MARGINAL. Residual non-linearity exists.")

if __name__ == "__main__":
    run_validation()
