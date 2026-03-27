import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

# --- Patch for WaveformGenerator registration ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except: pass
# ------------------------------------------------

def run_true_dark_characterization():
    print("--- Starting True Dark Current Sweep (Blu-Tac Encapsulation) ---")
    C_INT = 11e-12 
    t_ints = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(200000) 
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.psu.channel(2).off() # LED physical power off

        # High-sensitivity Zoom on the Reset Rail
        bench.osc.channel(1).setup(scale=0.005, offset=4.8, coupling="DC").enable()
        bench.osc.set_time_axis(scale=10e-3, position=0.0) 

        results = []

        for t in t_ints:
            print(f"  Measuring Integration T = {t:4.1f}s...", end="\r")
            
            # 1. RESET PHASE (Clear Cap)
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()
            time.sleep(2.0) 
            # Sample Start Mean
            data_start = bench.osc.read_channels(1)
            v_start = data_start.values["Channel 1 (V)"].mean()
            
            # 2. INTEGRATION PHASE (Gate Low)
            bench.siggen.channel(1).setup_dc(offset=0.0).enable()
            time.sleep(t)
            
            # 3. SAMPLE PHASE (End Mean)
            data_end = bench.osc.read_channels(1)
            v_end = data_end.values["Channel 1 (V)"].mean()
            
            delta_v = v_start - v_end
            i_app_pa = (C_INT * (delta_v / t)) * 1e12
            
            results.append({
                "t_int_s": t,
                "v_start": v_start,
                "v_end": v_end,
                "delta_v_mv": delta_v * 1000,
                "i_apparent_pa": i_app_pa
            })
            
            # Re-assert Reset to prevent long-term rail slamming
            bench.siggen.channel(1).setup_dc(offset=5.0).enable()

        # 4. Data Processing
        df = pl.DataFrame(results)
        df.write_csv("true_dark_results.csv")
        
        # Regression to find the real current (Slope)
        x = df["t_int_s"].to_numpy()
        y = df["delta_v_mv"].to_numpy()
        slope_mv_s, pedestal_mv = np.polyfit(x, y, 1)
        
        # True I = Slope * C
        true_dark_current_fa = (slope_mv_s / 1000.0) * C_INT * 1e15

        # 5. Output Results
        print("\n\n--- True Dark Current Analysis ---")
        print(f"Detected Switch Pedestal: {pedestal_mv:.3f} mV")
        print(f"Derived True Dark Current: {true_dark_current_fa:.3f} fA")
        print("-" * 40)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df["t_int_s"], df["delta_v_mv"], 'bo-', label='Measured Drop (mV)')
        # Plot the fit
        plt.plot(x, slope_mv_s * x + pedestal_mv, 'r--', alpha=0.6, label='Linear Drift Model')
        
        plt.xlabel("Integration Time (s)")
        plt.ylabel("Voltage Drop (mV)")
        plt.title(f"True Dark Characterisation: Blu-Tac Encased\nI_dark = {true_dark_current_fa:.1f} fA")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("true_dark_current_plot.png")
        print("\nResults archived. Plot: true_dark_current_plot.png")

if __name__ == "__main__":
    run_true_dark_characterization()
