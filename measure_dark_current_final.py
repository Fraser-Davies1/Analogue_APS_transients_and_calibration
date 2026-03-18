import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

# --- Patch for missing WaveformGenerator registration ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except: pass
# -------------------------------------------------------

def run_dark_current_final():
    print("--- Starting Final Dark Current Characterisation (CH2 Sync) ---")
    C_INT = 11e-12 
    t_ints = [0.5, 1.0, 2.0, 5.0]
    
    with Bench.open("bench.yaml") as bench:
        # Increase timeout for slow acquisitions
        bench.osc._backend.set_timeout(90000) 
        
        try:
            bench.psu.channel(1).set(voltage=5.0).on()
            bench.psu.channel(2).off()
            print("  [OK] VDD=5V, LED=OFF.")
        except:
            print("  [WARN] PSU setup failed.")

        # Setup Scope
        # CH1: Pixel Out (Sensitive scale, offset near reset level)
        # CH2: Reset Reference (Trigger source)
        bench.osc.channel(1).setup(scale=0.01, offset=4.8, coupling="DC").enable() 
        bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
        
        results = []

        for t_target in t_ints:
            print(f"\n--- Integration Time: {t_target:.1f} s ---")
            period = 2.0 * t_target
            freq = 1.0 / period
            
            # Setup Stimulus (Square Wave)
            bench.siggen.channel(1).setup_square(frequency=freq, amplitude=5.0, offset=2.5).enable()
            
            # Setup Scope Timebase
            # Total window = 1.5 * period. Scale per division = Window / 10.
            div_scale = (1.5 * period) / 10.0
            bench.osc.set_time_axis(scale=div_scale, position=period/2.0)
            
            # Ensure Trigger on Falling Edge of CH2
            bench.osc.trigger.setup_edge(source="CH2", level=2.5)
            
            # Wait for acquisition to complete (2 full periods for stabilization)
            wait_time = period * 2.0 + 2.0
            print(f"  Acquiring trace (Waiting {wait_time:.1f}s)...")
            time.sleep(wait_time)
            
            try:
                # Capture Trace data
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 2 (V)"].to_numpy()
                
                # Digital Edge Analysis on CH2
                is_high = v_rs > 2.5
                edges = np.diff(is_high.astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    potential_rises = rises[rises > idx_s]
                    
                    if len(potential_rises) > 0:
                        idx_e = potential_rises[0]
                        
                        actual_t = t_vec[idx_e] - t_vec[idx_s]
                        # Measure drop between edges (Average samples just before edges)
                        v_start = np.mean(v_px[max(0, idx_s-10):idx_s])
                        v_end = np.mean(v_px[max(0, idx_e-10):idx_e])
                        
                        delta_v = v_start - v_end
                        i_pa = (C_INT * (delta_v / actual_t)) * 1e12
                        
                        results.append({
                            "t_int_s": actual_t,
                            "delta_v": delta_v,
                            "i_pa": i_pa
                        })
                        print(f"  Result: ΔV={delta_v*1000:.3f}mV, I_leak={i_pa:.2f}pA")
                        continue

                print("  [ERROR] Integration window not detected in trace.")
            except Exception as e:
                print(f"  [ERROR] Trace capture failed: {e}")

        # Save and Plot
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("dark_current_results_final.csv")
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["t_int_s"], res_df["i_pa"], 'bo-')
            plt.xlabel("Integration Time (s)")
            plt.ylabel("Leakage Current (pA)")
            plt.title("APS Dark Current Characterisation (CH2 Sync)")
            plt.grid(True, alpha=0.3)
            plt.savefig("dark_current_final_report.png")
            print(f"\n--- Characterisation Complete. Plot: dark_current_final_report.png ---")
        else:
            print("\n--- Error: No valid data points were captured. ---")

if __name__ == "__main__":
    run_dark_current_final()
