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

def run_full_range_leakage():
    print("--- Starting Full Range Leakage Sweep (1ms to 10s) ---")
    C_INT = 11e-12 
    
    # Decadal and long integration times
    t_ints = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        # 1. Critical Backend Tuning
        # Extremely long timeout for the 10s capture transfer
        bench.osc._backend.set_timeout(150000) 
        
        # 2. Hardware initialization
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.psu.channel(2).off() # STIMULUS OFF
            print("  [OK] VDD=5V, LED=OFF.")
        except:
            print("  [WARN] PSU failed. Proceeding with timing...")

        # Setup Scope
        # CH1: Pixel (Maximum Sensitivity)
        # CH2: Reset Reference (Timing)
        bench.osc.channel(1).setup(scale=0.005, offset=4.9, coupling="DC").enable() 
        bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
        bench.osc.trigger.setup_edge(source="CH2", level=2.5)

        results = []

        for t_target in t_ints:
            print(f"\n--- integration Time: {t_target:.3f} s ---")
            
            period = 2.5 * t_target 
            freq = 1.0 / period
            
            bench.siggen.channel(1).setup_square(frequency=freq, amplitude=5.0, offset=2.5).enable()
            
            # Scope Window: See 1.2 cycles
            div_scale = (1.2 * period) / 10.0
            bench.osc.set_time_axis(scale=div_scale, position=t_target/2.0)
            
            # Wait for stabilization (2 full periods)
            wait_time = period * 2.0 + 2.0
            print(f"  Stabilizing ({wait_time:.1f}s)...")
            time.sleep(wait_time)
            
            try:
                print("  Capturing waveform...")
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 2 (V)"].to_numpy()
                
                # Precise Edge Detection on CH2
                is_high = v_rs > 2.5
                edges = np.diff(is_high.astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    # Next rising edge
                    pots = rises[rises > idx_s]
                    if len(pots) > 0:
                        idx_e = pots[0]
                        
                        actual_t = t_vec[idx_e] - t_vec[idx_s]
                        # Measure drop (Averaging 50 samples for noise reduction)
                        v_start = np.mean(v_px[max(0, idx_s-50):idx_s])
                        v_end = np.mean(v_px[max(0, idx_e-50):idx_e])
                        
                        delta_v = v_start - v_end
                        i_leak_pa = (C_INT * (delta_v / actual_t)) * 1e12
                        
                        results.append({
                            "t_int": actual_t,
                            "delta_v_mv": delta_v * 1000,
                            "i_leak_pa": i_leak_pa
                        })
                        print(f"  SUCCESS: ΔV={delta_v*1000:.3f}mV, I_leak={i_leak_pa:.3f}pA")
                        continue

                print("  [ERROR] Integration window not found.")
            except Exception as e:
                print(f"  [ERROR] Trace capture failed: {e}")

        # 3. Final Reporting
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("leakage_full_range_results.csv")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Plot 1: Linear Delta V (Should grow with time)
            ax1.plot(res_df["t_int"], res_df["delta_v_mv"], 'bo-', label='Measured Drop')
            ax1.set_xlabel("Integration Time (s)")
            ax1.set_ylabel("Voltage Drop (mV)")
            ax1.set_title("Total Discharge vs. Time")
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Derived Current (Should be constant)
            ax2.semilogx(res_df["t_int"], res_df["i_leak_pa"], 'rs-', label='Calculated I_leak')
            ax2.set_xlabel("Integration Time (s) [Log Scale]")
            ax2.set_ylabel("Leakage Current (pA)")
            ax2.set_title("Stability of Derived Leakage Current")
            ax2.grid(True, which="both", alpha=0.3)
            ax2.set_ylim(bottom=-10, top=max(50, res_df["i_leak_pa"].max()*1.2))
            
            plt.tight_layout()
            plt.savefig("leakage_full_range_report.png")
            print(f"\n--- Characterisation Complete. Report: leakage_full_range_report.png ---")
        else:
            print("\n[ERROR] No data captured.")

if __name__ == "__main__":
    run_full_range_leakage()
