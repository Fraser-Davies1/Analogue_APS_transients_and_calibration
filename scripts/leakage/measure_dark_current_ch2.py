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

def run_dark_current_ch2():
    print("--- Starting Dark Current Characterisation (CH2 Reset Sync) ---")
    C_INT = 11e-12 
    
    # Target integration times
    t_ints = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        # 1. Critical Backend Tuning
        # We must increase the network timeout for 10-second scope captures
        bench.osc._backend.set_timeout(120000) # 120 seconds
        
        # 2. Hardware Initialization
        try:
            bench.psu.channel(1).set(voltage=5.0).on()
            bench.psu.channel(2).off() # LED MUST BE OFF
            print("  [OK] VDD=5V, LED=OFF.")
        except:
            print("  [WARN] PSU setup failed. Proceeding with timing...")

        # Setup Scope Vertical
        bench.osc.channel(1).setup(scale=0.01, offset=4.9, coupling="DC").enable() # Pixel Out
        bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() # Reset Sync
        
        results = []

        for t_target in t_ints:
            print(f"\n--- integration Time: {t_target:.1f} s ---")
            
            period = 2.0 * t_target
            freq = 1.0 / period
            
            # Update Stimulus
            bench.siggen.channel(1).setup_square(frequency=freq, amplitude=5.0, offset=2.5).enable()
            
            # Update Scope Timebase: ensure we see exactly 1.2 periods
            # Window = 1.2 * period. Scale = Window / 10 divisions.
            div_scale = (1.2 * period) / 10.0
            bench.osc.set_time_axis(scale=div_scale, position=t_target/2.0)
            
            # Setup Trigger on CH2 (Reset Reference)
            bench.osc.trigger.setup_edge(source="CH2", level=2.5)
            
            # Wait for acquisition (2 periods + scope arming time)
            print(f"  Waiting {period*2 + 2:.1f}s for capture...")
            time.sleep(period * 2.0 + 2.0)
            
            try:
                # Capture Trace
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 2 (V)"].to_numpy()
                
                # Digital Edge Detection on CH2
                is_high = v_rs > 2.5
                edges = np.diff(is_high.astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    # Find rising edge after the falling one
                    pots = rises[rises > idx_s]
                    if len(pots) > 0:
                        idx_e = pots[0]
                        
                        actual_t = t_vec[idx_e] - t_vec[idx_s]
                        # Measure start and end levels (Mean of 20 samples to filter scope noise)
                        v_start = np.mean(v_px[max(0, idx_s-20):idx_s])
                        v_end = np.mean(v_px[max(0, idx_e-20):idx_e])
                        
                        delta_v = v_start - v_end
                        i_dark_pa = (C_INT * (delta_v / actual_t)) * 1e12
                        
                        results.append({
                            "t_int_s": actual_t,
                            "delta_v": delta_v,
                            "i_dark_pa": i_dark_pa
                        })
                        print(f"  SUCCESS: Drop={delta_v*1000:.3f}mV, I_dark={i_dark_pa:.2f}pA")
                        continue

                print("  [ERROR] Integration window not found in trace. Check CH2 probe.")
            except Exception as e:
                print(f"  [ERROR] Communication Error: {e}")

        # 3. Save and Report
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("dark_current_ch2_results.csv")
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["t_int_s"], res_df["i_dark_pa"], 'bo-')
            plt.xlabel("Integration Time (s)")
            plt.ylabel("Leakage Current (pA)")
            plt.title("APS Dark Current Characterisation (CH2 Sync)")
            plt.grid(True, alpha=0.3)
            plt.savefig("dark_current_ch2_report.png")
            print(f"\n--- Final Report saved: dark_current_ch2_report.png ---")
        else:
            print("\n--- Failed to capture data. Ensure Reset is on CH2. ---")

if __name__ == "__main__":
    run_dark_current_ch2()
