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

def run_leakage_sweep():
    print("--- Starting Background Leakage Sweep (1ms to 2s) ---")
    C_INT = 11e-12 
    
    # Logarithmic distribution from 1ms to 2s
    t_ints = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(30000)
        
        # Ensure LED is off for background check
        bench.psu.channel(2).off()
        bench.psu.channel(1).set(voltage=5.0).on()

        # Scope Config
        # CH1: Pixel Out. CH2: Reset Sync.
        bench.osc.channel(1).setup(scale=0.01, offset=4.9, coupling="DC").enable() 
        bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
        bench.osc.trigger.setup_edge(source="CH2", level=2.5)

        results = []

        for t in t_ints:
            print(f"  Testing T_int = {t*1000:.1f} ms...", end="\r")
            
            period = 2.5 * t # 40% duty cycle approx
            freq = 1.0 / period
            
            # Stimulus
            bench.siggen.channel(1).setup_square(frequency=freq, amplitude=5.0, offset=2.5).enable()
            
            # Scope Window (Catch 1.5 cycles)
            div_scale = (1.5 * period) / 10.0
            bench.osc.set_time_axis(scale=div_scale, position=t/2.0)
            
            # Wait for acquisition
            time.sleep(max(0.5, period * 2.0))
            
            try:
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 2 (V)"].to_numpy()
                
                # Edge detection
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    idx_e = rises[rises > idx_s][0]
                    
                    actual_t = t_vec[idx_e] - t_vec[idx_s]
                    # Average start/end to suppress noise
                    v_start = np.mean(v_px[max(0, idx_s-5):idx_s])
                    v_end = np.mean(v_px[max(0, idx_e-5):idx_e])
                    
                    delta_v = v_start - v_end
                    # Effective Current (nA)
                    i_eff_na = (C_INT * (delta_v / actual_t)) * 1e9
                    
                    results.append({
                        "t_int_s": actual_t,
                        "delta_v_mv": delta_v * 1000,
                        "i_leak_na": i_eff_na
                    })
            except: continue

        # 2. Analysis & Reporting
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("background_leakage_results.csv")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Voltage Drop vs Time
            ax1.plot(res_df["t_int_s"], res_df["delta_v_mv"], 'bo-')
            ax1.set_xlabel("Integration Time (s)")
            ax1.set_ylabel("Voltage Drop (mV)")
            ax1.set_title("Total Residual Integration Drop")
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Derived Current vs Time
            ax2.semilogx(res_df["t_int_s"], res_df["i_leak_na"], 'ro-')
            ax2.set_xlabel("Integration Time (s) [Log Scale]")
            ax2.set_ylabel("Derived Leakage (nA)")
            ax2.set_title("Derived Background Current")
            ax2.grid(True, which="both", alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("background_leakage_report.png")
            print(f"\n--- Complete. Report saved: background_leakage_report.png ---")
            print(f"Mean Residual Current: {res_df['i_leak_na'].mean():.4f} nA")
        else:
            print("\n[ERROR] No data points captured.")

if __name__ == "__main__":
    run_leakage_sweep()
