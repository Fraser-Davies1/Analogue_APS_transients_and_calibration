import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt
import os

# --- Patch for WaveformGenerator registration ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except: pass
# ------------------------------------------------

def run_leakage_capture(label="Encased"):
    print(f"--- Starting Leakage Sweep: {label} Mode ---")
    C_INT = 11e-12 
    V_REF = 1.0 
    
    # 10 Points: 1ms to 10s
    t_ints = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(150000) 
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off()

        # Scope prep
        bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable() 
        bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
        bench.osc.trigger.setup_edge(source="CH2", level=2.5)

        results = []

        for t_target in t_ints:
            period = 2.5 * t_target 
            freq = 1.0 / period
            print(f"  T_int = {t_target:5.3f} s ({freq:7.3f} Hz)...") # Changed to new line for better logging
            
            bench.siggen.channel(1).setup_square(frequency=freq, amplitude=5.0, offset=2.5).enable()
            div_scale = (1.5 * period) / 10.0
            bench.osc.set_time_axis(scale=div_scale, position=t_target/2.0)
            
            # Reduce stabilization for long points to avoid timeouts
            s_time = period * 2.0 + 1.0
            if t_target > 2: s_time = period * 1.5 + 1.0 
            
            time.sleep(s_time)
            
            try:
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_vec, v_px, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 2 (V)"].to_numpy()
                
                edges = np.diff((v_rs > 2.5).astype(int))
                falls, rises = np.where(edges == -1)[0], np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    idx_e = rises[rises > idx_s][0]
                    t0, v_start = t_vec[idx_s], v_px[idx_s]
                    actual_t = t_vec[idx_e] - t_vec[idx_s]
                    
                    win_v, win_t = v_px[idx_s:idx_e], t_vec[idx_s:idx_e]
                    
                    # TTS Logic
                    trig = np.where(win_v <= V_REF)[0]
                    if len(trig) > 0:
                        hit = trig[0]
                        v2, v1 = win_v[hit], win_v[max(0, hit-1)]
                        t2, t1 = win_t[hit], win_t[max(0, hit-1)]
                        t_cross = t1 + (V_REF - v1)*(t2-t1)/(v2-v1) if v1 != v2 else t2
                        delta_v = (v_start - V_REF) * (actual_t / (t_cross - t0))
                    else:
                        delta_v = v_start - win_v[-1]
                    
                    i_leak = (C_INT * (delta_v / actual_t)) * 1e12
                    results.append({"t_int": actual_t, "i_pa": i_leak})
                    
                    # Incremental save
                    pl.DataFrame(results).write_csv(f"leakage_data_{label.lower()}.csv")
                    print(f"    [SAVED] {i_leak:.3f} pA")
            except Exception as e: 
                print(f"    [ERROR] {e}")
                continue

        print(f"\n[OK] {label} data capture complete.")
        return pl.read_csv(f"leakage_data_{label.lower()}.csv")

def generate_comparison_plot():
    plt.figure(figsize=(10, 6))
    
    if os.path.exists("leakage_data_encased.csv"):
        d1 = pl.read_csv("leakage_data_encased.csv")
        plt.plot(d1["t_int"], d1["i_pa"], 'bo-', label='Encased (Dark Floor)')
        
    if os.path.exists("leakage_data_open.csv"):
        d2 = pl.read_csv("leakage_data_open.csv")
        plt.plot(d2["t_int"], d2["i_pa"], 'rs-', label='Open (Ambient Light)')
    
    plt.xlabel("Integration Time (s)")
    plt.ylabel("Derived Current (pA)")
    plt.title("Leakage Characterization Comparison: Encased vs. Open")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("leakage_comparison_report.png")
    print("Combined plot saved to leakage_comparison_report.png")

if __name__ == "__main__":
    import sys
    mode = "Encased" if len(sys.argv) < 2 else sys.argv[1]
    run_leakage_capture(mode)
    generate_comparison_plot()
