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

def run_leakage_tts_sweep(label="Open"):
    print(f"--- Starting WDR Leakage Sweep: {label} Mode ---")
    C_INT = 11e-12 
    V_REF = 1.0 
    
    # 7 points from 100ms to 20s
    t_ints = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(250000) # 250s timeout
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off() # LED Off

        # Scope Setup: CH1=Pixel, CH2=Reset Ref
        bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable() 
        bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
        bench.osc.trigger.setup_edge(source="CH2", level=2.5)

        results = []

        for t_target in t_ints:
            period = 2.0 * t_target 
            freq = 1.0 / period
            print(f"\n--- integration Time: {t_target:.1f} s ({freq:.3f} Hz) ---")
            
            bench.siggen.channel(1).setup_square(frequency=freq, amplitude=5.0, offset=2.5).enable()
            div_scale = (1.2 * period) / 10.0
            bench.osc.set_time_axis(scale=div_scale, position=t_target/2.0)
            
            wait_time = period * 2.0 + 2.0
            print(f"  Stabilizing ({wait_time:.1f}s)...")
            time.sleep(wait_time)
            
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
                    
                    # TTS Comparator Logic
                    trig_pts = np.where(win_v <= V_REF)[0]
                    if len(trig_pts) > 0:
                        hit = trig_pts[0]
                        v2, v1 = win_v[hit], win_v[max(0, hit-1)]
                        t2, t1 = win_t[hit], win_t[max(0, hit-1)]
                        t_cross = t1 + (V_REF-v1)*(t2-t1)/(v2-v1) if v1 != v2 else t2
                        t_sat = max(1e-9, t_cross - t0)
                        delta_v = (v_start - V_REF) * (actual_t / t_sat)
                        mode = "TTS"
                    else:
                        delta_v = v_start - win_v[-1]
                        mode = "INT"
                    
                    i_leak_pa = (C_INT * (delta_v / actual_t)) * 1e12
                    results.append({"t_int": actual_t, "i_pa": i_leak_pa, "mode": mode})
                    print(f"  SUCCESS: I={i_leak_pa:.3f} pA ({mode})")
            except Exception as e:
                print(f"  [ERROR] {e}")

        # 2. Final Report
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv(f"leakage_tts_{label.lower()}.csv")
            
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["t_int"], res_df["i_pa"], 'ro-', label=f'Current vs. Time ({label})')
            plt.xlabel("Integration Time (s)")
            plt.ylabel("Derived Current (pA)")
            plt.title(f"Background Leakage Current: {label} Sensor")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"leakage_tts_{label.lower()}.png")
            print(f"\n--- Complete. Plot: leakage_tts_{label.lower()}.png ---")

if __name__ == "__main__":
    run_leakage_tts_sweep("Open")
