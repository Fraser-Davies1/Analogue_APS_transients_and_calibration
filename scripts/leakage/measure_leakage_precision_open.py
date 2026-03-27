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

def run_leakage_precision_sweep(label="Open"):
    print(f"--- Starting Precision WDR Leakage Sweep: {label} Mode ---")
    C_INT = 11e-12 
    V_REF = 1.0 
    
    # 10 integration times from 0.1s to 10s
    t_ints = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        bench.osc._backend.set_timeout(150000) 
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off() 

        # Scope Setup
        bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable() 
        bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
        bench.osc.trigger.setup_edge(source="CH2", level=2.5)

        results = []

        for t_target in t_ints:
            period = 2.0 * t_target 
            freq = 1.0 / period
            print(f"\n--- integration Time: {t_target:.2f} s ---")
            
            bench.siggen.channel(1).setup_square(frequency=freq, amplitude=5.0, offset=2.5).enable()
            
            # Setup Scope Timebase to capture exactly 1.5 cycles
            div_scale = (1.5 * period) / 10.0
            bench.osc.set_time_axis(scale=div_scale, position=t_target/2.0)
            
            # Wait for acquisition stabilization
            wait_time = period * 2.2 + 2.0
            print(f"  Stabilizing for {wait_time:.1f}s...")
            time.sleep(wait_time)
            
            try:
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 2 (V)"].to_numpy()
                
                # Robust Edge Detection
                is_high = v_rs > 2.5
                edges = np.diff(is_high.astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    # Find first rising edge AFTER falling edge
                    pot_rises = rises[rises > idx_s]
                    if len(pot_rises) > 0:
                        idx_e = pot_rises[0]
                        
                        t0, v_start = t_vec[idx_s], v_px[idx_s]
                        actual_t = t_vec[idx_e] - t_vec[idx_s]
                        
                        win_v = v_px[idx_s:idx_e]
                        win_t = t_vec[idx_s:idx_e]
                        
                        # TTS Logic
                        trig_pts = np.where(win_v <= V_REF)[0]
                        if len(trig_pts) > 0:
                            hit = trig_pts[0]
                            v2, v1 = win_v[hit], win_v[max(0, hit-1)]
                            t2, t1 = win_t[hit], win_t[max(0, hit-1)]
                            t_cross = t1 + (V_REF - v1)*(t2-t1)/(v2-v1) if v1 != v2 else t2
                            t_sat = max(1e-9, t_cross - t0)
                            delta_v = (v_start - V_REF) * (actual_t / t_sat)
                            mode = "TTS"
                        else:
                            delta_v = v_start - win_v[-1]
                            mode = "INT"
                        
                        i_pa = (C_INT * (delta_v / actual_t)) * 1e12
                        results.append({"t_int": actual_t, "i_pa": i_pa, "mode": mode})
                        print(f"  SUCCESS: I={i_pa:.3f} pA ({mode})")
                        continue

                print("  [ERROR] Sync edges not found.")
            except Exception as e:
                print(f"  [ERROR] {e}")

        # 2. Results
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv(f"leakage_precision_{label.lower()}.csv")
            
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["t_int"], res_df["i_pa"], 'bo-', label=f'{label} Leakage')
            plt.xlabel("Integration Time (s)")
            plt.ylabel("Derived Current (pA)")
            plt.title(f"Precision Leakage Analysis: {label} Configuration")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"leakage_precision_{label.lower()}.png")
            print(f"\n--- Complete. Plot saved: leakage_precision_{label.lower()}.png ---")
            return res_df

if __name__ == "__main__":
    run_leakage_precision_sweep("Open")
