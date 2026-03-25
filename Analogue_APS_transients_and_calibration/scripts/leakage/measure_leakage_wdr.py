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

def run_leakage_wdr_sweep(label="Encased"):
    print(f"--- Starting WDR Leakage Sweep ({label}) ---")
    C_INT = 11e-12 
    V_REF = 1.0 # TTS Threshold
    
    # 10 points from 1ms to 10s
    t_ints = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    with Bench.open("bench.yaml") as bench:
        # Increase timeout for slow acquisitions
        bench.osc._backend.set_timeout(120000) 
        
        bench.psu.channel(1).set(voltage=5.0).on()
        bench.psu.channel(2).off() # Stimulus OFF

        # Scope Setup: CH1=Pixel, CH2=Reset Sync
        bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable() 
        bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
        bench.osc.trigger.setup_edge(source="CH2", level=2.5)

        results = []

        for t_target in t_ints:
            f = 1.0 / (2.0 * t_target) # 50% duty cycle timing
            print(f"\n--- integration Time: {t_target:.3f} s ({f:.3f} Hz) ---")
            
            # 1. Update Timing
            bench.siggen.channel(1).setup_square(frequency=f, amplitude=5.0, offset=2.5).enable()
            
            # 2. Adjust Scope Timebase
            # Total window = 1.2 * period. Scale = window / 10.
            period = 1.0 / f
            div_scale = (1.2 * period) / 10.0
            bench.osc.set_time_axis(scale=div_scale, position=period/2.0)
            
            # Wait for scope to settle (2 periods + padding)
            wait_time = period * 2.0 + 1.0
            print(f"  Stabilizing ({wait_time:.1f}s)...")
            time.sleep(wait_time)
            
            try:
                # 3. Capture and Analyze
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_vec, v_px, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 2 (V)"].to_numpy()
                
                # Detect edges on CH2
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    idx_e = rises[rises > idx_s][0]
                    
                    t0, v_start = t_vec[idx_s], v_px[idx_s]
                    actual_t_int = t_vec[idx_e] - t_vec[idx_s]
                    
                    win_v = v_px[idx_s:idx_e]
                    win_t = t_vec[idx_s:idx_e]
                    
                    # COMPARATOR LOGIC
                    trig_pts = np.where(win_v <= V_REF)[0]
                    if len(trig_pts) > 0:
                        # TTS Mode (Saturated)
                        hit = trig_pts[0]
                        v2, v1 = win_v[hit], win_v[max(0, hit-1)]
                        t2, t1 = win_t[hit], win_t[max(0, hit-1)]
                        t_cross = t1 + (V_REF - v1)*(t2-t1)/(v2-v1) if v1 != v2 else t2
                        t_sat = max(1e-9, t_cross - t0)
                        delta_v = (v_start - V_REF) * (actual_t_int / t_sat)
                        mode = "TTS"
                    else:
                        # INT Mode (Linear)
                        delta_v = v_start - win_v[-1]
                        mode = "INT"
                    
                    # 4. Derive Current
                    i_leak_pa = (C_INT * (delta_v / actual_t_int)) * 1e12
                    
                    results.append({
                        "t_int_s": actual_t_int,
                        "i_leak_pa": i_leak_pa,
                        "mode": mode
                    })
                    print(f"  Result: I_leak = {i_leak_pa:.3f} pA ({mode})")
                else:
                    print("  [ERROR] Window not found in trace.")
            except Exception as e:
                print(f"  [ERROR] Capture failed: {e}")

        # 5. Save and Plot
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv(f"leakage_wdr_{label.lower()}.csv")
            
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["t_int_s"], res_df["i_leak_pa"], 'bo-', label=f'{label} Current')
            plt.xlabel("Integration Time (s)")
            plt.ylabel("Derived Current (pA)")
            plt.title(f"Leakage Characterization: {label} Mode (C_int=11pF)")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"leakage_wdr_{label.lower()}.png")
            print(f"\n--- Complete. Report saved: leakage_wdr_{label.lower()}.png ---")
            return res_df

if __name__ == "__main__":
    run_leakage_wdr_sweep("Encased")
