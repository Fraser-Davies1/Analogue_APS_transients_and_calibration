import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_robust_verification():
    print("--- Starting Robust APS Linearity Verification ---")
    
    R_SENSE = 220.0 
    C_INT = 11e-12
    V_REF = 1.0 
    T_WINDOW = 0.002 

    with Bench.open("bench.yaml") as bench:
        # Hardware Initialization with Retry
        print("  Initializing hardware...")
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() # Pixel
            bench.osc.channel(2).setup(scale=0.1, offset=0.0).enable() # I_led
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Sync
            
            bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Rails and Timing ready.")
        except Exception as e:
            print(f"  [WARNING] PSU Init failed: {e}. Set manually if needed.")

        results = []
        v_steps = np.concatenate([np.arange(2.35, 2.5, 0.02), np.arange(2.5, 4.6, 0.2)])

        for v in v_steps:
            print(f"  Advancing Stimulus V_in to {v:.2f}V...", end="\r")
            try:
                bench.psu.channel(2).set(voltage=v).on()
                time.sleep(0.5)
            except:
                # If bridge is down, we still try to measure current state from scope
                pass
            
            # 3-Channel Precision Capture
            try:
                data = bench.osc.read_channels([1, 2, 3])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_i = df["Channel 2 (V)"].to_numpy()
                v_rs = df["Channel 3 (V)"].to_numpy()
                
                # 1. Independent Stimulus Current
                i_led_ma = (np.mean(v_i) / R_SENSE) * 1000.0
                i_pd_na = i_led_ma * 0.15 # Using our established scaling factor
                
                # 2. TTS Readout
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    idx_e = rises[rises > idx_s][0]
                    t0, v_start = t_vec[idx_s], v_px[idx_s]
                    win_v, win_t = v_px[idx_s:idx_e], t_vec[idx_s:idx_e]
                    
                    trigger = np.where(win_v <= V_REF)[0]
                    if len(trigger) > 0:
                        hit = trigger[0]
                        v2, v1 = win_v[hit], win_v[max(0, hit-1)]
                        t2, t1 = win_t[hit], win_t[max(0, hit-1)]
                        t_cross = t1 + (V_REF - v1)*(t2-t1)/(v2-v1) if v1 != v2 else t2
                        t_sat = max(1e-9, t_cross - t0)
                        v_out_tts = (v_start - V_REF) * (T_WINDOW / t_sat)
                    else:
                        v_out_tts = v_start - win_v[-1]
                    
                    results.append({
                        "i_pd_na": i_pd_na,
                        "v_out_inferred": v_out_tts
                    })
            except: continue

        # 3. Final Analysis
        if results:
            res_df = pl.DataFrame(results)
            res_df.write_csv("aps_pd_linearity_final.csv")
            
            plt.figure(figsize=(10, 6))
            x, y = res_df["i_pd_na"].to_numpy(), res_df["v_out_inferred"].to_numpy()
            plt.plot(x, y, 'mo-', markersize=4, label='APS Output vs. PD Stimulus')
            
            # Linearity check
            m, b = np.polyfit(x, y, 1)
            r_sq = np.corrcoef(x, y)[0, 1]**2
            plt.plot(x, m*x + b, 'k--', alpha=0.6, label=f'Linear Fit (R²={r_sq:.5f})')
            
            plt.xlabel("Photogenerated Current I_pd (nA)")
            plt.ylabel("Inferred Output Voltage V_out (V)")
            plt.title(f"APS Linearity Verification: I_pd Stimulus vs. TTS Readout\nC_int=11pF, R²={r_sq:.5f}")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig("aps_pd_linearity_plot.png")
            print(f"\n--- Success! Plot: aps_pd_linearity_plot.png ---")

if __name__ == "__main__":
    run_robust_verification()
