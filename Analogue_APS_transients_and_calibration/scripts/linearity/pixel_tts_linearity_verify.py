import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_tts_linearity_verify():
    print("--- Starting APS Linearity Verification (TTS Method) ---")
    
    R_SENSE = 220.0 
    C_INT = 11e-12
    V_REF = 1.0 # Comparator Threshold
    T_TOTAL = 0.002 # 2ms at 500Hz

    with Bench.open("bench.yaml") as bench:
        # Hardware Setup
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() # Pixel
            bench.osc.channel(2).setup(scale=0.1, offset=0.0).enable() # LED Current Sense
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Reset Sync
            
            bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Rails and Timing synchronized.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")

        results = []
        # Fine sweep through the active region
        v_steps = np.concatenate([np.arange(2.2, 2.5, 0.02), np.arange(2.5, 4.6, 0.1)])

        for v in v_steps:
            print(f"  Stimulus V_in = {v:.2f}V...", end="\r")
            try:
                bench.psu.channel(2).set(voltage=v).on()
                time.sleep(0.5)
            except: pass
            
            data = bench.osc.read_channels([1, 2, 3])
            df = data.values
            t_vec = df["Time (s)"].to_numpy()
            v_px = df["Channel 1 (V)"].to_numpy()
            v_i = df["Channel 2 (V)"].to_numpy()
            v_rs = df["Channel 3 (V)"].to_numpy()
            
            # 1. Independent Ground Truth: LED Current
            i_led_ma = (np.mean(v_i) / R_SENSE) * 1000.0
            
            # 2. TTS Analysis
            edges = np.diff((v_rs > 2.5).astype(int))
            falls = np.where(edges == -1)[0]
            rises = np.where(edges == 1)[0]
            
            if len(falls) > 0 and len(rises) > 0:
                idx_s = falls[0]
                idx_e = rises[rises > idx_s][0]
                
                t0, v_start = t_vec[idx_s], v_px[idx_s]
                
                # Check for saturation crossing within window
                win_v = v_px[idx_s:idx_e]
                win_t = t_vec[idx_s:idx_e]
                
                trig_pts = np.where(win_v <= V_REF)[0]
                
                if len(trig_pts) > 0:
                    # LATCHED (TTS Mode)
                    hit = trig_pts[0]
                    v2, v1 = win_v[hit], win_v[max(0, hit-1)]
                    t2, t1 = win_t[hit], win_t[max(0, hit-1)]
                    t_sat_abs = t1 + (V_REF - v1)*(t2-t1)/(v2-v1) if v1 != v2 else t2
                    
                    t_sat = max(1e-9, t_sat_abs - t0)
                    delta_v_tts = (v_start - V_REF) * (T_TOTAL / t_sat)
                    mode = "TTS"
                else:
                    # DIRECT (Linear Mode)
                    delta_v_tts = v_start - win_v[-1]
                    mode = "INT"
                
                results.append({
                    "i_led_ma": i_led_ma,
                    "delta_v_tts": delta_v_tts,
                    "mode": mode
                })

        # 3. Final Report
        res_df = pl.DataFrame(results)
        res_df.write_csv("aps_linearity_verify.csv")
        
        plt.figure(figsize=(10, 6))
        for m in ["INT", "TTS"]:
            sub = res_df.filter(pl.col("mode") == m)
            color = 'blue' if m == "INT" else 'red'
            plt.plot(sub["i_led_ma"], sub["delta_v_tts"], 'o', color=color, label=f'Mode: {m}')
        
        # Linear Fit
        x, y = res_df["i_led_ma"].to_numpy(), res_df["delta_v_tts"].to_numpy()
        m, b = np.polyfit(x, y, 1)
        r_sq = np.corrcoef(x, y)[0, 1]**2
        plt.plot(x, m*x + b, 'k--', alpha=0.5, label=f'Linear Fit (R²={r_sq:.4f})')
        
        plt.xlabel("LED Current (mA) [Independent Variable]")
        plt.ylabel("Inferred Pixel Drop (V) [Dependent Variable]")
        plt.title(f"APS Linearity Verification: TTS Extended Dynamic Range\nC_int=11pF, V_ref=1.0V")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("aps_linearity_report.png")
        print(f"\n--- Verification Complete. R²: {r_sq:.4f} ---")

if __name__ == "__main__":
    run_tts_linearity_verify()
