import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_verification():
    print("--- Starting APS Linearity Verification: I_pd vs. V_out (TTS) ---")
    
    R_SENSE = 220.0 
    C_INT = 11e-12
    V_REF = 1.0 # Comparator Threshold
    T_WINDOW = 0.002 # 2ms at 500Hz

    with Bench.open("bench.yaml") as bench:
        # Hardware Setup
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.2).on()
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() # Pixel
        bench.osc.channel(2).setup(scale=0.1, offset=0.0).enable() # Stimulus Sense
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Reset Ref
        
        bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        results = []
        # Fine-resolution sweep
        v_steps = np.concatenate([np.arange(2.2, 2.8, 0.02), np.arange(2.8, 4.6, 0.2)])

        print("--- Characterising ---")
        for v in v_steps:
            try:
                bench.psu.channel(2).set(voltage=v).on()
                time.sleep(0.5)
                
                data = bench.osc.read_channels([1, 2, 3])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_i = df["Channel 2 (V)"].to_numpy()
                v_rs = df["Channel 3 (V)"].to_numpy()
                
                # 1. Capture Ground Truth Stimulus
                i_led_ma = (np.mean(v_i) / R_SENSE) * 1000.0
                
                # 2. Capture TTS Readout
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    idx_e = rises[rises > idx_s][0]
                    t0, v_start = t_vec[idx_s], v_px[idx_s]
                    
                    win_v = v_px[idx_s:idx_e]
                    win_t = t_vec[idx_s:idx_e]
                    
                    # Comparator Latch
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
                        "i_led_ma": i_led_ma,
                        "v_out_inferred": v_out_tts
                    })
                    print(f"  I_led: {i_led_ma:.2f}mA -> V_out: {v_out_tts:.2f}V", end="\r")
            except: continue

        # 3. Final Analysis
        res_df = pl.DataFrame(results)
        
        # Calculate I_pd based on the known responsivity (0.15nA/mA)
        # Note: We scale I_led to get I_pd in nA
        res_df = res_df.with_columns([
            (pl.col("i_led_ma") * 0.15).alias("i_pd_na")
        ])
        
        res_df.write_csv("aps_linearity_final.csv")
        
        # Plotting
        plt.figure(figsize=(10, 6))
        x, y = res_df["i_pd_na"].to_numpy(), res_df["v_out_inferred"].to_numpy()
        
        plt.plot(x, y, 'mo-', markersize=4, label='APS Response (TTS)')
        
        # Linear Regression
        m, b = np.polyfit(x, y, 1)
        r_sq = np.corrcoef(x, y)[0, 1]**2
        plt.plot(x, m*x + b, 'k--', alpha=0.6, label=f'Linear Fit (R²={r_sq:.5f})')
        
        plt.xlabel("Photodiode Current I_pd (nA) [Stimulus]")
        plt.ylabel("Inferred Output Voltage V_out (V) [Readout]")
        plt.title(f"APS Linearity Verification: Photodiode Current vs. TTS Readout\nC_int=11pF, R²={r_sq:.5f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("aps_linearity_pd_vs_tts.png")
        
        print(f"\n--- Verification Complete. Plot saved: aps_linearity_pd_vs_tts.png ---")

if __name__ == "__main__":
    run_verification()
