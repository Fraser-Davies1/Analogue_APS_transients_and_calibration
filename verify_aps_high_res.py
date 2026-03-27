import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_high_res_verification():
    print("--- Starting High-Resolution APS Linearity Verification ---")
    
    R_SENSE = 220.0 
    C_INT = 11e-12
    V_REF = 1.0 
    T_WINDOW = 0.002 

    # 1. Define High-Density Voltage Steps
    v_points = np.unique(np.concatenate([
        np.arange(2.2, 2.8, 0.01),  # Fine: 10mV steps
        np.arange(2.8, 4.65, 0.05)   # Standard: 50mV steps
    ]))

    with Bench.open("bench.yaml") as bench:
        # Initial Setup
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(2).setup(scale=0.1, offset=0.0).enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
            
            bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print(f"  [OK] Ready to capture {len(v_points)} points.")
        except Exception as e:
            print(f"  [ERROR] Hardware init failed: {e}")
            return

        results = []

        for idx, v in enumerate(v_points):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(v_points)} (V_in = {v:.2f}V)...", end="\r")
            
            try:
                bench.psu.channel(2).set(voltage=v).on()
                time.sleep(0.4) # Slightly shorter settle for high-density runs
                
                # Precise 3-Channel Capture
                data = bench.osc.read_channels([1, 2, 3])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_i = df["Channel 2 (V)"].to_numpy()
                v_rs = df["Channel 3 (V)"].to_numpy()
                
                # 1. Derive I_pd from CH2
                i_led_ma = (np.mean(v_i) / R_SENSE) * 1000.0
                i_pd_na = i_led_ma * 0.15 # System constant
                
                # 2. Derive V_out via TTS logic
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    idx_e = rises[rises > idx_s][0]
                    t0, v_start = t_vec[idx_s], v_px[idx_s]
                    win_v, win_t = v_px[idx_s:idx_e], t_vec[idx_s:idx_e]
                    
                    trig = np.where(win_v <= V_REF)[0]
                    if len(trig) > 0:
                        hit = trig[0]
                        v2, v1 = win_v[hit], win_v[max(0, hit-1)]
                        t2, t1 = win_t[hit], win_t[max(0, hit-1)]
                        t_cross = t1 + (V_REF - v1)*(t2-t1)/(v2-v1) if v1 != v2 else t2
                        t_sat = max(1e-9, t_cross - t0)
                        v_out = (v_start - V_REF) * (T_WINDOW / t_sat)
                    else:
                        v_out = v_start - win_v[-1]
                    
                    results.append({"i_pd_na": i_pd_na, "v_out": v_out})
            except: continue

        # 3. Process and Plot
        res_df = pl.DataFrame(results)
        res_df.write_csv("aps_linearity_high_res.csv")
        
        plt.figure(figsize=(10, 6))
        x, y = res_df["i_pd_na"].to_numpy(), res_df["v_out"].to_numpy()
        plt.plot(x, y, 'b.', markersize=2, alpha=0.8, label='High-Res Data')
        
        # Fit
        m, b = np.polyfit(x, y, 1)
        r_sq = np.corrcoef(x, y)[0, 1]**2
        plt.plot(x, m*x + b, 'r--', label=f'Linear Fit (R²={r_sq:.5f})')
        
        plt.xlabel("Photogenerated Current I_pd (nA)")
        plt.ylabel("Inferred Output Voltage V_out (V)")
        plt.title(f"High-Resolution APS Linearity Verification\n(10mV steps in turn-on, R²={r_sq:.5f})")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig("aps_high_res_linearity.png")
        print(f"\n--- Sweep Complete. R²: {r_sq:.5f}. Plot: aps_high_res_linearity.png ---")

if __name__ == "__main__":
    run_high_res_verification()
