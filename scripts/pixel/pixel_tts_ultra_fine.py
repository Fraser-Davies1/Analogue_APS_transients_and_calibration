import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_ultra_fine_tts():
    print("--- Starting Ultra-Fine Hardware-Emulated TTS Sweep ---")
    
    # 1. LED I-V Mapping
    try:
        iv_data = pl.read_csv("led_iv_high_res.csv")
        def get_current(v_psu):
            return np.interp(v_psu, iv_data["v_in"], iv_data["i_ma"])
    except:
        print("Warning: led_iv_high_res.csv not found. Current will be approximate.")
        def get_current(v_psu): return (v_psu - 2.3) / 40.74 * 1000 if v_psu > 2.3 else 0

    # 2. Define Refined Voltage Points
    v_points = np.unique(np.concatenate([
        np.arange(2.2, 2.8, 0.01),  # Fine region (60 points)
        np.arange(2.8, 4.6, 0.1)    # Coarse region (18 points)
    ]))

    t_ints = [1.0, 0.1, 0.01, 0.001, 0.0001]
    V_REF = 1.0 # Virtual Comparator Trip Point

    with Bench.open("bench.yaml") as bench:
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.2).on()
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        master_data = []

        for t_window in t_ints:
            f = 1.0 / t_window
            print(f"\n--- T_int: {t_window*1000:.1f}ms ({f:.2f}Hz) ---")
            
            bench.siggen.channel(1).setup_square(frequency=f, amplitude=5.0, offset=2.5).enable()
            bench.osc.set_time_axis(scale=t_window/5.0, position=t_window/2.0)
            time.sleep(1.0)

            for idx, v in enumerate(v_points):
                curr = get_current(v)
                if idx % 10 == 0:
                    print(f"  Progress: {v:.2f}V ({curr:.2f}mA)...", end="\r")
                
                try:
                    bench.psu.channel(2).set(voltage=v).on()
                    time.sleep(max(0.1, t_window * 1.1))
                    
                    data = bench.osc.read_channels([1, 3])
                    df = data.values
                    t_vec, v_px, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 3 (V)"].to_numpy()
                    
                    # Find integration start
                    edges = np.diff((v_rs > 2.5).astype(int))
                    falls = np.where(edges == -1)[0]
                    if len(falls) == 0: continue
                    
                    idx_start = falls[0]
                    t0, v_start = t_vec[idx_start], v_px[idx_start]
                    
                    # Search window
                    mask = (t_vec >= t0) & (t_vec <= t0 + t_window)
                    v_win, t_win = v_px[mask], t_vec[mask]
                    
                    # Comparator Logic
                    trigger_pts = np.where(v_win <= V_REF)[0]
                    
                    if len(trigger_pts) > 0:
                        hit = trigger_pts[0]
                        # Sub-sample crossing time
                        v2, v1 = v_win[hit], v_win[max(0, hit-1)]
                        t2, t1 = t_win[hit], t_win[max(0, hit-1)]
                        t_cross = t1 + (V_REF - v1)*(t2-t1)/(v2-v1) if v1 != v2 else t2
                        
                        t_sat = max(1e-9, t_cross - t0)
                        val = (v_start - V_REF) * (t_window / t_sat)
                        mode = "TTS"
                    else:
                        val = v_start - v_win[-1]
                        mode = "INT"
                    
                    master_data.append({
                        "t_int_ms": t_window * 1000,
                        "i_ma": curr,
                        "delta_v": val,
                        "mode": mode
                    })
                except: continue

        # 3. Save and Report
        df_final = pl.DataFrame(master_data)
        df_final.write_csv("tts_ultra_fine_results.csv")
        
        plt.figure(figsize=(16, 8))
        
        # Plot 1: Linear Scale
        plt.subplot(1, 2, 1)
        for t in sorted(df_final["t_int_ms"].unique()):
            sub = df_final.filter(pl.col("t_int_ms") == t)
            plt.plot(sub["i_ma"], sub["delta_v"], 'o-', markersize=3, label=f"{t:.1f}ms")
        plt.xlabel("Input Current (mA)")
        plt.ylabel("Inferred Voltage Drop (V)")
        plt.title("WDR Linearity: Linear Scale")
        plt.grid(True, alpha=0.3); plt.legend()

        # Plot 2: Log-Log Scale
        plt.subplot(1, 2, 2)
        for t in sorted(df_final["t_int_ms"].unique()):
            sub = df_final.filter(pl.col("t_int_ms") == t)
            plt.loglog(sub["i_ma"], sub["delta_v"], 's-', markersize=3, label=f"{t:.1f}ms")
        plt.xlabel("Input Current (mA)")
        plt.ylabel("Inferred Voltage Drop (V)")
        plt.title("WDR Linearity: Log-Log Scale")
        plt.grid(True, which="both", alpha=0.3); plt.legend()

        plt.tight_layout()
        plt.savefig("tts_ultra_fine_report.png")
        print("\n--- Sweep Complete. Report: tts_ultra_fine_report.png ---")

if __name__ == "__main__":
    run_ultra_fine_tts()
