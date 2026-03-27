import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_tts_final():
    print("--- Starting Final Hardware-Emulated TTS Sweep ---")
    
    # 1. Current Mapping (from previous I-V sweep)
    iv_data = pl.read_csv("led_iv_high_res.csv")
    def get_current(v_psu):
        return np.interp(v_psu, iv_data["v_in"], iv_data["i_ma"])

    # 2. Parameters
    t_ints = [1.0, 0.1, 0.01, 0.001, 0.0001] # 1s to 100us
    v_steps = np.linspace(2.35, 4.5, 20)
    V_REF = 1.0 # Comparator Reference

    with Bench.open("bench.yaml") as bench:
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.2).on()
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        master_data = []

        for t_window in t_ints:
            f = 1.0 / t_window
            print(f"\nIntegration Time: {t_window*1000:.1f}ms")
            
            bench.siggen.channel(1).setup_square(frequency=f, amplitude=5.0, offset=2.5).enable()
            # View 1.5 cycles
            bench.osc.set_time_axis(scale=t_window/5.0, position=t_window/2.0)
            time.sleep(1.0)

            for v in v_steps:
                curr = get_current(v)
                try:
                    bench.psu.channel(2).set(voltage=v).on()
                    time.sleep(max(0.2, t_window * 1.1))
                    
                    data = bench.osc.read_channels([1, 3])
                    df = data.values
                    times = df["Time (s)"].to_numpy()
                    v_px = df["Channel 1 (V)"].to_numpy()
                    v_rs = df["Channel 3 (V)"].to_numpy()
                    
                    # Find integration start (falling edge of reset)
                    edges = np.diff((v_rs > 2.5).astype(int))
                    falls = np.where(edges == -1)[0]
                    if len(falls) == 0: continue
                    
                    idx_start = falls[0]
                    t0 = times[idx_start]
                    v_start = v_px[idx_start]
                    
                    # Limit search to one window
                    mask = (times >= t0) & (times <= t0 + t_window)
                    v_win = v_px[mask]
                    t_win = times[mask]
                    
                    # COMPARATOR LATCH
                    trigger_pts = np.where(v_win <= V_REF)[0]
                    
                    if len(trigger_pts) > 0:
                        idx_hit = trigger_pts[0]
                        # Sub-sample interpolation
                        v2, v1 = v_win[idx_hit], v_win[max(0, idx_hit-1)]
                        t2, t1 = t_win[idx_hit], t_win[max(0, idx_hit-1)]
                        
                        t_sat_abs = t1 + (V_REF - v1) * (t2 - t1) / (v2 - v1) if v1 != v2 else t2
                        t_sat = t_sat_abs - t0
                        
                        # Guard against zero-time crossing
                        if t_sat <= 0: t_sat = 1e-9 
                        
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

        # 3. Data Save
        df_final = pl.DataFrame(master_data)
        df_final.write_csv("tts_final_data.csv")

        # 4. Dual Plotting
        plt.figure(figsize=(16, 7))
        
        # Linear
        plt.subplot(1, 2, 1)
        for t in sorted(df_final["t_int_ms"].unique()):
            sub = df_final.filter(pl.col("t_int_ms") == t)
            plt.plot(sub["i_ma"], sub["delta_v"], 'o-', label=f"{t:.1f}ms")
        plt.xlabel("LED Current (mA)")
        plt.ylabel("Inferred Drop (V)")
        plt.title("TTS Linear Scale")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Log-Log
        plt.subplot(1, 2, 2)
        for t in sorted(df_final["t_int_ms"].unique()):
            sub = df_final.filter(pl.col("t_int_ms") == t)
            plt.loglog(sub["i_ma"], sub["delta_v"], 's-', label=f"{t:.1f}ms")
        plt.xlabel("LED Current (mA)")
        plt.ylabel("Inferred Drop (V)")
        plt.title("TTS Log-Log Scale")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig("tts_final_plots.png")
        print("\n--- Success! Data: tts_final_data.csv, Plots: tts_final_plots.png ---")

if __name__ == "__main__":
    run_tts_final()
