import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_wdr_family_sweep():
    print("--- Starting WDR Family Sweep (Semilog-X) ---")
    R_SENSE = 220.0
    V_REF = 1.0 
    t_ints = [0.0005, 0.005, 0.05, 0.5]
    v_points = np.linspace(2.2, 4.8, 15) 

    with Bench.open("bench.yaml") as bench:
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        # Explicitly set DC coupling
        bench.osc._send_command(":CHANnel1:COUPling DC")
        bench.osc._send_command(":CHANnel2:COUPling DC")
        bench.osc._send_command(":CHANnel3:COUPling DC")
        
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
        bench.osc.channel(2).setup(scale=0.5, offset=0.0).enable() 
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
        
        master_results = []

        for t_window in t_ints:
            f = 1.0 / (2 * t_window)
            print(f"\n--- Integration Time: {t_window*1000:.1f}ms ---")
            bench.siggen.channel(1).setup_square(frequency=f, amplitude=5.0, offset=2.5).enable()
            bench.osc.set_time_axis(scale=(2*t_window)/10.0, position=t_window)
            time.sleep(1.0)

            for v in v_points:
                try:
                    bench.psu.channel(2).set(voltage=v).on()
                    time.sleep(max(0.1, t_window * 0.5))
                    
                    data = bench.osc.read_channels([1, 2, 3])
                    df = data.values
                    t_vec, v_px, v_i, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 2 (V)"].to_numpy(), df["Channel 3 (V)"].to_numpy()
                    
                    # Measurement from scope
                    i_led_ma = (np.mean(v_i) / R_SENSE) * 1000.0
                    # Safety fallback: use PSU current if scope is dead
                    if abs(i_led_ma) < 0.1:
                         i_led_ma = bench.psu.read_current(2) * 1000.0

                    rs_mid = (np.max(v_rs) + np.min(v_rs)) / 2.0
                    edges = np.diff((v_rs > rs_mid).astype(int))
                    falls = np.where(edges == -1)[0]
                    
                    if len(falls) > 0:
                        idx_s = falls[0]
                        rises = np.where(edges == 1)[0]
                        valid_rises = rises[rises > idx_s]
                        idx_e = valid_rises[0] if len(valid_rises) > 0 else len(v_px)-1
                        
                        win_v = v_px[idx_s:idx_e]
                        win_t = t_vec[idx_s:idx_e]
                        t0, v_start = t_vec[idx_s], v_px[idx_s]
                        
                        trigger = np.where(win_v <= V_REF)[0]
                        if len(trigger) > 0:
                            hit = trigger[0]
                            t_cross = win_t[hit]
                            t_sat = max(1e-8, t_cross - t0)
                            v_out = (v_start - V_REF) * (t_window / t_sat)
                        else:
                            v_out = max(0, v_start - win_v[-1])
                        
                        master_results.append({"t_int_ms": t_window * 1000, "i_led_ma": i_led_ma, "v_out_inferred": v_out})
                        print(f"  I={i_led_ma:.2f}mA, V_out={v_out:.2f}V", end="\r")
                except: continue

        res_df = pl.DataFrame(master_results)
        res_df.write_csv("pixel_wdr_family_data.csv")
        print("\nData saved.")

if __name__ == "__main__":
    run_wdr_family_sweep()
