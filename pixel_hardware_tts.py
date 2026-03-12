import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_hardware_tts_sweep():
    print("--- Starting Hardware-Emulated TTS Sweep ---")
    
    # 1. Load LED Characterization
    iv_data = pl.read_csv("led_iv_high_res.csv")
    def get_current(v_psu):
        return np.interp(v_psu, iv_data["v_in"], iv_data["i_ma"])

    # 2. Parameters
    t_ints = [1.0, 0.1, 0.01, 0.001, 0.0001] # 1s to 100us
    v_steps = np.linspace(2.35, 4.5, 25)
    V_REF = 1.0 # Comparator Threshold

    with Bench.open("bench.yaml") as bench:
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.2).on()
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable()
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        master_data = []

        for t_window in t_ints:
            f = 1.0 / t_window
            print(f"\n--- T_int: {t_window*1000:.1f}ms ({f:.1f}Hz) ---")
            
            bench.siggen.channel(1).setup_square(frequency=f, amplitude=5.0, offset=2.5).enable()
            # Set timebase to see exactly 1.5 cycles
            bench.osc.set_time_axis(scale=t_window/5.0, position=t_window/2.0)
            time.sleep(1.0)

            for v in v_steps:
                curr = get_current(v)
                bench.psu.channel(2).set(voltage=v).on()
                time.sleep(max(0.3, t_window * 1.1))
                
                data = bench.osc.read_channels([1, 3])
                df = data.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 3 (V)"].to_numpy()
                
                # Find the start of integration (Falling edge of CH3)
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                if len(falls) == 0: continue
                
                start_idx = falls[0]
                t0 = t_vec[start_idx]
                v_start = v_px[start_idx]
                
                # Search for the threshold crossing within the window
                # We limit the search to one period (t0 to t0 + t_window)
                search_mask = (t_vec >= t0) & (t_vec <= t0 + t_window)
                v_window = v_px[search_mask]
                t_window_vec = t_vec[search_mask]
                
                # Find the first index where v_px < V_REF
                trigger_indices = np.where(v_window < V_REF)[0]
                
                if len(trigger_indices) > 0:
                    # COMPARATOR TRIGGERED
                    idx = trigger_indices[0]
                    # Linear interpolation for sub-sample timing precision
                    # V1 is just before crossing, V2 is just after
                    v2, v1 = v_window[idx], v_window[max(0, idx-1)]
                    t2, t1 = t_window_vec[idx], t_window_vec[max(0, idx-1)]
                    
                    # Exact time t_sat where v = V_REF
                    if v1 != v2:
                        t_sat_abs = t1 + (V_REF - v1) * (t2 - t1) / (v2 - v1)
                    else:
                        t_sat_abs = t2
                    
                    t_sat = t_sat_abs - t0
                    inferred_v = (v_start - V_REF) * (t_window / t_sat)
                    mode = "TTS"
                else:
                    # LINEAR INTEGRATION (No Saturation)
                    v_final = v_window[-1]
                    inferred_v = v_start - v_final
                    mode = "INT"
                
                master_data.append({
                    "t_int_ms": t_window * 1000,
                    "i_led_ma": curr,
                    "delta_v": inferred_v,
                    "mode": mode
                })
                print(f"  I:{curr:.1f}mA | V:{inferred_v:.2f}V ({mode})", end="\r")

        # 3. Save
        pl.DataFrame(master_data).write_csv("hardware_tts_results.csv")
        print("\n--- Sweep Complete ---")

if __name__ == "__main__":
    run_hardware_tts_sweep()
