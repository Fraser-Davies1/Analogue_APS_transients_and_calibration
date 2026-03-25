import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_tts_test():
    print("--- Starting Time-to-Saturation (TTS) Linearity Test ---")
    
    # Load I-V mapping
    iv_data = pl.read_csv("led_iv_high_res.csv")
    def get_current(v_psu):
        return np.interp(v_psu, iv_data["v_in"], iv_data["i_ma"])

    with Bench.open("bench.yaml") as bench:
        # Hardware Init
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        # 500Hz Reset
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        
        # OSC Setup: 500us/div (5ms window) to capture 2.5 cycles
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
        bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        v_steps = np.linspace(2.3, 4.5, 30) # High density to check transition
        results = []
        T_WINDOW = 0.002 # 2ms integration period
        V_SAT_THRESHOLD = 0.5 # Rail detection

        print("--- Sweeping V_in and Inferring Theoretical Delta V ---")
        for v in v_steps:
            curr = get_current(v)
            bench.psu.channel(2).set(voltage=v).on()
            time.sleep(0.5)
            
            data = bench.osc.read_channels([1, 3])
            df = data.values
            t = df["Time (s)"].to_numpy()
            v_px = df["Channel 1 (V)"].to_numpy()
            v_rs = df["Channel 3 (V)"].to_numpy()
            
            # Find falling edge of reset (t=0 for our integrator)
            edges = np.diff((v_rs > 2.5).astype(int))
            falls = np.where(edges == -1)[0]
            rises = np.where(edges == 1)[0]
            
            if len(falls) > 0:
                f_idx = falls[0]
                # End of integration window
                r_pts = rises[rises > f_idx]
                if len(r_pts) > 0:
                    r_idx = r_pts[0]
                    t0 = t[f_idx]
                    t_end = t[r_idx]
                    actual_T = t_end - t0
                    
                    v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                    
                    # Check for saturation within the window
                    window_v = v_px[f_idx:r_idx]
                    window_t = t[f_idx:r_idx]
                    sat_indices = np.where(window_v < V_SAT_THRESHOLD)[0]
                    
                    if len(sat_indices) > 0:
                        # Saturated!
                        sat_idx = sat_indices[0]
                        t_sat = window_t[sat_idx] - t0
                        # Calculate slope and infer delta_V at full T
                        slope = (v_start - V_SAT_THRESHOLD) / t_sat
                        inferred_delta = slope * actual_T
                        is_sat = True
                    else:
                        # Not saturated
                        v_final = np.mean(v_px[max(0, r_idx-10):r_idx])
                        inferred_delta = v_start - v_final
                        is_sat = False
                    
                    results.append({
                        "i_ma": curr,
                        "delta_v": inferred_delta,
                        "is_saturated": is_sat
                    })
                    print(f"  I={curr:.2f}mA | ΔV={'[EXT] ' if is_sat else '      '}{inferred_delta:.3f}V", end="\r")

        # 2. Report and Plot
        res_df = pl.DataFrame(results)
        res_df.write_csv("tts_results.csv")
        
        plt.figure(figsize=(10, 6))
        # Plot real integration
        real = res_df.filter(pl.col("is_saturated") == False)
        ext = res_df.filter(pl.col("is_saturated") == True)
        
        plt.plot(real["i_ma"], real["delta_v"], 'bo', label='Direct Integration')
        plt.plot(ext["i_ma"], ext["delta_v"], 'rx', label='TTS Extrapolated')
        
        # Fit overall line
        m, b = np.polyfit(res_df["i_ma"], res_df["delta_v"], 1)
        plt.plot(res_df["i_ma"], m*res_df["i_ma"] + b, 'k--', alpha=0.5, label=f'System Linearity R²={np.corrcoef(res_df["i_ma"], res_df["delta_v"])[0,1]**2:.4f}')
        
        plt.xlabel("Input Current (mA)")
        plt.ylabel("Inferred Pixel Drop (V)")
        plt.title("Extended Dynamic Range Linearity: Direct vs. Time-to-Saturation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("tts_linearity_verify.png")
        print("\n--- Test Complete. Report: tts_linearity_verify.png ---")

if __name__ == "__main__":
    run_tts_test()
