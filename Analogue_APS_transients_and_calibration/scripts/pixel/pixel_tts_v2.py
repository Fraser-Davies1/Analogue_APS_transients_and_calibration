import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_tts_v2():
    print("--- Starting Robust Slope-Based TTS Test ---")
    
    # 1. Calibration Mapping
    iv_data = pl.read_csv("led_iv_high_res.csv")
    def get_current(v_psu):
        return np.interp(v_psu, iv_data["v_in"], iv_data["i_ma"])

    with Bench.open("bench.yaml") as bench:
        # Hardware Setup
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        
        # OSC: Use 1ms/div to ensure 500Hz cycles are clearly visible
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
        bench.osc.set_time_axis(scale=1e-3, position=5e-3)
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)
        
        # Force high resolution
        try: bench.osc._send_command(":WAVeform:POINts:MODE MAXimum")
        except: pass

        v_steps = np.linspace(2.4, 4.5, 25)
        results = []
        T_INT = 0.002 # 2ms period

        print("--- Characterising Slope vs Input Current ---")
        for v in v_steps:
            curr = get_current(v)
            bench.psu.channel(2).set(voltage=v).on()
            time.sleep(0.5)
            
            data = bench.osc.read_channels([1, 3])
            df = data.values
            t = df["Time (s)"].to_numpy()
            v_px = df["Channel 1 (V)"].to_numpy()
            v_rs = df["Channel 3 (V)"].to_numpy()
            
            # 2. Identify Integration Window
            edges = np.diff((v_rs > 2.5).astype(int))
            falls = np.where(edges == -1)[0]
            rises = np.where(edges == 1)[0]
            
            if len(falls) > 0 and len(rises) > 0:
                f_idx = falls[0]
                r_pts = rises[rises > f_idx]
                if len(r_pts) > 0:
                    r_idx = r_pts[0]
                    
                    # 3. Robust Ramp Slicing
                    # We only fit data when the pixel is in its linear region (1.0V to 4.5V)
                    # This ignores the reset transient and the saturation floor.
                    ramp_v = v_px[f_idx:r_idx]
                    ramp_t = t[f_idx:r_idx]
                    
                    mask = (ramp_v > 1.0) & (ramp_v < 4.5)
                    
                    if np.sum(mask) > 10:
                        # Perform Linear Regression on the ramp
                        slope, intercept = np.polyfit(ramp_t[mask], ramp_v[mask], 1)
                        inferred_delta = abs(slope) * T_INT
                        is_sat = ramp_v[-1] < 1.0 # True if it hit floor before end
                    else:
                        # If the ramp is too fast to catch, use the last valid slope or skip
                        inferred_delta = np.nan
                        is_sat = True

                    results.append({
                        "i_ma": curr,
                        "delta_v": inferred_delta,
                        "is_saturated": is_sat
                    })
                    print(f"  I={curr:.2f}mA | ΔV={'[EXT] ' if is_sat else '      '}{inferred_delta:.3f}V", end="\r")

        # 4. Final Processing and Plotting
        res_df = pl.DataFrame(results).drop_nulls()
        res_df.write_csv("tts_v2_results.csv")
        
        plt.figure(figsize=(10, 6))
        plt.plot(res_df["i_ma"], res_df["delta_v"], 'go-', label='Inferred Voltage (Slope * 2ms)')
        
        # Add linearity fit
        m, b = np.polyfit(res_df["i_ma"], res_df["delta_v"], 1)
        plt.plot(res_df["i_ma"], m*res_df["i_ma"] + b, 'r--', alpha=0.5, label=f'Linear Fit R²={np.corrcoef(res_df["i_ma"], res_df["delta_v"])[0,1]**2:.4f}')
        
        plt.xlabel("Input Current (mA)")
        plt.ylabel("Inferred Integration Drop (V)")
        plt.title("Slope-Based Extended Dynamic Range Linearity (500Hz)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("tts_v2_linearity.png")
        print("\n--- Test Complete. Report: tts_v2_linearity.png ---")

if __name__ == "__main__":
    run_tts_v2()
