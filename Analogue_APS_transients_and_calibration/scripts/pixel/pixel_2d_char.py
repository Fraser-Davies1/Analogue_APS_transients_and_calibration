import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_2d_sweep():
    print("--- Starting 2D Pixel Characterisation ---")
    
    # 1. Load I-V data for Current Mapping
    iv_data = pl.read_csv("led_iv_high_res.csv")
    def get_current(v_psu):
        # Linearly interpolate mA from our previous characterisation
        return np.interp(v_psu, iv_data["v_in"], iv_data["i_ma"])

    # 2. Define Sweep Points
    # Integration times: 5s, 500ms, 50ms, 5ms, 0.5ms
    t_ints = [5.0, 0.5, 0.05, 0.005, 0.0005]
    freqs = [1.0/t for t in t_ints]
    
    # Intensity sweep (PSU Volts)
    v_steps = np.linspace(2.4, 4.0, 10) 

    with Bench.open("bench.yaml") as bench:
        # Hardware Init
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable()
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        master_results = []

        for t_target, f in zip(t_ints, freqs):
            print(f"\n--- Testing Integration Time: {t_target*1000:.1f} ms ({f:.2f} Hz) ---")
            
            # Adjust SigGen
            bench.siggen.channel(1).setup_square(frequency=f, amplitude=5.0, offset=2.5).enable()
            
            # Adjust Scope Timebase (Scale = T / 10 to see one full cycle)
            scope_scale = t_target / 5.0 # View 2 cycles
            bench.osc.set_time_axis(scale=scope_scale, position=t_target)
            time.sleep(1.0) # Settle timing

            for v in v_steps:
                current_ma = get_current(v)
                print(f"  Setting I_LED = {current_ma:.2f} mA...", end="\r")
                
                try:
                    bench.psu.channel(2).set(voltage=v).on()
                    # Long settle for slow integration
                    time.sleep(max(0.5, t_target * 1.2)) 
                    
                    data = bench.osc.read_channels([1, 3])
                    ch1, ch3 = data.values["Channel 1 (V)"].to_numpy(), data.values["Channel 3 (V)"].to_numpy()
                    
                    # Edge detection
                    edges = np.diff((ch3 > 2.5).astype(int))
                    falls = np.where(edges == -1)[0]
                    rises = np.where(edges == 1)[0]
                    
                    if len(falls) > 0 and len(rises) > 0:
                        t_start = falls[0]
                        stops = rises[rises > t_start]
                        if len(stops) > 0:
                            t_stop = stops[0]
                            v_start = np.mean(ch1[max(0, t_start-5):t_start])
                            v_end = np.mean(ch1[max(0, t_stop-5):t_stop])
                            
                            master_results.append({
                                "t_int_s": t_target,
                                "i_led_ma": current_ma,
                                "delta_v": v_start - v_end
                            })
                except Exception as e:
                    print(f"\n[ERROR] Step failed: {e}")
                    continue

        # 3. Final Report
        df = pl.DataFrame(master_results)
        df.write_csv("pixel_2d_results.csv")
        
        plt.figure(figsize=(10, 6))
        for t in t_ints:
            sub = df.filter(pl.col("t_int_s") == t)
            if not sub.is_empty():
                plt.plot(sub["i_led_ma"], sub["delta_v"], 'o-', label=f"t_int = {t*1000:.1f}ms")
        
        plt.xlabel("LED Current (mA)")
        plt.ylabel("Pixel Integration Drop (V)")
        plt.title("Pixel Linearity: Light Intensity vs. Integration Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("pixel_2d_char_plot.png")
        print("\n--- 2D Sweep Complete. Report saved to pixel_2d_char_plot.png ---")

if __name__ == "__main__":
    run_2d_sweep()
