import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_fine_2d_sweep():
    print("--- Starting Fine-Resolution 2D Pixel Characterisation ---")
    
    # 1. Load I-V data for High-Precision Current Mapping
    iv_data = pl.read_csv("led_iv_high_res.csv")
    def get_current(v_psu):
        return np.interp(v_psu, iv_data["v_in"], iv_data["i_ma"])

    # 2. Define Integration Times (5 steps across the requested range)
    t_ints = [1.0, 0.1, 0.01, 0.001, 0.0001] # 1s to 100us
    
    # Fine Current Sweep (PSU Voltage points)
    v_steps = np.linspace(2.2, 4.0, 40) # 40 points for high resolution

    with Bench.open("bench.yaml") as bench:
        # Hardware Setup
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable()
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        master_results = []

        for t in t_ints:
            f = 1.0 / t
            print(f"\n--- integration Time: {t*1000 if t<1 else t:.1f} {'ms' if t<1 else 's'} ({f:.2f} Hz) ---")
            
            # Sync timing
            bench.siggen.channel(1).setup_square(frequency=f, amplitude=5.0, offset=2.5).enable()
            
            # Adjust Scope: ensure we see exactly 2 cycles
            # Scale = (2 * Period) / 10 divisions
            scope_scale = (2 * t) / 10.0
            bench.osc.set_time_axis(scale=scope_scale, position=t)
            time.sleep(1.0) 

            for v in v_steps:
                curr = get_current(v)
                print(f"  I_LED = {curr:.2f} mA", end="\r")
                
                try:
                    bench.psu.channel(2).set(voltage=v).on()
                    # Wait for integration to complete (longer for slow times)
                    time.sleep(max(0.4, t * 1.5)) 
                    
                    data = bench.osc.read_channels([1, 3])
                    ch1, ch3 = data.values["Channel 1 (V)"].to_numpy(), data.values["Channel 3 (V)"].to_numpy()
                    
                    # Precision edge analysis
                    edges = np.diff((ch3 > 2.5).astype(int))
                    falls = np.where(edges == -1)[0]
                    rises = np.where(edges == 1)[0]
                    
                    if len(falls) > 0 and len(rises) > 0:
                        idx_f = falls[0]
                        idx_r = rises[rises > idx_f][0]
                        
                        v_start = np.mean(ch1[max(0, idx_f-5):idx_f])
                        v_end = np.mean(ch1[max(0, idx_r-5):idx_r])
                        
                        master_results.append({
                            "t_int_ms": t * 1000,
                            "i_led_ma": curr,
                            "delta_v": v_start - v_end
                        })
                except:
                    continue

        # 3. Save Data
        df = pl.DataFrame(master_results)
        df.write_csv("pixel_2d_fine_results.csv")
        
        # 4. Plotting (Output Voltage against Input Current)
        plt.figure(figsize=(10, 7))
        for t_ms in df["t_int_ms"].unique():
            subset = df.filter(pl.col("t_int_ms") == t_ms)
            plt.plot(subset["i_led_ma"], subset["delta_v"], 'o-', markersize=3, label=f"t_int = {t_ms:.2f}ms")
        
        plt.xlabel("LED Current (mA)")
        plt.ylabel("Pixel Output (Integration Drop [V])")
        plt.title("Fine-Resolution 2D Pixel Characterization")
        plt.legend(title="Integration Time")
        plt.grid(True, alpha=0.3)
        plt.savefig("fine_2d_pixel_char_plot.png")
        print("\n--- Sweep Complete. Data: pixel_2d_fine_results.csv, Plot: fine_2d_pixel_char_plot.png ---")

if __name__ == "__main__":
    run_fine_2d_sweep()
