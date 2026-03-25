import numpy as np
import polars as pl
from pytestlab import Bench
import time

def run_fast_test():
    print("--- Running High-Speed Sweep (5kHz) to avoid Saturation ---")
    with Bench.open("bench.yaml") as bench:
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            # 5kHz = 200us period (much shorter integration)
            bench.siggen.channel(1).setup_square(frequency=5000, amplitude=5.0, offset=2.5).enable()
            
            # OSC: 100us/div -> 1ms total window
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
            bench.osc.set_time_axis(scale=100e-6, position=500e-6)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
        except Exception as e:
            print(f"Setup failed: {e}")
            return

        results = []
        for v in np.linspace(2.3, 4.0, 15): # Focus on 2.3V - 4.0V
            bench.psu.channel(2).set(voltage=v, current_limit=0.05).on()
            time.sleep(0.3)
            
            data = bench.osc.read_channels([1, 3])
            ch1, ch3 = data.values["Channel 1 (V)"].to_numpy(), data.values["Channel 3 (V)"].to_numpy()
            
            edges = np.diff((ch3 > 2.5).astype(int))
            fall_pts, rise_pts = np.where(edges == -1)[0], np.where(edges == 1)[0]
            
            if len(fall_pts) > 0:
                t_start = fall_pts[0]
                stops = rise_pts[rise_pts > t_start]
                if len(stops) > 0:
                    t_stop = stops[0]
                    v_start, v_end = np.mean(ch1[t_start-5:t_start]), np.mean(ch1[t_stop-5:t_stop])
                    results.append({"v_led": v, "delta_v": v_start - v_end})

        df = pl.DataFrame(results)
        df.write_csv("fast_pixel_results.csv")
        r_sq = np.corrcoef(df["v_led"], df["delta_v"])[0, 1]**2
        print(f"--- Fast Sweep Complete ---")
        print(f"New R-squared: {r_sq:.5f}")

if __name__ == "__main__":
    run_fast_test()
