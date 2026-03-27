import numpy as np
import polars as pl
from pytestlab import Bench
import time

def single_point_pd_measure():
    print("--- Single Point Photodiode Current Measurement ---")
    print("--- CH1=Pixel, CH3=Reset Ref, C_int=11pF ---")
    
    C_INT = 11e-12
    
    with Bench.open("bench.yaml") as bench:
        # Check if OSC and SigGen are alive
        try:
            # Sync timing
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            # Scope Setup
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() 
            bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            
            time.sleep(1)
            
            # Capture
            data = bench.osc.read_channels([1, 3])
            df = data.values
            t_vec = df["Time (s)"].to_numpy()
            v_px = df["Channel 1 (V)"].to_numpy()
            v_rs = df["Channel 3 (V)"].to_numpy()
            
            # Edge Analysis
            edges = np.diff((v_rs > 2.5).astype(int))
            falls = np.where(edges == -1)[0]
            rises = np.where(edges == 1)[0]
            
            if len(falls) > 0 and len(rises) > 0:
                idx_s = falls[0]
                idx_e = rises[rises > idx_s][0]
                
                t_int = t_vec[idx_e] - t_vec[idx_s]
                v_start = np.mean(v_px[max(0, idx_s-5):idx_s])
                v_end = v_px[idx_e]
                
                delta_v = v_start - v_end
                i_pd_na = (C_INT * (delta_v / t_int)) * 1e9
                
                print(f"\n--- Result ---")
                print(f"Integration Time: {t_int*1000:.3f} ms")
                print(f"Voltage Drop:     {delta_v:.3f} V")
                print(f"Derived I_pd:     {i_pd_na:.3f} nA")
                
            else:
                print("\n[ERROR] Integration window not found in capture.")
                
        except Exception as e:
            print(f"\n[ERROR] Measurement failed: {e}")

if __name__ == "__main__":
    single_point_pd_measure()
