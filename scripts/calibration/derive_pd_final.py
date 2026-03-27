import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def derive_pd_current():
    print("--- Starting Photodiode Current Derivation (C_int = 11pF) ---")
    
    # 1. Load I-V Data
    iv_data = pl.read_csv("led_iv_high_res.csv")
    def get_i_led(v_psu):
        return np.interp(v_psu, iv_data["v_in"], iv_data["i_ma"])

    C_INT = 11e-12 # 11pF

    with Bench.open("bench.yaml") as bench:
        # Setup Rails and Reset (500Hz)
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        
        # Scope: CH1=Pixel, CH2=Reset Sync
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
        bench.osc.channel(2).setup(scale=1.0, offset=2.5).enable() 
        bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
        bench.osc.trigger.setup_edge(source="CH2", level=2.5)

        results = []
        v_steps = np.linspace(2.3, 4.5, 30)

        for v in v_steps:
            i_led = get_i_led(v)
            bench.psu.channel(2).set(voltage=v).on()
            time.sleep(0.5)
            
            data = bench.osc.read_channels([1, 2])
            df = data.values
            t_vec, v_px, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 2 (V)"].to_numpy()
            
            # Find edges on CH2
            edges = np.diff((v_rs > 2.5).astype(int))
            falls = np.where(edges == -1)[0]
            rises = np.where(edges == 1)[0]
            
            if len(falls) > 0 and len(rises) > 0:
                idx_s = falls[0]
                idx_e = rises[rises > idx_s][0]
                
                t_int = t_vec[idx_e] - t_vec[idx_s]
                v_start = np.mean(v_px[max(0, idx_s-5):idx_s])
                v_end = v_px[idx_e] # Direct sample at rising edge
                
                delta_v = v_start - v_end
                # Formula: I_pd = C * dV / dt
                i_pd_amps = C_INT * (delta_v / t_int)
                i_pd_na = i_pd_amps * 1e9

                results.append({
                    "i_led_ma": i_led,
                    "delta_v": delta_v,
                    "i_pd_na": i_pd_na,
                    "quantum_efficiency": (i_pd_amps / (i_led/1000.0)) if i_led > 0 else 0
                })
                print(f"  I_LED: {i_led:.2f}mA -> I_PD: {i_pd_na:.2f}nA", end="\r")

        # 2. Final Analysis and Plot
        res_df = pl.DataFrame(results)
        res_df.write_csv("pd_derivation_results.csv")
        
        plt.figure(figsize=(10, 6))
        plt.plot(res_df["i_led_ma"], res_df["i_pd_na"], 'go-', label='Derived I_pd (nA)')
        plt.xlabel("LED Current (mA)")
        plt.ylabel("Photodiode Current (nA)")
        plt.title(f"APS Photodiode Characterization (C_int = {C_INT*1e12}pF)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("pd_current_final.png")
        
        print("\n--- Derivation Complete. Plot: pd_current_final.png ---")

if __name__ == "__main__":
    derive_pd_current()
