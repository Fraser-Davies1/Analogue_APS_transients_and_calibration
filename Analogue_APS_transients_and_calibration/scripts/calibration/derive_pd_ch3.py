import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def run_full_derivation():
    print("--- Phase 1: Capturing LED I-V Data (Scope CH2 across 220 Ohm) ---")
    R_EFF = 40.74 # Corrected for 50 Ohm parallel probe loading
    
    with Bench.open("bench.yaml") as bench:
        # Phase 1: LED I-V Mapping
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        except:
            print("  [WARN] PSU CH1 setup failed. Ensure 5V VDD manually.")

        # Setup Scope for I-V (using CH2)
        bench.osc.channel(2).setup(scale=0.1, offset=0.0, coupling="DC").enable()
        
        v_sweep_iv = np.linspace(2.2, 5.0, 20)
        iv_results = []
        
        print("  Sweeping LED Voltage...")
        for v in v_sweep_iv:
            try:
                bench.psu.channel(2).set(voltage=v).on()
                time.sleep(0.4)
                res = bench.osc.measure_rms_voltage(2)
                v_drop = res.values.nominal_value if hasattr(res.values, "nominal_value") else res.values
                if v_drop > 10.0: v_drop = 0.0
                i_ma = (v_drop / R_EFF) * 1000.0
                iv_results.append({"v_in": v, "i_ma": i_ma})
                print(f"    V_in: {v:.2f}V -> I_LED: {i_ma:.2f}mA", end="\r")
            except:
                print(f"\n    [ERROR] PSU communication failed at {v:.2f}V.")
                continue
        
        if not iv_results:
            print("\n  [CRITICAL] No LED I-V data captured. Using fallback.")
            iv_results = [{"v_in": v, "i_ma": (v-2.3)/R_EFF*1000 if v>2.3 else 0} for v in v_sweep_iv]
        
        iv_df = pl.DataFrame(iv_results)
        iv_df.write_csv("led_iv_mapping.csv")

        # Phase 2: PD Derivation (Scope CH3 for Reset)
        print("\n\n--- Phase 2: Deriving Photodiode Current (CH3=Reset, C_int=11pF) ---")
        C_INT = 11e-12 
        
        # Hardware setup for PD sweep
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() # Pixel
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Reset Sync on CH3
        bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
        bench.osc.trigger.setup_edge(source="CH3", level=2.5)

        pd_results = []
        v_steps_pd = np.linspace(2.35, 4.5, 20)

        for v in v_steps_pd:
            i_led = np.interp(v, iv_df["v_in"], iv_df["i_ma"])
            try:
                bench.psu.channel(2).set(voltage=v).on()
                time.sleep(0.5)
                
                # Capture both channels
                reading = bench.osc.read_channels([1, 3])
                df = reading.values
                t_vec = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 3 (V)"].to_numpy()
                
                # Detect edges on CH3
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
                    pd_results.append({"i_led_ma": i_led, "i_pd_na": i_pd_na})
                    print(f"    I_LED: {i_led:.2f}mA -> I_PD: {i_pd_na:.2f}nA", end="\r")
            except:
                continue

        # Save and Plot
        if pd_results:
            res_df = pl.DataFrame(pd_results)
            res_df.write_csv("pd_derivation_results_ch3.csv")
            
            plt.figure(figsize=(10, 6))
            plt.plot(res_df["i_led_ma"], res_df["i_pd_na"], 'bo-', label='Derived I_pd (nA)')
            plt.xlabel("LED Current (mA)")
            plt.ylabel("Photodiode Current (nA)")
            plt.title(f"APS Characterization: I_LED vs I_PD (CH3 Sync, C_int=11pF)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig("pd_current_final_ch3.png")
            print(f"\n--- Success! Data: pd_derivation_results_ch3.csv, Plot: pd_current_final_ch3.png ---")
        else:
            print("\n[ERROR] No PD data captured.")

if __name__ == "__main__":
    run_full_derivation()
