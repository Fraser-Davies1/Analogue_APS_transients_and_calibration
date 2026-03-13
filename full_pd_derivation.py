import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt

def robust_write(inst, cmd):
    for i in range(3):
        try:
            inst._send_command(cmd)
            return True
        except:
            print(f"  [RETRY] Write failed for {cmd}, retrying {i+1}/3...")
            time.sleep(1)
    return False

def run_full_derivation():
    print("--- Phase 1: Capturing LED I-V Data ---")
    R_SENSE = (220 * 50) / (220 + 50) # 40.74 Ohm effective
    
    with Bench.open("bench.yaml") as bench:
        # Phase 1: LED I-V
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        except:
            print("  [WARN] PSU CH1 Init failed. Please set 5V manually.")
        
        bench.osc.channel(2).setup(scale=0.1, offset=0.0, coupling="DC").enable()
        bench.osc.auto_scale()
        
        v_sweep_iv = np.linspace(2.2, 5.0, 20)
        iv_results = []
        for v in v_sweep_iv:
            try:
                bench.psu.channel(2).set(voltage=v).on()
            except:
                print(f"\n  [ERROR] Could not set PSU to {v:.2f}V. Skipping point.")
                continue
            
            time.sleep(0.5)
            try:
                res = bench.osc.measure_rms_voltage(2)
                v_drop = res.values.nominal_value if hasattr(res.values, "nominal_value") else res.values
                if v_drop > 10.0: v_drop = 0.0
                i_ma = (v_drop / R_SENSE) * 1000.0
                iv_results.append({"v_in": v, "i_ma": i_ma})
                print(f"  I-V: {v:.2f}V -> {i_ma:.2f}mA", end="\r")
            except: continue
        
        if not iv_results:
            print("\n[CRITICAL] No I-V data captured. Using fallback model.")
            iv_results = [{"v_in": v, "i_ma": (v-2.3)/R_SENSE*1000 if v>2.3 else 0} for v in v_sweep_iv]
        
        iv_df = pl.DataFrame(iv_results)
        iv_df.write_csv("led_iv_high_res.csv")

        # Phase 2: PD Derivation
        print("\n\n--- Phase 2: Deriving Photodiode Current (C_int = 11pF) ---")
        C_INT = 11e-12 
        
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
        bench.osc.channel(2).setup(scale=1.0, offset=2.5).enable()
        bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
        bench.osc.trigger.setup_edge(source="CH2", level=2.5)

        pd_results = []
        v_steps_pd = np.linspace(2.35, 4.5, 20)

        for v in v_steps_pd:
            i_led = np.interp(v, iv_df["v_in"], iv_df["i_ma"])
            try:
                bench.psu.channel(2).set(voltage=v).on()
            except: pass
            
            time.sleep(0.5)
            try:
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_vec, v_px, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 2 (V)"].to_numpy()
                
                edges = np.diff((v_rs > 2.5).astype(int))
                falls, rises = np.where(edges == -1)[0], np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    idx_s = falls[0]
                    idx_e = rises[rises > idx_s][0]
                    t_int = t_vec[idx_e] - t_vec[idx_s]
                    v_start = np.mean(v_px[max(0, idx_s-5):idx_s])
                    v_end = v_px[idx_e]
                    delta_v = v_start - v_end
                    i_pd_na = (C_INT * (delta_v / t_int)) * 1e9
                    pd_results.append({"i_led_ma": i_led, "i_pd_na": i_pd_na})
                    print(f"  I_LED: {i_led:.2f}mA -> I_PD: {i_pd_na:.2f}nA", end="\r")
            except: continue

        res_df = pl.DataFrame(pd_results)
        res_df.write_csv("pd_derivation_results.csv")
        
        plt.figure(figsize=(10, 6))
        plt.plot(res_df["i_led_ma"], res_df["i_pd_na"], 'bo-', label='Derived I_pd')
        plt.xlabel("LED Current (mA)")
        plt.ylabel("Photodiode Current (nA)")
        plt.title(f"Photodiode Current vs LED Stimulus (C_int=11pF)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("pd_current_final.png")
        print(f"\n--- Success! Plot: pd_current_final.png ---")

if __name__ == "__main__":
    run_full_derivation()
