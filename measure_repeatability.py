
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pytestlab import Bench
import time
import os
from scipy import stats

def run_input_referred_repeatability_audit():
    output_dir = "repeatability"
    os.makedirs(output_dir, exist_ok=True)
    
    num_noise_samples = 500
    results = {}

    print("="*60)
    print("      INPUT-REFERRED SENSOR REPEATABILITY AUDIT       ")
    print("="*60)

    with Bench.open("bench.yaml") as bench:
        # --- PHASE 1: LINEARITY (CALIBRATION) ---
        input("\n[ACTION] Ensure Sensor is exposed to LIGHT stimulus. Press [ENTER] to start Linearity Sweep...")
        
        # Setup similar to pixel_automated_linearity.py
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
            bench.osc.channel(2).setup(scale=1.0, offset=2.5).enable() 
            bench.osc.set_time_axis(scale=1e-3, position=5e-3)
            bench.osc.trigger.setup_edge(source="CH2", level=2.5)
        except Exception as e:
            print(f"  [ERROR] Hardware initialization failed: {e}")
            return

        # Load Calibration Mapping
        try:
            lut = pl.read_csv("results/data/led_iv_mapping.csv")
        except:
            lut = pl.read_csv("led_iv_mapping.csv")

        # Perform Sweep
        v_led_sweep = np.linspace(2.3, 4.5, 10)
        lin_data = []
        
        print(f">>> Performing Linearity Sweep ({len(v_led_sweep)} steps)...")
        for v in v_led_sweep:
            bench.psu.channel(2).set(voltage=v, current_limit=0.05).on()
            time.sleep(0.5)
            
            data = bench.osc.read_channels([1, 2])
            df = data.values
            ch1 = df["Channel 1 (V)"].to_numpy()
            ch2 = df["Channel 2 (V)"].to_numpy()
            
            # Edge detection logic from automated_linearity
            # Add debugging to see if we are getting valid signals
            edges = np.diff((ch2 > 2.0).astype(int)) # Lowered threshold slightly for robustness
            fall_pts = np.where(edges == -1)[0]
            rise_pts = np.where(edges == 1)[0]
            
            if len(fall_pts) > 0 and len(rise_pts) > 0:
                # Use the last fall/rise pair in the buffer to ensure stability
                t_start = fall_pts[-1]
                # Find rising edges AFTER this falling edge
                stops = rise_pts[rise_pts > t_start]
                
                # If no rising edge follows, look for one BEFORE (cyclical)
                if len(stops) == 0:
                    t_stop = rise_pts[-1]
                    t_start = fall_pts[fall_pts < t_stop][-1]
                else:
                    t_stop = stops[0]

                # Ensure indices are valid
                if t_stop > t_start:
                    # Measurement "just before" edges (average samples)
                    v_start = np.mean(ch1[max(0, t_start-20):t_start])
                    v_end = np.mean(ch1[max(0, t_stop-20):t_stop])
                    delta_v = v_start - v_end
                    
                    # Get I_pd (estimated from LED current)
                    # Use a more robust filter
                    matching_rows = lut.filter(pl.col("v_in").ge(v))
                    if len(matching_rows) > 0:
                        i_led_ma = matching_rows["i_ma"][0]
                    else:
                        i_led_ma = lut["i_ma"][-1]
                    
                    i_pd_na = i_led_ma * 0.1 * 1e6 
                    
                    lin_data.append({"i_pd_na": i_pd_na, "delta_v": delta_v})
                    print(f"    V_LED={v:.2f}V -> DeltaV={delta_v:.3f}V (Start={v_start:.3f}V, End={v_end:.3f}V)")
            else:
                print(f"    [WARN] No edges detected on CH2 at V_LED={v:.2f}V. Check Reset Signal.")

        if not lin_data:
            print("[ERROR] Linearity sweep failed to capture data.")
            return

        lin_df = pl.DataFrame(lin_data)
        res_lin = stats.linregress(lin_df["i_pd_na"], lin_df["delta_v"])
        gain_v_na = res_lin.slope # Gain in Volts per nA (at 1s/500Hz = 2ms integration)
        
        print(f"  [OK] Derived Gain: {gain_v_na*1000:.4e} mV/nA")

        # --- PHASE 2: DARK REPEATABILITY (NOISE) ---
        input("\n[ACTION] Ensure Sensor is COVERED in Blu-tac (DARK). Press [ENTER] to start...")
        
        # High resolution noise capture
        bench.osc.channel(1).setup(scale=0.01, offset=2.5).enable() 
        time.sleep(1)

        noise_voltages = []
        print(f">>> Collecting {num_noise_samples} Noise Samples...")
        for i in range(num_noise_samples):
            if (i+1) % 50 == 0: print(f"  Progress: {i+1}/{num_noise_samples}")
            data = bench.osc.read_channels([1])
            pixel_v = data.values["Channel 1 (V)"].to_numpy()
            noise_voltages.append(np.mean(pixel_v[-20:]))

        results['dark'] = np.array(noise_voltages)

        # --- PHASE 3: LIGHT REPEATABILITY ---
        input("\n[ACTION] REMOVE Blu-tac from Sensor (LIGHT). Press [ENTER] to start...")
        
        # Reset scope to see larger signal
        bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
        # Set stimulus to a high-but-not-saturated level (e.g., 90% of max sweep)
        bench.psu.channel(2).set(voltage=v_led_sweep[-2]).on() 
        time.sleep(1)

        light_voltages = []
        print(f">>> Collecting {num_noise_samples} Light Samples...")
        for i in range(num_noise_samples):
            if (i+1) % 50 == 0: print(f"  Progress: {i+1}/{num_noise_samples}")
            data = bench.osc.read_channels([1])
            pixel_v = data.values["Channel 1 (V)"].to_numpy()
            light_voltages.append(np.mean(pixel_v[-20:]))

        results['light'] = np.array(light_voltages)

    # --- FINAL CALCULATIONS & REPORTING ---
    v_std_dark = np.std(results['dark'])
    i_noise_rms_pa = (v_std_dark / gain_v_na) * 1000 # Convert nA to pA
    
    # I_sat is the max current from the linearity sweep
    i_sat_na = max(lin_df["i_pd_na"])
    dr_db = 20 * np.log10(i_sat_na / (i_noise_rms_pa/1000))

    print("\n" + "="*60)
    print("             INPUT-REFERRED NOISE REPORT             ")
    print("="*60)
    print(f"  Voltage Noise (σ_v):    {v_std_dark*1000:10.3f} mV")
    print(f"  Equivalent Gain:        {gain_v_na*1000:10.4e} mV/nA")
    print(f"  Input Current Noise:    {i_noise_rms_pa:10.2f} pA rms")
    print(f"  Saturation Current:     {i_sat_na:10.2f} nA")
    print(f"  System Dynamic Range:   {dr_db:10.2f} dB")
    print("="*60)

    # Plotting
    plt.figure(figsize=(14, 10))
    
    # 1. Linearity
    plt.subplot(2, 2, 1)
    plt.plot(lin_df["i_pd_na"], lin_df["delta_v"], 'bo-')
    plt.title("Transimpedance Calibration")
    plt.xlabel("I_pd (nA)")
    plt.ylabel("Delta V (V)")
    plt.grid(True, alpha=0.3)

    # 2. Noise Time Series
    plt.subplot(2, 2, 2)
    plt.plot(results['dark'], color='#d62728', alpha=0.6)
    plt.title(rf"Dark Noise Time Series\n$\sigma_i = {i_noise_rms_pa:.1f}$ pA")
    plt.ylabel("Voltage (V)")

    # 3. Repeatability Traces
    plt.subplot(2, 2, 3)
    plt.plot(results['light'], color='#2ca02c', alpha=0.6, label="Light")
    plt.plot(results['dark'], color='#1f77b4', alpha=0.6, label="Dark")
    plt.title("Voltage Trace Comparison")
    plt.legend()

    # 4. Dynamic Range Log Plot
    plt.subplot(2, 2, 4)
    plt.bar([r'Noise ($\sigma_i$)', r'Saturation ($I_{sat}$)'], [i_noise_rms_pa, i_sat_na*1000], color=['red', 'green'])
    plt.yscale('log')
    plt.ylabel("Current (pA) - Log Scale")
    plt.title(f"Input-Referred DR: {dr_db:.2f} dB")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_referred_repeatability_report.png"))
    print(f"\nAudit Complete. Report saved to {output_dir}/input_referred_repeatability_report.png")

if __name__ == "__main__":
    run_input_referred_repeatability_audit()
