import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import os
import time
from scipy import stats
from pytestlab import Bench

def collect_noise_dark(bench, num_samples=500):
    """
    Step 1: Collect noise using repeatability method (Dark only).
    """
    print(f"\n>>> [STEP 1] Collecting {num_samples} Noise Samples (DARK)...")
    voltages = []
    
    # Setup for Noise
    bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
    bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
    
    # High resolution for noise
    bench.osc.channel(1).setup(scale=0.01, offset=2.5).enable() # Pixel Out
    bench.osc.channel(2).setup(scale=1.0, offset=2.5).enable() # Reset Ref (CH2)
    bench.osc.set_time_axis(scale=100e-6, position=1.0e-3) 
    bench.osc.trigger.setup_edge(source="CH2", level=2.5)
    
    time.sleep(1)

    for i in range(num_samples):
        if (i+1) % 100 == 0: print(f"  Progress: {i+1}/{num_samples}")
        data = bench.osc.read_channels([1])
        pixel_v = data.values["Channel 1 (V)"].to_numpy()
        # End of integration sampling
        voltages.append(np.mean(pixel_v[-20:]))
        
    v_std = np.std(voltages)
    print(f"  [DONE] Voltage Noise (sigma_v): {v_std*1000:.3f} mV")
    return v_std

def measure_ambient_light(bench, duration_s=1.0):
    """
    Step 2: Measure ambient light level in the box using integration ramp.
    """
    print(f"\n>>> [STEP 2] Measuring Ambient Light Level...")
    C_integrator = 11.0e-12 # 11pF
    
    # Setup for TTS-style ramp
    bench.osc.channel(1).setup(scale=0.05, offset=0, coupling="DC").enable()
    bench.osc.set_time_axis(scale=duration_s/10.0, position=duration_s/2.0)
    
    time.sleep(1.0)
    bench.osc._send_command(":DIGitize CHANnel1")
    data = bench.osc.read_channels([1])
    
    v = data.values["Channel 1 (V)"].to_numpy()
    t = data.values["Time (s)"].to_numpy()
    
    res = stats.linregress(t, v)
    i_ambient = res.slope * C_integrator
    print(f"  [DONE] Ambient Current: {i_ambient*1e9:.3f} nA (R²={res.rvalue**2:.4f})")
    return i_ambient

def run_linearity_transimpedance(bench):
    """
    Step 3: Conduct linearity check to find transimpedance and saturation.
    """
    print(f"\n>>> [STEP 3] Running Linearity Check (Transimpedance Calibration)...")
    
    # Re-setup for standard linearity (2ms integration window)
    bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
    bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
    bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable() 
    bench.osc.channel(2).setup(scale=1.0, offset=2.5).enable() 
    bench.osc.set_time_axis(scale=1e-3, position=5e-3)
    bench.osc.trigger.setup_edge(source="CH2", level=2.5)

    # Load calibration
    try:
        lut = pl.read_csv("results/data/pd_final_characterization.csv")
    except:
        lut = pl.read_csv("pd_final_characterization.csv")

    v_led_sweep = np.linspace(2.3, 4.5, 15)
    lin_data = []

    for v in v_led_sweep:
        bench.psu.channel(2).set(voltage=v, current_limit=0.05).on()
        time.sleep(0.5)
        
        data = bench.osc.read_channels([1, 2])
        df = data.values
        ch1 = df["Channel 1 (V)"].to_numpy()
        ch2 = df["Channel 2 (V)"].to_numpy()
        
        # Edge Detection
        edges = np.diff((ch2 > 2.0).astype(int))
        fall_pts = np.where(edges == -1)[0]
        rise_pts = np.where(edges == 1)[0]
        
        if len(fall_pts) > 0 and len(rise_pts) > 0:
            t_start = fall_pts[-1]
            stops = rise_pts[rise_pts > t_start]
            t_stop = stops[0] if len(stops) > 0 else rise_pts[-1]
            
            if t_stop > t_start:
                v_start = np.mean(ch1[max(0, t_start-20):t_start])
                v_end = np.mean(ch1[max(0, t_stop-20):t_stop])
                delta_v = v_start - v_end
                
                # Get I_pd from calibration
                match = lut.filter(pl.col("v_in").ge(v)).limit(1)
                i_pd_na = match["i_pd_na"][0] if len(match) > 0 else lut["i_pd_na"][-1]
                
                lin_data.append({"i_pd_na": i_pd_na, "delta_v": delta_v})
                print(f"    V_LED={v:.2f}V -> I_pd={i_pd_na:.2f}nA, DeltaV={delta_v:.3f}V")

    lin_df = pl.DataFrame(lin_data)
    
    # Calculate Transimpedance (V per nA)
    # Filter for linear region if necessary, here we take the full sweep
    res = stats.linregress(lin_df["i_pd_na"], lin_df["delta_v"])
    gain_v_na = res.slope 
    i_sat_na = lin_df["i_pd_na"].max()
    
    print(f"  [DONE] Transimpedance: {gain_v_na*1000:.2f} mV/nA")
    print(f"  [DONE] Max Saturation Current: {i_sat_na:.2f} nA")
    
    return gain_v_na, i_sat_na, lin_df

def run_system_audit():
    output_dir = "results/dynamic_range"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("             SENSOR DYNAMIC RANGE AUDIT              ")
    print("="*60)

    with Bench.open("bench.yaml") as bench:
        # Step 1: Noise (Repeatability Dark)
        input("\n[ACTION] Cover Sensor in BLU-TAC (DARK). Press [ENTER] to start Noise Measurement...")
        v_sigma = collect_noise_dark(bench, 500)

        # Step 2: Ambient Light
        input("\n[ACTION] REMOVE Blu-tac. Ensure sensor is in the test BOX. Press [ENTER] to measure Ambient Light...")
        i_ambient = measure_ambient_light(bench)

        # Step 3: Linearity & Gain
        input("\n[ACTION] Confirm stimulus is READY. Press [ENTER] to start Linearity Check...")
        gain_v_na, i_sat_na, lin_df = run_linearity_transimpedance(bench)

    # FINAL CALCULATIONS
    # Input referred noise floor
    i_noise_rms_na = v_sigma / gain_v_na
    
    # Dynamic Range calculation
    dr_ratio = i_sat_na / i_noise_rms_na
    dr_db = 20 * np.log10(dr_ratio)

    print("\n" + "="*60)
    print("                FINAL AUDIT REPORT                    ")
    print("="*60)
    print(f"  Voltage Noise Floor (σ_v): {v_sigma*1000:10.3f} mV")
    print(f"  Transimpedance Gain:       {gain_v_na*1000:10.2f} mV/nA")
    print(f"  Input Current Noise (σ_i): {i_noise_rms_na*1000:10.2f} pA rms")
    print(f"  Ambient Light Current:     {i_ambient*1e9:10.2f} nA")
    print(f"  Max Saturation Current:    {i_sat_na:10.2f} nA")
    print(f"  SYSTEM DYNAMIC RANGE:      {dr_db:10.2f} dB")
    print("="*60)

    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Panel 1: Linearity Fit
    plt.subplot(2, 2, 1)
    plt.plot(lin_df["i_pd_na"], lin_df["delta_v"], 'bo-', label="Measured")
    plt.title("Transimpedance Calibration")
    plt.xlabel("I_pd (nA)")
    plt.ylabel("Delta V (V)")
    plt.grid(True, alpha=0.3)

    # Panel 2: DR Bar Chart
    plt.subplot(2, 2, 2)
    plt.bar(['Noise Floor ($\sigma_i$)', 'Saturation ($I_{sat}$)'], [i_noise_rms_na, i_sat_na], color=['red', 'green'])
    plt.yscale('log')
    plt.ylabel("Current (nA) - Log Scale")
    plt.title(f"Dynamic Range: {dr_db:.2f} dB")

    # Panel 3: Stats Summary Table (Simplified in Plot)
    plt.subplot(2, 1, 2)
    plt.axis('off')
    stats_text = (
        f"Voltage Noise Floor: {v_sigma*1000:.3f} mV\n"
        f"Equivalent Transimpedance: {gain_v_na*1000:.2f} mV/nA\n"
        f"Input Referred Noise: {i_noise_rms_na*1000:.2f} pA rms\n"
        f"Ambient Light: {i_ambient*1e9:.3f} nA\n"
        f"Saturation Limit: {i_sat_na:.2f} nA\n"
        f"RESULTING DYNAMIC RANGE: {dr_db:.2f} dB"
    )
    plt.text(0.1, 0.5, stats_text, fontsize=14, family='monospace', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "system_dynamic_range_report.png")
    plt.savefig(plot_path)
    print(f"\nAudit Complete. Report saved to {plot_path}")

if __name__ == "__main__":
    run_system_audit()
