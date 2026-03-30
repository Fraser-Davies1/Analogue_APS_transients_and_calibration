import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import os
import time
from scipy import stats
from pytestlab import Bench

def capture_integration_slope(bench, label, duration_s=1.0):
    """
    Captures a 1s integration ramp and returns the slope (dV/dt).
    """
    # Scope Setup: 100ms/div = 1s window
    bench.osc.channel(1).setup(scale=0.1, offset=0, coupling="DC").enable()
    bench.osc.set_time_axis(scale=duration_s/10.0, position=duration_s/2.0)
    
    time.sleep(0.5) 
    bench.osc._send_command(":DIGitize CHANnel1")
    data = bench.osc.read_channels([1])
    
    v = data.values["Channel 1 (V)"].to_numpy()
    t = data.values["Time (s)"].to_numpy()
    
    # Linear Regression to find dV/dt
    res = stats.linregress(t, v)
    return res.slope, res.rvalue**2

def collect_noise_samples(bench, num_samples=500):
    """
    Collects raw voltage noise samples from the sensor in DARK.
    """
    print(f"\n>>> Collecting {num_samples} Noise Samples (DARK)...")
    voltages = []
    
    # Standard Setup for Noise (Transient sampling)
    bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
    bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
    
    bench.osc.channel(1).setup(scale=0.01, offset=2.5).enable() # High vertical resolution
    bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable() # Reset Ref
    bench.osc.set_time_axis(scale=100e-6, position=1.0e-3) 
    bench.osc.trigger.setup_edge(source="CH3", level=2.5)
    
    time.sleep(1)

    for i in range(num_samples):
        if (i+1) % 50 == 0: print(f"  Progress: {i+1}/{num_samples}")
        data = bench.osc.read_channels([1])
        df = data.values
        pixel_v = df["Channel 1 (V)"].to_numpy()
        sample_point = np.mean(pixel_v[-20:]) # Average end of trace
        voltages.append(sample_point)
        
    return np.array(voltages)

def run_input_referred_audit():
    output_dir = "results/input_referred"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load LED to Photodiode Mapping
    # Format: v_in (V), i_ma (Current through LED)
    try:
        lut = pl.read_csv("results/data/led_iv_mapping.csv")
    except:
        lut = pl.read_csv("led_iv_mapping.csv")

    # We'll use a fixed set of test voltages from the mapping
    test_voltages = [2.4, 3.0, 3.5, 4.0, 4.5] 
    
    slopes_dvdt = []
    i_pd_list = []

    with Bench.open("bench.yaml") as bench:
        # 1. Linearity Sweep to find Transimpedance
        input("\n[ACTION] Ensure Sensor is exposed to LIGHT stimulus. Press [ENTER] to start Linearity Sweep...")

        for v_led in test_voltages:
            # Interpolate or find closest mapping for i_pd
            # For this audit, we assume i_pd is proportional to LED current (i_ma)
            # In your actual setup, i_pd (nA) might need a specific scale factor
            # Let's assume a dummy gain for now or use i_ma as a proxy
            i_led_ma = lut.filter(pl.col("v_in").ge(v_led)).limit(1)["i_ma"][0]
            
            # Assuming i_pd (nA) = i_led_ma * 0.1 (typical photodiode coupling factor)
            # Replace 0.1 with your actual coupling constant if known
            i_pd = i_led_ma * 0.1 * 1e-9 # A
            
            print(f"  Measuring Linearization @ LED={v_led:.2f}V (Est. I_pd={i_pd*1e9:.2f} nA)...")
            bench.psu.channel(2).set(voltage=v_led).on()
            time.sleep(1)
            
            dvdt, r2 = capture_integration_slope(bench, f"LIN_{v_led}V")
            slopes_dvdt.append(dvdt)
            i_pd_list.append(i_pd)
            print(f"    Slope: {dvdt:.4f} V/s (R²={r2:.4f})")

        # Calculate Equivalent Transimpedance (R_eq = dV_dt / I_pd)
        res_trans = stats.linregress(i_pd_list, slopes_dvdt)
        R_eq = res_trans.slope # V/s per Ampere
        print(f"\n>>> Derived Transimpedance Gain: {R_eq:.2e} V/s per Ampere")

        # 2. Dark Noise Measurement
        input("\n[ACTION] Cover Sensor in BLU-TAC (DARK). Press [ENTER] to measure Noise Floor...")
        noise_samples = collect_noise_samples(bench, 500)
        v_std = np.std(noise_samples)
        v_mean = np.mean(noise_samples)

        # 3. Input Referencing
        i_noise_rms = v_std / R_eq 
        
        # 4. Dynamic Range
        i_sat = max(i_pd_list) 
        dr_ratio = i_sat / i_noise_rms
        dr_db = 20 * np.log10(dr_ratio)

    # Report
    print("\n" + "="*60)
    print("             INPUT-REFERRED NOISE REPORT             ")
    print("="*60)
    print(f"  System Transimpedance:  {R_eq:10.2e} V/A (1s integration)")
    print(f"  Voltage Noise (σ_v):    {v_std*1000:10.3f} mV")
    print(f"  Current Noise (σ_i):    {i_noise_rms*1e12:10.2f} pA")
    print(f"  Saturation Current:     {i_sat*1e9:10.2f} nA")
    print(f"  Input-Referred DR:      {dr_db:10.2f} dB")
    print("="*60)

    # Plotting
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(np.array(i_pd_list)*1e9, slopes_dvdt, 'o-', label="Measured Slopes")
    plt.plot(np.array(i_pd_list)*1e9, res_trans.intercept + res_trans.slope * np.array(i_pd_list), 'r--', label=f"Fit: R_eq={R_eq:.2e}")
    plt.title("Sensor Transimpedance Calibration")
    plt.xlabel("Est. Photodiode Current (nA)")
    plt.ylabel("Integration Slope (V/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(noise_samples * 1000, bins=30, color='gray', alpha=0.7)
    plt.title(f"Dark Noise Floor\n$\sigma={v_std*1000:.2f}$ mV")
    plt.xlabel("Voltage (mV)")

    plt.subplot(2, 2, 4)
    plt.bar(['Noise ($\sigma_i$)', 'Saturation ($I_{sat}$)'], [i_noise_rms*1e12, i_sat*1e12], color=['red', 'green'])
    plt.yscale('log')
    plt.ylabel("Current (pA) - Log Scale")
    plt.title(f"Dynamic Range: {dr_db:.2f} dB")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_referred_audit.png"))
    print(f"\nAudit Complete. Report saved to {output_dir}/input_referred_audit.png")

if __name__ == "__main__":
    run_input_referred_audit()
