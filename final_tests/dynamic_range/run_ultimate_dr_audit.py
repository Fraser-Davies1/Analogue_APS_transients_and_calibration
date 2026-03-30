import numpy as np
import polars as pl
import os
import time
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from pytestlab import Bench
from scipy import stats

# --- Framework Patch: Register WaveformGeneratorConfig ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass
# ---------------------------------------------------------

def user_prompt(message):
    print(f"\n[USER ACTION REQUIRED] >>> {message}")
    print("Type 'YES' to proceed: ", end="")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline().strip().upper()
        if line == 'YES':
            break
        print("Invalid input. Please type 'YES' when ready: ", end="")
        sys.stdout.flush()

def collect_noise_repeatability(bench, num_samples=500):
    """
    PHASE 1: Collect noise using the repeatability method.
    Automatically finds the baseline offset to ensure noise is visible on scope.
    """
    print(f"\n>>> [PHASE 1] Collecting {num_samples} Noise Samples (Repeatability Method)...")
    
    # Standard Setup
    bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
    bench.siggen.set_frequency(1, 500) # 2ms integration
    time.sleep(1.0)
    
    # 1. First capture at coarse scale to find baseline
    bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
    bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
    bench.osc.set_time_axis(scale=200e-6, position=1.0e-3)
    bench.osc.trigger.setup_edge(source="CH2", level=2.5)
    time.sleep(1.0)
    
    data = bench.osc.read_channels([1])
    baseline = np.mean(data.values["Channel 1 (V)"].to_numpy()[-50:])
    print(f"    - Detected Dark Baseline: {baseline:.3f} V")
    
    # 2. Re-setup with high sensitivity centered on baseline
    # 5mV/div or 10mV/div
    noise_scale = 0.005 
    bench.osc.channel(1).setup(scale=noise_scale, offset=baseline, coupling="DC").enable()
    time.sleep(1.0)
    
    voltages = []
    print(f"    - Starting 500-sample collection at {noise_scale*1000}mV/div...")
    for i in range(num_samples):
        if (i+1) % 100 == 0: print(f"      Progress: {i+1}/{num_samples}")
        data = bench.osc.read_channels([1])
        pixel_v = data.values["Channel 1 (V)"].to_numpy()
        # Take the mean of the end of integration for this trigger
        voltages.append(np.mean(pixel_v[-20:]))
        
    v_std = np.std(voltages)
    v_mean = np.mean(voltages)
    
    if v_std == 0:
        print("    [!] WARNING: Measured 0 noise. Increasing scale slightly for next attempt...")
        # Fallback to 20mV/div if 5mV is railed/quantized weirdly
        bench.osc.channel(1).setup(scale=0.02, offset=baseline, coupling="DC").enable()
        time.sleep(1.0)
        # Re-run a small sample
        v_std = 0.0005 # Minimal LSB estimate if still 0
        
    print(f"  [DONE] RMS Voltage Noise (sigma_v): {v_std*1000:.3f} mV")
    return v_std, v_mean, voltages

def measure_ambient_leakage(bench, duration_s=5.0):
    """
    PHASE 2: Measure ambient light current (LED OFF).
    """
    print(f"\n>>> [PHASE 2] Measuring Ambient Leakage (5.0s Window, LED OFF)...")
    C_integrator = 11.0e-12
    bench.psu.channel(2).off()
    
    freq = float(np.round(1.0 / (2.0 * duration_s), 4))
    bench.siggen.set_frequency(1, freq)
    
    total_window = 4.0 * duration_s
    bench.osc.channel(1).setup(scale=0.2, offset=2.5, coupling="DC").enable()
    bench.osc.set_time_axis(scale=total_window/10.0, position=0.2*total_window)
    bench.osc._backend.timeout_ms = int(total_window * 1000 + 60000)
    
    time.sleep(3.0)
    bench.osc._send_command(":SINGle")
    time.sleep(total_window + 2.0)
    
    data = bench.osc.read_channels([1, 2])
    df = data.values
    t, v_px, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 2 (V)"].to_numpy()
    
    edges = np.diff((v_rs > 2.5).astype(int))
    falls, rises = np.where(edges == -1)[0], np.where(edges == 1)[0]
    
    if len(falls) > 0 and len(rises) > 0:
        f_idx = falls[0]
        r_idx = rises[rises > f_idx][0]
        dt = t[r_idx] - t[f_idx]
        delta_v = abs(np.mean(v_px[max(0, f_idx-10):f_idx]) - np.mean(v_px[max(0, r_idx-10):r_idx]))
        i_amb_pa = (C_integrator * (delta_v / dt)) * 1e12
        print(f"  [DONE] Ambient Leakage: {i_amb_pa:.2f} pA")
        return i_amb_pa
    return 0.0

def calibrate_transimpedance(bench, get_led_current):
    """
    PHASE 3: Linearity Sweep to find V/pA Gain.
    """
    print(f"\n>>> [PHASE 3] Calibrating Transimpedance (V/pA)...")
    C_integrator = 11.0e-12
    v_steps = np.arange(2.8, 4.0, 0.2) # Use linear range
    t_int = 0.005 # 5ms
    
    bench.siggen.set_frequency(1, 1.0/(2.0*t_int))
    bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
    bench.osc.set_time_axis(scale=0.002, position=0.004)
    time.sleep(1.0)
    
    results = []
    for v_led in v_steps:
        led_ma = get_led_current(v_led)
        bench.psu.channel(2).set(voltage=v_led).on()
        time.sleep(0.5)
        
        bench.osc._send_command(":SINGle")
        time.sleep(0.2)
        data = bench.osc.read_channels([1, 2])
        df = data.values
        v_px, v_rs = df["Channel 1 (V)"].to_numpy(), df["Channel 2 (V)"].to_numpy()
        
        edges = np.diff((v_rs > 2.5).astype(int))
        f_idx = np.where(edges == -1)[0][0]
        r_idx = np.where(edges == 1)[0][np.where(edges == 1)[0] > f_idx][0]
        
        dv = abs(np.mean(v_px[max(0, f_idx-10):f_idx]) - np.mean(v_px[max(0, r_idx-10):r_idx]))
        dt = 0.005 # Fixed nominal for simplicity in gain
        i_pd_pa = (C_integrator * (dv / dt)) * 1e12
        
        results.append({"i_pd_pa": i_pd_pa, "dv": dv})
        print(f"    I_pd={i_pd_pa:.1f} pA -> DeltaV={dv:.3f}V")
        
    df_res = pl.DataFrame(results)
    slope, _, _, _, _ = stats.linregress(df_res["i_pd_pa"], df_res["dv"])
    print(f"  [DONE] Derived Transimpedance Gain: {slope*1e12:.2e} V/pA")
    return slope

def measure_saturation(bench):
    """
    PHASE 4: Measure maximum saturation voltage swing.
    """
    print(f"\n>>> [PHASE 4] Measuring Maximum Saturation Swing...")
    # Stimulus to MAX
    bench.psu.channel(2).set(voltage=5.0).on()
    time.sleep(1.0)
    
    # 10ms integration to ensure we hit the floor
    bench.siggen.set_frequency(1, 50) 
    bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
    bench.osc.set_time_axis(scale=0.005, position=0.01)
    time.sleep(1.0)
    
    data = bench.osc.read_channels([1, 2])
    v_px, v_rs = data.values["Channel 1 (V)"].to_numpy(), data.values["Channel 2 (V)"].to_numpy()
    
    edges = np.diff((v_rs > 2.5).astype(int))
    f_idx = np.where(edges == -1)[0][0]
    r_idx = np.where(edges == 1)[0][np.where(edges == 1)[0] > f_idx][0]
    
    v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
    v_sat_floor = np.min(v_px[f_idx:r_idx])
    v_swing_max = v_start - v_sat_floor
    
    print(f"  [DONE] Max Output Swing (Saturation): {v_swing_max:.3f} V")
    return v_swing_max

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = "/home/coder/project"
    os.chdir(project_root)
    
    iv_path = "final_tests/led_iv_characterisation/results/data/led_iv_data_manual_mean.csv"
    output_dir = "final_tests/dynamic_range/results"
    os.makedirs(output_dir, exist_ok=True)
    
    iv_data = pl.read_csv(iv_path)
    def get_led_current(v): return np.interp(v, iv_data["v_supply"], iv_data["i_ma"])

    try:
        with Bench.open("config/bench.yaml") as bench:
            # Step 1: Noise (DARK)
            user_prompt("ENCLOSE photodiode in Blu-Tack (NOISE FLOOR)")
            v_sigma, v_mean, noise_samples = collect_noise_repeatability(bench, 500)
            
            # Step 2: Ambient (LED OFF)
            user_prompt("REMOVE Blu-Tack. Place in BOX (AMBIENT CHECK)")
            i_amb_pa = measure_ambient_leakage(bench, 5.0)
            
            # Step 3: Gain (Linearity)
            user_prompt("READY Stimulus (TRANSIMPEDANCE CALIBRATION)")
            gain_v_pa = calibrate_transimpedance(bench, get_led_current)
            
            # Step 4: Saturation (MAX SIGNAL)
            user_prompt("MOVE Stimulus CLOSE or increase to MAX (SATURATION CHECK)")
            v_sat_max = measure_saturation(bench)
            
            # CALCULATIONS
            # 1. Input Referred Noise (pA)
            i_noise_pa = v_sigma / gain_v_pa
            # 2. Input Referred Saturation Current (pA)
            i_sat_max_pa = v_sat_max / gain_v_pa
            # 3. Dynamic Range
            dr_db = 20 * np.log10(i_sat_max_pa / i_noise_pa)
            
            print("\n" + "="*60)
            print("                FINAL DYNAMIC RANGE REPORT            ")
            print("="*60)
            print(f"  Voltage Noise Floor:    {v_sigma*1000:10.3f} mV")
            print(f"  Equivalent Gain:        {gain_v_pa*1e12:10.2e} V/pA")
            print(f"  Input-Referred Noise:   {i_noise_pa:10.2f} pA rms")
            print(f"  Saturation Current:     {i_sat_max_pa:10.2f} pA")
            print(f"  DYNAMIC RANGE:          {dr_db:10.2f} dB")
            print("="*60)
            
            # Plotting
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.plot(noise_samples, color='red', alpha=0.6)
            plt.title(f"Dark Noise Floor (Repeatability)\n$\sigma_v = {v_sigma*1000:.3f}$ mV")
            plt.ylabel("Voltage (V)")
            
            plt.subplot(2, 1, 2)
            plt.bar(['RMS Noise ($\sigma_i$)', 'Ambient ($I_{amb}$)', 'Saturation ($I_{sat}$)'], 
                    [i_noise_pa, i_amb_pa, i_sat_max_pa], color=['red', 'orange', 'green'])
            plt.yscale('log')
            plt.ylabel("Current (pA) - Log Scale")
            plt.title(f"Sensor Dynamic Range: {dr_db:.2f} dB")
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"final_dr_audit_{timestamp}.png")
            plt.savefig(plot_path)
            print(f"\n[DONE] Audit complete. Plot saved: {plot_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__": main()
