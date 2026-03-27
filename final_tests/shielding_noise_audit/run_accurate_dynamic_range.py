import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy import signal, integrate
import os
import time
import sys
from datetime import datetime
from pytestlab import Bench

# --- Framework Patch: Register WaveformGeneratorConfig ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass
# ---------------------------------------------------------

C_INT = 11.0e-12  # 11pF integration capacitance
V_SAT = 0.2       # Saturation floor (V)
V_RESET = 5.0     # Reset rail (V)

def user_prompt(message):
    """Blocks execution for hardware changes."""
    print(f"\n{'='*75}")
    print(f"  [USER ACTION REQUIRED]")
    print(f"  >>> {message}")
    print(f"{'='*75}")
    while True:
        try:
            user_in = input("\nType 'YES' and press ENTER when ready: ").strip().upper()
            if user_in == 'YES':
                print("Confirmed. Proceeding...\n")
                break
            print(f"Invalid input. Please type 'YES' to continue.")
        except EOFError: time.sleep(1); continue

def capture_voltage_noise_psd(bench, config_label):
    """Measures a single high-resolution voltage noise PSD at the pixel output."""
    print(f"    -> Measuring Noise Floor for {config_label}...")
    bench.osc.channel(1).setup(scale=0.005, offset=0, coupling="AC").enable()
    # 2s window for good low-freq resolution
    bench.osc.set_time_axis(scale=0.2, position=1.0)
    
    num_frames = 10
    psd_accumulator = []
    f = None
    
    for i in range(num_frames):
        print(f"      - Capturing Frame {i+1}/{num_frames}...", end='\r')
        bench.osc._send_command(":DIGitize CHANnel1")
        data = bench.osc.read_channels([1])
        v = data.values["Channel 1 (V)"].to_numpy()
        t = data.values["Time (s)"].to_numpy()
        fs = 1.0 / (t[1] - t[0])
        n_fft = min(len(v), 262144)
        f, psd = signal.welch(v, fs, window='blackmanharris', nperseg=n_fft, scaling='density')
        psd_accumulator.append(psd)
    
    return f, np.mean(psd_accumulator, axis=0)

def run_linearity_sweep(bench, label, integration_times):
    """
    Sweeps stimulus (2.7V - 3.2V, 0.01V steps) using the robust transient method.
    Derived from run_sensor_linearity_audit.py
    """
    v_led_steps = np.arange(2.7, 3.21, 0.01) 
    sweep_results = []

    for t_int in integration_times:
        freq = 1.0 / (2.0 * t_int)
        print(f"    - Linearity Sweep T_int: {t_int:.1f}s...")
        
        bench.osc.clear_status()
        total_window = float(np.round(4.0 * t_int, 4))
        time_div = float(np.round(total_window / 10.0, 5))
        trigger_pos = float(np.round(0.2 * total_window, 5))
        
        bench.osc._send_command(f":TIMebase:SCALe {time_div}")
        bench.osc._send_command(f":TIMebase:POSition {trigger_pos}")
        bench.siggen.set_frequency(1, float(np.round(freq, 4)))
        time.sleep(1.0)

        for v_led in v_led_steps:
            bench.psu.channel(2).set(voltage=float(np.round(v_led, 2))).on()
            time.sleep(0.2)
            
            bench.osc._send_command(":SINGle")
            time.sleep(total_window + 0.3)
            
            try:
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_raw, v_px, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 2 (V)"].to_numpy()
                
                edges = np.diff((v_rs > 2.5).astype(int))
                falls, rises = np.where(edges == -1)[0], np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    f_idx, r_idx = falls[0], rises[rises > falls[0]][0]
                    v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                    v_final = v_px[r_idx]
                    dt_actual = t_raw[r_idx] - t_raw[f_idx]
                    
                    # Core physics: I = C * dV/dt
                    i_pd = C_INT * (abs(v_start - v_final) / dt_actual)
                    
                    sweep_results.append({
                        "t_int": t_int,
                        "v_led": v_led,
                        "i_pd": i_pd,
                        "v_out_final": v_final
                    })
            except: continue

    return pl.DataFrame(sweep_results)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("====================================================")
    print("   ACCURATE DYNAMIC RANGE CHARACTERIZATION          ")
    print("   Method: Input-Referred Noise (G = dV/dI_pd)      ")
    print("====================================================\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    while project_root != "/" and not os.path.exists(os.path.join(project_root, "bench.yaml")):
        project_root = os.path.dirname(project_root)
    os.chdir(project_root)
    
    plot_dir = "final_tests/shielding_noise_audit/results/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    integration_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    configs = ["1PD", "2PD"]
    
    config_noise = {} # Stores noise PSD for 1PD and 2PD
    config_linearity = {} # Stores linearity frames

    try:
        with Bench.open("bench.yaml") as bench:
            # Set global timeouts for long acquisitions
            for inst in bench._instrument_instances.values():
                inst._backend.timeout_ms = 120000

            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen.set_output_state(1, "ON")

            for cfg in configs:
                print(f"\n\n### CHARACTERIZING CONFIGURATION: {cfg} ###")
                user_prompt(f"Set up {cfg} circuit. ENSURE photodiode is ENCLOSED (Dark).")
                
                # 1. Single Noise Capture per Configuration
                f, psd_v = capture_voltage_noise_psd(bench, cfg)
                config_noise[cfg] = (f, psd_v)
                
                user_prompt(f"Keep {cfg} setup, but UNCOVER photodiode (Ambient/Stimulus).")
                
                # 2. Linearity Sweep Batch
                config_linearity[cfg] = run_linearity_sweep(bench, cfg, integration_times)

            bench.psu.channel(1).off()
            bench.psu.channel(2).off()
            bench.siggen.set_output_state(1, "OFF")

        # --- POST-PROCESSING ---
        print("\n>>> ANALYZING RESULTS...")
        dr_results = {cfg: [] for cfg in configs}
        noise_plots_data = []

        for cfg in configs:
            f, psd_v = config_noise[cfg]
            lin_df = config_linearity[cfg]
            
            for tint in integration_times:
                subset = lin_df.filter(pl.col("t_int") == tint)
                i_vals, v_vals = subset["i_pd"].to_numpy(), subset["v_out_final"].to_numpy()
                
                # Extract Gradient (Transimpedance) in linear region
                valid = (v_vals > 0.5) & (v_vals < 4.5)
                if np.sum(valid) >= 2:
                    slope, _ = np.polyfit(i_vals[valid], v_vals[valid], 1)
                    transimpedance = abs(slope)
                else:
                    transimpedance = tint / C_INT
                
                # Input Refer Noise
                psd_i = psd_v / (transimpedance**2)
                # RMS Noise (Integrated In-Band)
                f_mask = (f > 0.1)
                i_noise_rms = np.sqrt(integrate.trapezoid(psd_i[f_mask], f[f_mask]))
                
                # Max Signal (Saturation limit)
                i_max = (V_RESET - V_SAT) * C_INT / tint
                dr_db = 20 * np.log10(i_max / i_noise_rms)
                
                dr_results[cfg].append({"tint": tint, "dr": dr_db})
                
                # Store one PSD sample (e.g. at 2s) for plotting
                if tint == 2.0:
                    noise_plots_data.append({"cfg": cfg, "f": f, "psd_i": psd_i, "rms": i_noise_rms})

        # --- PLOTTING ---
        print("\n>>> GENERATING FINAL AUDIT PLOTS...")
        
        # Plot 1: Input Referred Noise PSDs
        fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, p in enumerate(noise_plots_data):
            axes[idx].loglog(p["f"], np.sqrt(p["psd_i"])*1e15, color='darkblue' if p['cfg']=="1PD" else 'darkgreen')
            axes[idx].set_title(f"{p['cfg']} Input Referred Current Noise")
            axes[idx].set_ylabel("Density (fA/√Hz)")
            axes[idx].set_xlabel("Frequency (Hz)")
            axes[idx].grid(True, which="both", alpha=0.3)
            axes[idx].text(0.05, 0.05, f"Integrated Noise:\n{p['rms']*1e15:.2f} fA_rms", 
                           transform=axes[idx].transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        fig1.savefig(os.path.join(plot_dir, f"input_referred_noise_summary_{timestamp}.png"))

        # Plot 2: Combined Linearity
        plt.figure(figsize=(11, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(integration_times)))
        for cfg, style in [("1PD", "-"), ("2PD", "--")]:
            for idx, tint in enumerate(integration_times):
                subset = config_linearity[cfg].filter(pl.col("t_int") == tint)
                plt.plot(subset["i_pd"]*1e12, subset["v_out_final"], 
                         label=f"{cfg} {tint}s", color=colors[idx], linestyle=style, marker='.', markersize=3)
        plt.axhline(V_SAT, color='red', linestyle=':', label="Saturation")
        plt.title("System Linearity: V_out vs Physical I_pd")
        plt.xlabel("Photodiode Current (pA)")
        plt.ylabel("Output Voltage (V)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"master_linearity_audit_{timestamp}.png"))

        # Plot 3: DR Comparison
        plt.figure(figsize=(9, 6))
        for cfg in configs:
            t = [x["tint"] for x in dr_results[cfg]]
            dr = [x["dr"] for x in dr_results[cfg]]
            plt.plot(t, dr, 'o-', label=f"{cfg} Configuration", linewidth=2)
        plt.title("System Dynamic Range Performance")
        plt.xlabel("Integration Window (s)")
        plt.ylabel("Dynamic Range (dB)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, f"dr_performance_audit_{timestamp}.png"))

        plt.show()
        print(f"\n[DONE] Dynamic Range Audit Complete. Reports in: {plot_dir}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__": main()
