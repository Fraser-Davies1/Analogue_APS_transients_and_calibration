import numpy as np
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

def user_prompt(message):
    """Wait for explicit user confirmation in terminal."""
    print(f"\n{'='*70}")
    print(f"  [USER ACTION REQUIRED]")
    print(f"  >>> {message}")
    print(f"{'='*70}")
    
    while True:
        try:
            user_in = input("\nType 'YES' and press ENTER to proceed: ").strip().upper()
            if user_in == 'YES':
                print("Confirmed. Proceeding...\n")
                break
            else:
                print(f"Received '{user_in}'. Please type exactly 'YES' to continue.")
        except EOFError:
            time.sleep(1)
            continue

def capture_psd(bench, num_frames, siggen_on):
    """Captures raw hardware data and performs full-BW noise integration."""
    bench.osc.clear_status()
    
    if siggen_on:
        # 500Hz Square wave stimulus
        bench.siggen.set_function(1, "SQU")
        bench.siggen.set_frequency(1, 500)
        bench.siggen.set_amplitude(1, 5.0)
        bench.siggen.set_offset(1, 2.5)
        bench.siggen.set_output_state(1, "ON")
    else:
        bench.siggen.set_output_state(1, "OFF")
    
    time.sleep(2.0)
    
    # High-sensitivity AC setup
    bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="AC").enable()
    bench.osc.set_time_axis(scale=20e-3, position=100e-3)
    
    psd_accumulator = []
    f = None
    
    for i in range(num_frames):
        print(f"      - Capturing Frame {i+1}/{num_frames}...", end='\r')
        bench.osc._send_command(":DIGitize CHANnel1")
        data = bench.osc.read_channels([1])
        v = data.values["Channel 1 (V)"].to_numpy()
        t = data.values["Time (s)"].to_numpy()
        fs = 1.0 / (t[1] - t[0])
        n_fft = 524288 if len(v) >= 524288 else len(v)
        f, psd = signal.welch(v, fs, window='blackmanharris', nperseg=n_fft, scaling='density')
        psd_accumulator.append(psd)
    
    mean_psd = np.mean(psd_accumulator, axis=0)
    
    # 1. Full-Bandwidth Noise Integration
    total_noise_vrms = np.sqrt(integrate.trapezoid(mean_psd, f))
    
    # 2. Dynamic Range Calculation (Max signal = 5.0V rail)
    dynamic_range_db = 20 * np.log10(5.0 / total_noise_vrms)
    
    print(f"\n      - Total Noise: {total_noise_vrms*1e6:.2f} µVrms | DR: {dynamic_range_db:.1f} dB")
    
    return f, mean_psd, total_noise_vrms, dynamic_range_db

def run_noise_audit_sequence(bench, config_label):
    """Runs SigGen OFF and ON sequences for a given PD configuration."""
    print(f"\n>>> Running Noise Audit for: {config_label}")
    
    # SigGen OFF
    print(f"    -> {config_label} | SigGen OFF (Thermal/Environmental Floor)...")
    f, psd_off, rms_off, dr_off = capture_psd(bench, num_frames=5, siggen_on=False)
    
    # SigGen ON
    print(f"    -> {config_label} | SigGen ON (Active Stimulus Interference)...")
    f, psd_on, rms_on, dr_on = capture_psd(bench, num_frames=5, siggen_on=True)
    
    return {
        "OFF": {"f": f, "psd": psd_off, "rms": rms_off, "dr": dr_off},
        "ON":  {"f": f, "psd": psd_on,  "rms": rms_on,  "dr": dr_on}
    }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("====================================================")
    print("   DUAL-PD NOISE & SPECTRAL DENSITY AUDIT           ")
    print(f"   Run ID: {timestamp}                             ")
    print("====================================================\n")
    
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    plot_dir = os.path.join(project_root, "final_tests/shielding_noise_audit/results/plots")
    os.makedirs(plot_dir, exist_ok=True)
    os.chdir(project_root)
    
    all_results = {}

    try:
        with Bench.open("bench.yaml") as bench:
            # Global PSU Init
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen._send_command("OUTP1:LOAD INF")

            # --- CONFIG 1: SINGLE PHOTODIODE ---
            print("\n[PHASE 1] SINGLE PHOTODIODE CHARACTERIZATION")
            user_prompt("Setup SINGLE photodiode and ensure it is ENCLOSED.")
            all_results["1PD"] = run_noise_audit_sequence(bench, "1PD")

            # --- CONFIG 2: DUAL PHOTODIODE ---
            print("\n[PHASE 2] DUAL PHOTODIODE CHARACTERIZATION")
            user_prompt("ADD the SECOND photodiode in parallel and ensure BOTH are ENCLOSED.")
            all_results["2PD"] = run_noise_audit_sequence(bench, "2PD")

            bench.psu.channel(1).off()
            bench.siggen.set_output_state(1, "OFF")

        # --- COMPARATIVE PLOTTING ---
        print("\n>>> GENERATING MULTI-PANEL NOISE REPORT...")
        plt.close('all')
        fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True, sharey=True)
        fig.suptitle(f"Analogue APS: Single vs Dual PD Noise Characterization (Run: {timestamp})\nEnclosed sensor | Full BW Noise Integration", fontsize=18)
        
        plot_map = [
            ("1PD", "OFF", axes[0, 0], 'black',     "1PD Floor (SigGen OFF)"),
            ("1PD", "ON",  axes[0, 1], 'royalblue', "1PD Active (SigGen ON)"),
            ("2PD", "OFF", axes[1, 0], 'grey',      "2PD Floor (SigGen OFF)"),
            ("2PD", "ON",  axes[1, 1], 'firebrick', "2PD Active (SigGen ON)")
        ]
        
        for cfg, state, ax, color, title in plot_map:
            data = all_results[cfg][state]
            density = np.sqrt(data["psd"]) * 1e9 # nV/√Hz
            
            ax.loglog(data["f"], density, color=color, alpha=0.9, linewidth=0.8)
            
            # Annotate metrics
            stat_text = (f"Total Noise: {data['rms']*1e6:.1f} µVrms\n"
                         f"Dynamic Range: {data['dr']:.1f} dB")
            ax.text(0.05, 0.05, stat_text, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                    verticalalignment='bottom', fontweight='bold', fontsize=13)
            
            ax.set_title(title, fontweight='bold', fontsize=14)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim(10, 2.5e6)

        for ax in axes[1, :]: ax.set_xlabel("Frequency (Hz)", fontsize=12)
        for ax in axes[:, 0]: ax.set_ylabel("Noise Density (nV / √Hz)", fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        report_path = os.path.join(plot_dir, f"dual_pd_noise_report_{timestamp}.png")
        plt.savefig(report_path)
        plt.show()
        print(f"\n[DONE] Audit complete. Report: {report_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
