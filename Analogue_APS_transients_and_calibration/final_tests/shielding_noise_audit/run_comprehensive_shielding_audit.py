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
    print(f"\n[USER ACTION REQUIRED] >>> {message}")
    print("Type 'YES' to proceed: ", end="")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline().strip().upper()
        if line == 'YES':
            break
        print("Invalid input. Please type 'YES' when ready: ", end="")
        sys.stdout.flush()

def capture_psd(bench, num_frames, siggen_on):
    """Captures raw hardware data and performs full-BW noise integration."""
    bench.osc.clear_status()
    
    if siggen_on:
        bench.siggen._send_command("SOUR1:FUNC SQU")
        bench.siggen._send_command("SOUR1:FREQ 500")
        bench.siggen._send_command("SOUR1:VOLT 5.0")
        bench.siggen._send_command("SOUR1:VOLT:OFFS 2.5")
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
    # RMS = sqrt( integral of PSD over frequency )
    total_noise_vrms = np.sqrt(integrate.trapezoid(mean_psd, f))
    
    # 2. Dynamic Range Calculation (Max signal = 5.0V rail)
    # DR (dB) = 20 * log10( V_max / V_noise )
    dynamic_range_db = 20 * np.log10(5.0 / total_noise_vrms)
    
    print(f"\n      - Total Noise: {total_noise_vrms*1e6:.2f} µVrms | DR: {dynamic_range_db:.1f} dB")
    
    return f, mean_psd, total_noise_vrms, dynamic_range_db

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("====================================================")
    print("   QUAD-GRAPH NOISE & DYNAMIC RANGE AUDIT           ")
    print(f"   Run ID: {timestamp}                             ")
    print("====================================================\n")
    
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    plot_dir = os.path.join(project_root, "final_tests/shielding_noise_audit/results/plots")
    
    os.makedirs(plot_dir, exist_ok=True)
    os.chdir(project_root)
    
    results = {}
    grounding_phases = [
        {"msg": "Ensure Ground is CONNECTED", "label": "Grounded"},
        {"msg": "Ensure Ground is REMOVED (Floating)", "label": "Ungrounded"}
    ]

    try:
        with Bench.open("config/bench.yaml") as bench:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            
            for phase in grounding_phases:
                user_prompt(phase["msg"])
                
                # SigGen OFF
                print(f"    -> {phase['label']} | SigGen OFF...")
                f, p, rms, dr = capture_psd(bench, num_frames=5, siggen_on=False)
                results[f"{phase['label']}_SigGen_OFF"] = {"f": f, "psd": p, "rms": rms, "dr": dr}
                
                # SigGen ON
                print(f"    -> {phase['label']} | SigGen ON...")
                f, p, rms, dr = capture_psd(bench, num_frames=5, siggen_on=True)
                results[f"{phase['label']}_SigGen_ON"] = {"f": f, "psd": p, "rms": rms, "dr": dr}

            bench.psu.channel(1).off()
            bench.siggen.set_output_state(1, "OFF")

        # --- QUAD-PANEL PLOTTING ---
        print("\n>>> GENERATING FINAL SPECTRAL REPORT...")
        plt.close('all')
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
        fig.suptitle(f"Analogue APS: Integrated Noise & Dynamic Range Audit (Run: {timestamp})\nEnclosed sensor | Total BW Noise Integration (10Hz - Nyquist)", fontsize=16)
        
        plot_map = [
            ("Grounded_SigGen_OFF", axes[0, 0], 'black'),
            ("Grounded_SigGen_ON",  axes[0, 1], 'royalblue'),
            ("Ungrounded_SigGen_OFF", axes[1, 0], 'grey'),
            ("Ungrounded_SigGen_ON",  axes[1, 1], 'firebrick')
        ]
        
        for label, ax, color in plot_map:
            data = results[label]
            density = np.sqrt(data["psd"]) * 1e9 # nV/√Hz
            
            ax.loglog(data["f"], density, color=color, alpha=0.9, linewidth=0.8)
            
            # Annotate with Noise and DR metrics
            stat_text = (f"Total Noise: {data['rms']*1e6:.1f} µVrms\n"
                         f"Dynamic Range: {data['dr']:.1f} dB")
            ax.text(0.05, 0.05, stat_text, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                    verticalalignment='bottom', fontweight='bold', fontsize=12)
            
            ax.set_title(label.replace("_", " "), fontweight='bold')
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim(10, 2.5e6)

        for ax in axes[1, :]: ax.set_xlabel("Frequency (Hz)")
        for ax in axes[:, 0]: ax.set_ylabel("Noise Density (nV / √Hz)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        report_path = os.path.join(plot_dir, f"noise_dr_report_{timestamp}.png")
        plt.savefig(report_path)
        print(f"\n[DONE] Audit complete. Report: {report_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
