import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import time
import sys
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
    """Captures raw hardware data with sub-10Hz resolution."""
    if siggen_on:
        bench.siggen._send_command("SOUR1:FUNC SQU")
        bench.siggen._send_command("SOUR1:FREQ 500")
        bench.siggen._send_command("SOUR1:VOLT 5.0")
        bench.siggen._send_command("SOUR1:VOLT:OFFS 2.5")
        bench.siggen.set_output_state(1, "ON")
    else:
        bench.siggen.set_output_state(1, "OFF")
    
    bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="AC").enable()
    bench.osc.set_time_axis(scale=20e-3, position=100e-3)
    time.sleep(1.0)
    
    psd_accumulator = []
    for i in range(num_frames):
        bench.osc._send_command(":DIGitize CHANnel1")
        data = bench.osc.read_channels([1])
        v = data.values["Channel 1 (V)"].to_numpy()
        t = data.values["Time (s)"].to_numpy()
        fs = 1.0 / (t[1] - t[0])
        n_fft = 524288 if len(v) >= 524288 else len(v)
        f, psd = signal.welch(v, fs, window='blackmanharris', nperseg=n_fft, scaling='density')
        psd_accumulator.append(psd)
    
    return f, np.mean(psd_accumulator, axis=0)

def main():
    print("====================================================")
    print("   QUAD-GRAPH NOISE FLOOR ANALYSIS (ENCLOSED)       ")
    print("   Grounding vs. SigGen Harmonics | 10Hz-2.5MHz     ")
    print("====================================================\n")
    
    project_root = os.path.abspath("/home/coder/project/Analogue_APS_transients_and_calibration")
    plot_dir = os.path.join(project_root, "shielding_noise_audit/results/plots")
    data_dir = os.path.join(project_root, "shielding_noise_audit/results/data")
    
    os.chdir(project_root)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
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
                
                # Measure SigGen OFF
                print(f"    -> {phase['label']} | SigGen OFF...")
                f_off, psd_off = capture_psd(bench, num_frames=5, siggen_on=False)
                results[f"{phase['label']}_SigGen_OFF"] = {"f": f_off, "psd": psd_off}
                
                # Measure SigGen ON
                print(f"    -> {phase['label']} | SigGen ON...")
                f_on, psd_on = capture_psd(bench, num_frames=5, siggen_on=True)
                results[f"{phase['label']}_SigGen_ON"] = {"f": f_on, "psd": psd_on}

            bench.psu.channel(1).off()
            bench.siggen.set_output_state(1, "OFF")

        # --- QUAD-PANEL PLOTTING ---
        print("\n>>> GENERATING QUAD-PANEL SPECTRAL REPORT...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
        fig.suptitle("Analogue APS: Spectral Noise Floor Audit (Enclosed sensor)\nHigh-Resolution PSD comparison (Welch Method)", fontsize=16)
        
        plot_map = [
            ("Grounded_SigGen_OFF", axes[0, 0], 'black'),
            ("Grounded_SigGen_ON",  axes[0, 1], 'royalblue'),
            ("Ungrounded_SigGen_OFF", axes[1, 0], 'grey'),
            ("Ungrounded_SigGen_ON",  axes[1, 1], 'firebrick')
        ]
        
        for label, ax, color in plot_map:
            data = results[label]
            f = data["f"]
            density = np.sqrt(data["psd"]) * 1e9 # nV/√Hz
            
            ax.loglog(f, density, color=color, alpha=0.9, linewidth=0.8)
            ax.set_title(label.replace("_", " "), fontweight='bold')
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim(10, 2.5e6)
            
            # Save raw data
            np.savez(os.path.join(data_dir, f"{label}_psd.npz"), f=f, psd=data["psd"])

        # Label outer axes
        for ax in axes[1, :]: ax.set_xlabel("Frequency (Hz)")
        for ax in axes[:, 0]: ax.set_ylabel("Noise Density (nV / √Hz)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        report_path = os.path.join(plot_dir, "quad_noise_audit_report.png")
        plt.savefig(report_path)
        print(f"\n[DONE] Noise audit complete.")
        print(f"QUAD-PANEL REPORT SAVED TO: {report_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
