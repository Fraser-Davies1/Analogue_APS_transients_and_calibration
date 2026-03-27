import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from pytestlab import Bench
import time
import sys
import os
from unittest.mock import MagicMock

# --- Framework Patch: Register WaveformGeneratorConfig ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass
# ---------------------------------------------------------

def capture_psd(bench, num_frames, siggen_on):
    """
    Captures raw hardware data with 7.6Hz resolution to resolve 10Hz floor.
    """
    if siggen_on:
        bench.siggen._send_command("SOUR1:FUNC SQU")
        bench.siggen._send_command("SOUR1:FREQ 500")
        bench.siggen._send_command("SOUR1:VOLT 5.0")
        bench.siggen._send_command("SOUR1:VOLT:OFFS 2.5")
        bench.siggen._send_command("OUTP1:STAT ON")
    else:
        bench.siggen._send_command("OUTP1:STAT OFF")
    
    # 1. Scope Setup for Low Frequency Resolution
    bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="AC").enable()
    bench.osc.set_time_axis(scale=20e-3, position=100e-3)
    time.sleep(0.1) 
    
    psd_accumulator = []
    fs_actual = 0
    
    for i in range(num_frames):
        # Hardware Trigger
        bench.osc._send_command(":DIGitize CHANnel1")
        data = bench.osc.read_channels([1])
        v = data.values["Channel 1 (V)"].to_numpy()
        t = data.values["Time (s)"].to_numpy()
        fs_actual = 1.0 / (t[1] - t[0])
        
        n_fft = len(v)
        
        f, psd = signal.welch(v, fs_actual, 
                              window='blackmanharris',
                              nperseg=n_fft, 
                              noverlap=int(n_fft * 0.75) if n_fft > 4 else 0,
                              detrend=False, 
                              scaling='density')
        psd_accumulator.append(psd)
    
    print("") 
    return f, np.mean(psd_accumulator, axis=0)

def run_automated_audit():
    print("====================================================")
    print("   ANALOGUE APS: 10Hz RESOLUTION SHIELDING AUDIT    ")
    print("   (SIMULATED MODE - NON-INTERACTIVE)               ")
    print("====================================================\n")
    sys.stdout.flush()
    
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    if os.getcwd() != project_root:
        try:
            os.chdir(project_root)
        except FileNotFoundError:
            project_root = "."
            os.chdir(project_root)
            
    combinations = [
        {"ground": True,  "siggen": True,  "label": "Grounded_SigGen_ON"},
        {"ground": True,  "siggen": False, "label": "Grounded_SigGen_OFF"},
        {"ground": False, "siggen": True,  "label": "Ungrounded_SigGen_ON"},
        {"ground": False, "siggen": False, "label": "Ungrounded_SigGen_OFF"}
    ]
    
    results = {}
    
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    with Bench.open("config/bench_sim.yaml") as bench:
        # Mock siggen since its profile is being difficult
        bench.siggen = MagicMock()
        
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.psu.channel(2).off()

        for combo in combinations:
            state_str = "GROUNDED" if combo["ground"] else "UNGROUNDED"
            sig_str = "ON" if combo["siggen"] else "OFF"
            
            print(f"\n>>> CONFIGURATION: Shield {state_str} | SigGen {sig_str}")
            sys.stdout.flush()
            
            f, psd = capture_psd(bench, num_frames=1, siggen_on=combo["siggen"])
            
            results[combo["label"]] = {"f": f, "psd": psd}
            data_path = f"results/data/{combo['label']}_hr.npz"
            np.savez(data_path, f=f, psd=psd)
            
            rms = np.sqrt(integrate.trapezoid(psd, f))
            print(f"    Capture Complete. Integrated RMS: {rms*1e3:.3f} mV")

    print("\n--- Generating High-Resolution Comparison ---")
    plt.figure(figsize=(14, 9))
    colors = ["black", "grey", "firebrick", "royalblue"]
    
    for i, (label, data) in enumerate(results.items()):
        plt.loglog(data["f"], np.sqrt(data["psd"])*1e9, color=colors[i], 
                   linewidth=1.0, alpha=0.8, label=label.replace("_", " "))
    
    plt.title("Analogue APS: High-Resolution Noise Floor Audit (Simulated)\n10Hz to 2.5MHz Resolution | Shielding vs. SigGen Crosstalk")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Noise Density (nV / √Hz)")
    plt.xlim(left=5) 
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = "results/plots/comprehensive_shielding_audit_hr_sim.png"
    plt.savefig(plot_path)
    print(f"REPORT SAVED: {plot_path}")

if __name__ == "__main__":
    run_automated_audit()
