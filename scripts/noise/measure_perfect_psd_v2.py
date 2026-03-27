import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from pytestlab import Bench
import time
import sys
import os

# --- Framework Patch: Register WaveformGeneratorConfig ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass
# ---------------------------------------------------------

def run_perfect_analysis(num_frames=5, output_name="psd_hardware.png", siggen_on=True):
    """
    Performs high-fidelity Noise PSD characterization on real hardware.
    Includes low-frequency support down to 10Hz.
    """
    state_str = "SIGGEN ON" if siggen_on else "SIGGEN OFF"
    print(f"--- Starting REAL Hardware Noise PSD ({state_str}) ---")
    
    with Bench.open("config/bench.yaml") as bench:
        # 1. Hardware Initialization
        # Explicit AC coupling, offset=0
        bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="AC").enable()
        
        # Verify Coupling
        actual_coupling = bench.osc._query(":CHAN1:COUP?").strip().upper()
        if "AC" not in actual_coupling:
            print(f"  [ERROR] Scope is in {actual_coupling} mode. AC expected.")
            sys.exit(1)

        # 2. SigGen Control
        if siggen_on:
            bench.siggen._send_command("SOUR1:FUNC SQU")
            bench.siggen._send_command("SOUR1:FREQ 500")
            bench.siggen._send_command("SOUR1:VOLT 5.0")
            bench.siggen._send_command("SOUR1:VOLT:OFFS 2.5")
            bench.siggen._send_command("OUTP1:STAT ON")
        else:
            bench.siggen._send_command("OUTP1:STAT OFF")
        
        # 3. Timebase for 10Hz resolution (200ms total window)
        bench.osc.set_time_axis(scale=20e-3, position=100e-3)
        
        psd_accumulator = []
        fs_actual = 0
        
        print(f"  Acquiring {num_frames} hardware frames (200ms each)...")
        for i in range(num_frames):
            bench.osc._send_command(":DIGitize CHANnel1")
            data = bench.osc.read_channels([1])
            v = data.values["Channel 1 (V)"].to_numpy()
            t = data.values["Time (s)"].to_numpy()
            fs_actual = 1.0 / (t[1] - t[0])
            
            # FFT Size for ~10Hz resolution
            n_fft = 524288
            if len(v) < n_fft: n_fft = len(v)
            
            f, psd = signal.welch(v, fs_actual, 
                                  window='blackmanharris',
                                  nperseg=n_fft, 
                                  noverlap=int(n_fft * 0.75),
                                  detrend=False, 
                                  scaling='density')
            
            psd_accumulator.append(psd)
            print(f"    Frame {i+1}/{num_frames} captured (df={f[1]-f[0]:.2f} Hz)")
        
        avg_psd = np.mean(psd_accumulator, axis=0)
        
        # Store data
        data_path = f"results/data/{output_name.replace('.png', '.npz')}"
        np.savez(data_path, f=f, psd=avg_psd)
        
        # 4. Plotting
        plt.figure(figsize=(12, 8))
        noise_density_nv = np.sqrt(avg_psd) * 1e9
        plt.loglog(f, noise_density_nv, color='black', linewidth=0.7)
        plt.title(f"Hardware Noise PSD: {state_str}\n10Hz support | 5M Points | AC Coupled")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Noise Density (nV / √Hz)")
        plt.xlim(left=5)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        
        plot_path = f"results/plots/{output_name}"
        plt.savefig(plot_path)
        
        rms_integrated = np.sqrt(integrate.trapezoid(avg_psd, f))
        print(f"\n--- Metrology Summary ({state_str}) ---")
        print(f"Integrated RMS Noise: {rms_integrated*1e3:.3f} mV")
        print(f"Report saved as {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Output filename")
    parser.add_argument("--off", action="store_true", help="Turn SigGen OFF")
    args = parser.parse_args()
    
    run_perfect_analysis(output_name=args.output, siggen_on=not args.off)
