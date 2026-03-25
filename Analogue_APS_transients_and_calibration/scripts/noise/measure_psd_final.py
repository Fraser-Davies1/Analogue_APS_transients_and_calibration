import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from pytestlab import Bench
import time

# --- Framework Patch: Register WaveformGeneratorConfig ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass
# ---------------------------------------------------------

def run_low_freq_analysis(num_frames=5, output_name="psd_low_freq_grounded.png"):
    """
    Captures Noise PSD with 1Hz resolution on real hardware.
    """
    print(f"--- Starting High-Resolution PSD (1Hz to 250kHz) ---")
    
    with Bench.open("config/bench.yaml") as bench:
        bench.osc.reset()
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.psu.channel(2).off() 
        
        bench.siggen._send_command("SOUR1:FUNC SQU")
        bench.siggen._send_command("SOUR1:FREQ 500")
        bench.siggen._send_command("SOUR1:VOLT 5.0")
        bench.siggen._send_command("SOUR1:VOLT:OFFS 2.5")
        bench.siggen._send_command("OUTP1:STAT ON")
        
        bench.osc.set_time_axis(scale=200e-3, position=1.0)
        
        # Target bias point
        target_offset = 4.75 
        bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="AC").enable()
        
        psd_accumulator = []
        
        print(f"  Acquiring {num_frames} frames (2.0s each)...")
        for i in range(num_frames):
            data = bench.osc.read_channels([1])
            v = data.values["Channel 1 (V)"].to_numpy()
            t = data.values["Time (s)"].to_numpy()
            fs = 1.0 / (t[1] - t[0])
            
            n_fft = 524288
            f, psd = signal.welch(v, fs, 
                                  window='blackmanharris',
                                  nperseg=n_fft, 
                                  noverlap=int(n_fft * 0.75),
                                  detrend='constant',
                                  scaling='density')
            
            psd_accumulator.append(psd)
            print(f"    Frame {i+1} acquired (fs={fs/1e3:.1f} kSa/s)")
        
        avg_psd = np.mean(psd_accumulator, axis=0)
        
        # Data paths
        data_path = f"results/data/{output_name.replace('.png', '.npz')}"
        plot_path = f"results/plots/{output_name}"
        
        np.savez(data_path, f=f, psd=avg_psd)
        
        plt.figure(figsize=(12, 8))
        noise_density_nv = np.sqrt(avg_psd) * 1e9
        plt.loglog(f, noise_density_nv, color='black', linewidth=0.7)
        plt.title(f"Hardware Noise PSD (11pF APS)\nFrames: {num_frames} | BW: {f[-1]/1e3:.1f}kHz | RBW: {f[1]-f[0]:.2f}Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Noise Density (nV / √Hz)")
        plt.xlim(left=1)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path)
        
        rms = np.sqrt(integrate.trapezoid(avg_psd, f))
        print(f"  Integrated RMS Noise: {rms*1e3:.3f} mV")
        print(f"  Finished. Report saved as {plot_path}")

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "psd_low_freq_grounded.png"
    run_low_freq_analysis(output_name=out)
