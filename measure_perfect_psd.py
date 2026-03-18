import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from pytestlab import Bench
import time

def run_perfect_analysis(num_frames=10):
    print(f"--- Starting Perfected Noise FFT (Blackman-Harris, 75% Overlap) ---")
    
    with Bench.open("bench.yaml") as bench:
        # 1. Hardware Optimization
        bench.osc.reset()
        
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.psu.channel(2).off() 
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        
        # Acquisition setup
        try:
            # Setting waveform points to max
            bench.osc._send_command(":WAVeform:POINts 1000000")
            bench.osc._send_command(":ACQuire:TYPE HRESolution")
            print("  OSC: High-Resolution and Max Points configured.")
        except: pass
        
        bench.osc.set_time_axis(scale=10e-3, position=50e-3)
        bench.osc.channel(1).setup(scale=0.01, offset=4.8).enable()
        
        psd_accumulator = []
        fs_actual = 0
        
        print(f"  Transferring {num_frames} frames of 1M points each...")
        for i in range(num_frames):
            data = bench.osc.read_channels([1])
            v = data.values["Channel 1 (V)"].to_numpy()
            t = data.values["Time (s)"].to_numpy()
            fs_actual = 1.0 / (t[1] - t[0])
            
            # --- Advanced PSD Calculation ---
            f, psd = signal.welch(v, fs_actual, 
                                  window='blackmanharris',
                                  nperseg=65536, 
                                  noverlap=49152,
                                  detrend='linear',
                                  scaling='density')
            
            psd_accumulator.append(psd)
            print(f"  Processed Frame {i+1}/{num_frames}")
        
        avg_psd = np.mean(psd_accumulator, axis=0)
        
        # 2. Results and Plotting
        plt.figure(figsize=(12, 8))
        noise_density_nv = np.sqrt(avg_psd) * 1e9
        plt.loglog(f, noise_density_nv, color='darkslategrey', linewidth=0.7)
        
        plt.title(f"Artifact-Free Dark Noise PSD (11pF APS)\nRBW: {f[1]-f[0]:.2f}Hz | Total Points: {num_frames}M")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Noise Density (nV / √Hz)")
        plt.grid(True, which="both", alpha=0.2)
        plt.axvline(50, color='red', alpha=0.3, linestyle='--', label="50Hz")
        plt.legend()
        plt.tight_layout()
        plt.savefig("perfect_noise_fft.png")
        
        # Statistical Export using scipy.integrate
        rms_integrated = np.sqrt(integrate.trapezoid(avg_psd, f))
        print(f"\n--- Metrology Summary ---")
        print(f"Integrated RMS Noise: {rms_integrated*1e3:.3f} mV")
        idx_1m = np.argmin(np.abs(f-1e6))
        print(f"Noise Floor @ 1MHz: {noise_density_nv[idx_1m]:.1f} nV/√Hz")
        print(f"Analysis complete. Report saved as perfect_noise_fft.png")

if __name__ == "__main__":
    run_perfect_analysis()
