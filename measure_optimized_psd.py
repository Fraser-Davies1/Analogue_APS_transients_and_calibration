import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pytestlab import Bench
import time

def run_optimized_analysis(num_frames=10):
    print(f"--- Starting Optimized Noise FFT ({num_frames}M Points total) ---")
    
    with Bench.open("bench.yaml") as bench:
        # 1. Clear state and setup PSU
        bench.osc.reset()
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.psu.channel(2).off() # LED strictly OFF
        
        # 2. Maximize scope memory
        # DSOX1204G max is 1,000,000 pts
        try:
            bench.osc._send_command(":ACQuire:POINts:ANALog 1000000")
            print("  Scope memory depth set to 1,000,000 points.")
        except: pass
        
        # 3. Timebase Setup: 10ms/div -> 100ms total capture
        # position=50e-3 centers the 100ms window
        bench.osc.set_time_axis(scale=10e-3, position=50e-3)
        
        # 4. Vertical Sensitivity: Focus on the 4.8V plateau
        bench.osc.channel(1).setup(scale=0.01, offset=4.8).enable()
        
        psd_accumulator = []
        fs_actual = 0
        
        print(f"  Transferring {num_frames} frames of 1M points each...")
        for i in range(num_frames):
            start_t = time.time()
            
            # read_channels handles 'digitize' and transfer
            data = bench.osc.read_channels([1])
            v = data.values["Channel 1 (V)"].to_numpy()
            t = data.values["Time (s)"].to_numpy()
            fs_actual = 1.0 / (t[1] - t[0])
            
            # Compute PSD for this frame using a large segment for high resolution
            # nperseg=32768 gives approx fs/32768 frequency resolution
            f, psd = signal.welch(v, fs_actual, nperseg=32768, scaling='density')
            psd_accumulator.append(psd)
            
            elapsed = time.time() - start_t
            print(f"  Frame {i+1}/{num_frames} captured ({elapsed:.1f}s)")
        
        # Average the spectra to reduce variance
        avg_psd = np.mean(psd_accumulator, axis=0)
        
        # 5. Plotting
        plt.figure(figsize=(12, 7))
        # Plot in Noise Density (uV/sqrt(Hz))
        noise_density = np.sqrt(avg_psd) * 1e6
        plt.loglog(f, noise_density, color='black', linewidth=0.8)
        
        plt.title(f"Optimized Dark Noise Spectrum (11pF APS)\nRBW: {f[1]-f[0]:.1f}Hz | Points: {num_frames}M | Fs: {fs_actual/1e6:.1f}MHz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Noise Density (µV / √Hz)")
        plt.grid(True, which="both", alpha=0.3)
        
        # Reference markers
        plt.axvline(50, color='red', linestyle='--', alpha=0.5, label="Mains (50Hz)")
        if fs_actual > 1000:
             plt.axvline(500, color='blue', linestyle='--', alpha=0.5, label="Reset Sync (500Hz)")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig("optimized_noise_fft.png")
        print("\nAnalysis complete. Plot saved as optimized_noise_fft.png")

if __name__ == "__main__":
    run_optimized_analysis()
