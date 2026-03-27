import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pytestlab import Bench
import time

def calibrate_and_measure():
    print("--- Starting High-Resolution FFT Verification ---")
    
    with Bench.open("bench.yaml") as bench:
        # 1. Setup Stimulus (500Hz Reset Pulse)
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        
        # 2. Scope Setup (Wide Window to capture multiple 500Hz cycles)
        # 10ms/div = 100ms total window = 50 cycles of 500Hz.
        bench.osc.set_time_axis(scale=10e-3, position=50e-3)
        # Pixel on CH1 (sensitive scale)
        bench.osc.channel(1).setup(scale=0.01, offset=4.8).enable() 
        # Reset Ref on CH3 (full scale)
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable()
        
        print("  Acquiring 1M points for calibration...")
        # Note: read_channels calls digitize internally
        data = bench.osc.read_channels([1, 3])
        v_pixel = data.values["Channel 1 (V)"].to_numpy()
        v_reset = data.values["Channel 3 (V)"].to_numpy()
        t = data.values["Time (s)"].to_numpy()
        
        # Calculate Sampling Frequency from the actual time array
        dt = t[1] - t[0]
        fs = 1.0 / dt
        print(f"  Calculated Sample Rate: {fs/1e6:.4f} MHz")
        print(f"  Capture Duration: {t[-1] - t[0]:.4f} s")

        # --- PHASE 1: Verify X-Axis with CH3 ---
        # We use a very high nperseg to get a sharp peak
        f_res, psd_res = signal.welch(v_reset, fs, nperseg=131072, window='blackmanharris', detrend='linear')
        
        # Find the fundamental peak (ignoring DC)
        # We look for the maximum in a range around 500Hz
        mask = (f_res > 100) & (f_res < 1000)
        if np.any(mask):
            measured_spike = f_res[mask][np.argmax(psd_res[mask])]
            print(f"  VERIFICATION: Reset Signal Peak found at {measured_spike:.2f} Hz")
        else:
            measured_spike = 0
            print("  VERIFICATION: Could not find peak in 100-1000Hz range.")
        
        # --- PHASE 2: Perfect PSD for CH1 ---
        # Using Blackman-Harris and 75% overlap for the noise floor
        # nperseg=131072 provides RBW = fs / nperseg approx 10MHz / 131072 = 76 Hz
        f, psd = signal.welch(v_pixel, fs, 
                              window='blackmanharris', 
                              nperseg=131072, 
                              noverlap=98304, 
                              detrend='linear',
                              scaling='density')

        # 3. Plotting with dual verification
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Time Domain Zoom (First 5ms)
        zoom_pts = int(0.005 * fs)
        ax1.plot(t[:zoom_pts]*1e3, v_reset[:zoom_pts], label="Reset (CH3)", alpha=0.5)
        ax1.plot(t[:zoom_pts]*1e3, v_pixel[:zoom_pts], label="Pixel (CH1)")
        ax1.set_title("Time Domain Verification (First 5ms)")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Voltage (V)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Frequency Domain (PSD)
        ax2.loglog(f, np.sqrt(psd)*1e6, color='black', label="Pixel Noise Density")
        ax2.loglog(f_res, np.sqrt(psd_res)*1e6, color='red', alpha=0.4, label="Reset Spectrum (Ref)")
        ax2.set_title(f"Verified PSD (X-Axis Reference: {measured_spike:.1f}Hz Spike)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Noise Density (µV / √Hz)")
        ax2.grid(True, which="both", alpha=0.3)
        
        # Vertical indicators
        ax2.axvline(500, color='blue', linestyle=':', alpha=0.5, label="Target 500Hz")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("verified_noise_fft.png")
        print("\nPlot saved as verified_noise_fft.png")

if __name__ == "__main__":
    calibrate_and_measure()
