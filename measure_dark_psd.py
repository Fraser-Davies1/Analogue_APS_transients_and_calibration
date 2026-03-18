import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy import signal
from pytestlab import Bench
import time

def analyze_dark_psd():
    print("--- Starting Dark Noise & PSD Analysis ---")
    with Bench.open("bench.yaml") as bench:
        # Reset and clear
        bench.osc.reset()
        
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        bench.psu.channel(2).off()
        bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
        
        # Wide view first to confirm signal
        print("  Checking signal presence...")
        bench.osc.channel(1).setup(scale=1.0, offset=2.5).enable()
        bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable()
        bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
        bench.osc.trigger.setup_edge(source="CH3", level=2.5, slope="NEG")
        
        time.sleep(1.0)
        data = bench.osc.read_channels([1, 3])
        v1 = data.values["Channel 1 (V)"].to_numpy()
        print(f"  CH1 Mean: {np.mean(v1):.3f}V, Std: {np.std(v1)*1e3:.3f}mV")
        print(f"  CH1 Range: {np.min(v1):.3f}V to {np.max(v1):.3f}V")
        
        # Now zoom in if signal is stable
        print("  Zooming in for noise measurement...")
        # Focus on the flat part of the transient (near the end of the 1ms cycle)
        # 500Hz -> 2ms period. 1ms High, 1ms Low.
        # Let's look at 0.8ms after trigger.
        bench.osc.set_time_axis(scale=10e-6, position=800e-6)
        bench.osc.channel(1).setup(scale=0.01, offset=np.mean(v1)).enable()
        
        time.sleep(1.0)
        data = bench.osc.read_channels([1])
        v_noise = data.values["Channel 1 (V)"].to_numpy()
        t = data.values["Time (s)"].to_numpy()
        
        rms_noise = np.std(v_noise)
        print(f"  Final RMS Noise: {rms_noise*1e6:.2f} µV")
        print(f"  Unique Values: {len(np.unique(v_noise))}")

        f, psd = signal.welch(v_noise, 1.0/(t[1]-t[0]), nperseg=1024)
        
        plt.figure(figsize=(10, 8))
        plt.subplot(211)
        plt.plot(t*1e6, v_noise)
        plt.title(f"Dark Noise Transient (RMS: {rms_noise*1e6:.1f}µV)")
        plt.xlabel("Time (µs)"); plt.ylabel("Voltage (V)")
        
        plt.subplot(212)
        plt.semilogy(f, psd)
        plt.title("Noise PSD")
        plt.xlabel("Frequency (Hz)"); plt.ylabel("V^2/Hz")
        
        plt.tight_layout()
        plt.savefig("dark_noise_psd_report.png")
        print("Analysis complete.")

if __name__ == "__main__":
    analyze_dark_psd()
