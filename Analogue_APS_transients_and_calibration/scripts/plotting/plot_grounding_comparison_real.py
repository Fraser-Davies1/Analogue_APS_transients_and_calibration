import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os

def plot_hardware_comparison():
    print("--- Generating Real Hardware Grounding Comparison ---")
    
    # Define file paths
    base_dir = "/home/coder/project/Analogue_APS_transients_and_calibration"
    data_dir = os.path.join(base_dir, "results/data")
    plot_dir = os.path.join(base_dir, "results/plots")
    
    data_g_path = os.path.join(data_dir, "grounded_ac_real.npz")
    data_u_path = os.path.join(data_dir, "ungrounded_ac_real.npz")
    
    # Load datasets
    try:
        data_g = np.load(data_g_path)
        data_u = np.load(data_u_path)
    except FileNotFoundError as e:
        print(f"Error: Required result files not found: {e}")
        return

    f_g, psd_g = data_g['f'], data_g['psd']
    f_u, psd_u = data_u['f'], data_u['psd']
    
    plt.figure(figsize=(12, 8))
    
    # Plot Grounded (Shielded Baseline)
    plt.loglog(f_g, np.sqrt(psd_g)*1e9, color='black', linewidth=1.0, label="Grounded (Shielded Baseline)", alpha=0.9)
    
    # Plot Ungrounded (No Protection)
    plt.loglog(f_u, np.sqrt(psd_u)*1e9, color='firebrick', alpha=0.7, linewidth=0.8, label="Ungrounded (No Protection)")
    
    plt.title("Analogue APS Grounding Comparison: REAL HARDWARE\nNoise PSD: 10M Samples | AC Coupled (No Detrend)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Noise Density (nV / √Hz)")
    plt.grid(True, which="both", alpha=0.3)
    
    # Metrology
    rms_g = np.sqrt(integrate.trapezoid(psd_g, f_g))
    rms_u = np.sqrt(integrate.trapezoid(psd_u, f_u))
    
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(plot_dir, "noise_fft_real_hardware_comparison.png")
    plt.savefig(save_path)
    
    print(f"\n--- Comparative Metrology Results (REAL HARDWARE) ---")
    print(f"Grounded Integrated Noise:   {rms_g*1e3:.3f} mV")
    print(f"Ungrounded Integrated Noise: {rms_u*1e3:.3f} mV")
    print(f"Change in Noise Level:        {((rms_u/rms_g)-1)*100:.2f} %")
    print(f"\nComparison plot saved as {save_path}")

if __name__ == "__main__":
    plot_hardware_comparison()
