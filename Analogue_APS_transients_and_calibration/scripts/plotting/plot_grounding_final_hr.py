import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def plot_grounding_final_hr():
    print("--- Generating High-Resolution Grounding Comparison ---")
    
    # Load stored datasets
    try:
        data_g = np.load("psd_low_freq_grounded.npz")
        data_u = np.load("psd_low_freq_ungrounded.npz")
    except FileNotFoundError as e:
        print(f"Error: Required result files not found: {e}")
        return

    f_g, psd_g = data_g['f'], data_g['psd']
    f_u, psd_u = data_u['f'], data_u['psd']
    
    plt.figure(figsize=(12, 8))
    
    # Plot Grounded (Shielded) in black
    plt.loglog(f_g, np.sqrt(psd_g)*1e9, color='black', linewidth=0.8, label="Grounded (Shielded Baseline)", alpha=0.9)
    
    # Plot Ungrounded (No Protection) in firebrick
    plt.loglog(f_u, np.sqrt(psd_u)*1e9, color='firebrick', alpha=0.7, linewidth=0.6, label="Ungrounded (No Protection)")
    
    plt.title("Analogue APS Grounding Comparison (High Resolution)\nNoise PSD: 1Hz to 250kHz | Faraday Cage Verification")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Noise Density (nV / √Hz)")
    plt.xlim(left=1) # Focus from 1Hz
    plt.grid(True, which="both", alpha=0.3)
    
    # Metrology
    rms_g = np.sqrt(integrate.trapezoid(psd_g, f_g))
    rms_u = np.sqrt(integrate.trapezoid(psd_u, f_u))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("noise_fft_final_comparison_hr.png")
    
    print(f"\n--- Comparative Metrology Results (1Hz Resolution) ---")
    print(f"Grounded Integrated Noise:   {rms_g*1e3:.3f} mV")
    print(f"Ungrounded Integrated Noise: {rms_u*1e3:.3f} mV")
    print(f"Change in Noise Level:        {((rms_u/rms_g)-1)*100:.2f} %")
    print("\nHigh-resolution comparison plot saved as noise_fft_final_comparison_hr.png")

if __name__ == "__main__":
    plot_grounding_final_hr()
