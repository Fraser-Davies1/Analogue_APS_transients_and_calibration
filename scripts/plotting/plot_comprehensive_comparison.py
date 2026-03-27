import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os

def plot_final_comparison():
    print("--- Generating 3-Way Hardware Grounding & SigGen Comparison ---")
    
    # Define file paths
    base_dir = "/home/coder/project/Analogue_APS_transients_and_calibration"
    data_dir = os.path.join(base_dir, "results/data")
    plot_dir = os.path.join(base_dir, "results/plots")
    
    # Files: 
    # 1. Grounded + SigGen ON (Reference)
    # 2. Ungrounded + SigGen ON (EMI check)
    # 3. Grounded/Ungrounded (Current) + SigGen OFF (Crosstalk check)
    files = {
        "Grounded (SigGen ON)": "grounded_ac_real.npz",
        "Ungrounded (SigGen ON)": "ungrounded_ac_real.npz",
        "Ungrounded (SigGen OFF)": "siggen_off_ac_real.npz"
    }
    
    plt.figure(figsize=(14, 9))
    colors = {"Grounded (SigGen ON)": "black", "Ungrounded (SigGen ON)": "firebrick", "Ungrounded (SigGen OFF)": "royalblue"}
    linewidths = {"Grounded (SigGen ON)": 1.0, "Ungrounded (SigGen ON)": 0.8, "Ungrounded (SigGen OFF)": 1.2}
    alphas = {"Grounded (SigGen ON)": 0.9, "Ungrounded (SigGen ON)": 0.6, "Ungrounded (SigGen OFF)": 0.8}

    print("\n--- Comparative Metrology Results ---")
    for label, filename in files.items():
        path = os.path.join(data_dir, filename)
        try:
            data = np.load(path)
            f, psd = data['f'], data['psd']
            
            # Integrated Noise
            rms = np.sqrt(integrate.trapezoid(psd, f))
            print(f"{label:25}: {rms*1e3:.3f} mV")
            
            # Plot
            plt.loglog(f, np.sqrt(psd)*1e9, color=colors[label], 
                       linewidth=linewidths[label], alpha=alphas[label], label=label)
            
        except FileNotFoundError:
            print(f"Warning: File {filename} not found. Skipping.")

    plt.title("Analogue APS Comprehensive Noise PSD Comparison\nHardware AC Coupled | Grounding & SigGen Interaction")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Noise Density (nV / √Hz)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(plot_dir, "noise_fft_comprehensive_comparison.png")
    plt.savefig(save_path)
    print(f"\nFinal comparison plot saved as {save_path}")

if __name__ == "__main__":
    plot_final_comparison()
