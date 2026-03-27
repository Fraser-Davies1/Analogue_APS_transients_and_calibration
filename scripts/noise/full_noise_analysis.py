import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import scipy.constants as sc

# Constants for 11pF Analogue APS
C_int = 11e-12 
T = 300        
q = sc.e       
k_B = sc.k     

def simulate_pixel(I_pd, T_int, v_read_rms=0.5e-3, num_points=1000):
    t = np.linspace(0, T_int, num_points)
    
    # 1. Reset Noise (kTC)
    # sigma_v = sqrt(kT/C)
    v_kTC_rms = np.sqrt(k_B * T / C_int)
    v_reset = 3.3 + np.random.normal(0, v_kTC_rms)
    
    # 2. Signal (Integration)
    v_ideal = v_reset - (I_pd * t / C_int)
    
    # 3. Shot Noise (Voltage dependent)
    # Variance in electrons: N_e = I_pd * t / q
    # sigma_e = sqrt(N_e)
    # sigma_v = sigma_e * q / C_int = sqrt(I_pd * t / q) * q / C_int = sqrt(q * I_pd * t) / C_int
    v_shot_rms = np.sqrt(q * (I_pd * t) / C_int**2)
    v_shot = np.random.normal(0, v_shot_rms)
    
    # 4. Readout Noise
    v_readout = np.random.normal(0, v_read_rms, size=num_points)
    
    v_total = v_ideal + v_shot + v_readout
    return t, v_total, v_ideal

def perform_full_analysis():
    T_int = 1e-3  # 1ms
    i_pd_sweep = np.logspace(-12, -7, 50)  # 1pA to 100nA
    
    results = []
    
    for i_pd in i_pd_sweep:
        # Run multiple trials for each point to get statistics
        num_trials = 100
        trial_data = []
        for _ in range(num_trials):
            t, v_noisy, v_ideal = simulate_pixel(i_pd, T_int, num_points=1)
            # Final voltage drop (Signal)
            signal = v_ideal[0] - v_ideal[-1]
            # Actual noisy final value
            final_val = v_noisy[-1]
            trial_data.append(final_val)
        
        mean_v = np.mean(trial_data)
        var_v = np.var(trial_data)
        noise_rms = np.sqrt(var_v)
        
        # Signal is the drop from reset
        sig_drop = 3.3 - mean_v
        snr = 20 * np.log10(sig_drop / noise_rms) if sig_drop > 0 else 0
        
        results.append({
            "i_pd": i_pd,
            "signal_v": sig_drop,
            "noise_v_rms": noise_rms,
            "variance_v": var_v,
            "snr_db": snr
        })
    
    df = pl.DataFrame(results)
    
    # Create the 2x2 Plot Matrix
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. SNR vs Signal (The "Sensitivity Curve")
    axs[0, 0].semilogx(df["i_pd"], df["snr_db"], 'b-', linewidth=2)
    axs[0, 0].set_title("SNR vs Light Intensity (Photodiode Current)")
    axs[0, 0].set_xlabel("I_pd (A)")
    axs[0, 0].set_ylabel("SNR (dB)")
    axs[0, 0].grid(True, which="both", alpha=0.3)
    
    # 2. PTC (Photon Transfer Curve: Log Variance vs Log Mean)
    # This identifies Read Noise vs Shot Noise regions
    axs[0, 1].loglog(df["signal_v"], df["variance_v"], 'go-', markersize=4)
    # Plot slope 1 line for shot noise reference
    ref_x = np.array([1e-3, 1])
    # Variance = k * Mean in shot noise region
    # sigma^2 = (q/C) * Mean
    ref_y = (q/C_int) * ref_x
    axs[0, 1].loglog(ref_x, ref_y, 'k--', label="Ideal Shot Noise Slope (1)")
    axs[0, 1].set_title("Photon Transfer Curve (Variance vs Signal)")
    axs[0, 1].set_xlabel("Signal Voltage Drop (V)")
    axs[0, 1].set_ylabel("Voltage Variance (V^2)")
    axs[0, 1].legend()
    axs[0, 1].grid(True, which="both", alpha=0.3)
    
    # 3. Noise Floor Components
    axs[1, 0].loglog(df["signal_v"], df["noise_v_rms"] * 1e3, 'r-', label="Total Noise")
    # Read noise floor (constant)
    axs[1, 0].axhline(0.5, color='k', linestyle=':', label="Read Noise (500uV)")
    axs[1, 0].set_title("Noise Voltage RMS vs Signal")
    axs[1, 0].set_xlabel("Signal Voltage Drop (V)")
    axs[1, 0].set_ylabel("Noise RMS (mV)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, which="both", alpha=0.3)
    
    # 4. Final Transient Overlay for Visual Context
    t, v1, vi1 = simulate_pixel(1e-9, T_int)
    t, v2, vi2 = simulate_pixel(10e-9, T_int)
    axs[1, 1].plot(t*1e3, v1, label="1nA (Low Light)")
    axs[1, 1].plot(t*1e3, v2, label="10nA (High Light)")
    axs[1, 1].set_title("Transient Noise Visualization")
    axs[1, 1].set_xlabel("Time (ms)")
    axs[1, 1].set_ylabel("Voltage (V)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("full_noise_analysis_report.png")
    print("Full analysis complete. Report saved as full_noise_analysis_report.png")

if __name__ == "__main__":
    perform_full_analysis()
