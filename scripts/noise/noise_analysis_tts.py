import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import scipy.constants as sc

# Constants for 11pF Analogue APS
C_int = 11e-12 
T_K = 300        
q = sc.e       
k_B = sc.k     
V_DD = 5.0
V_SAT = 0.5  # Saturation floor (ground rail + buffer headroom)

def simulate_pixel_with_tts(I_pd, T_int, v_read_rms=1.0e-3):
    """
    Simulates pixel integration with physical saturation and TTS readout.
    """
    # 1. Reset Noise (kTC)
    v_kTC_rms = np.sqrt(k_B * T_K / C_int)
    v_reset = V_DD - 0.2 + np.random.normal(0, v_kTC_rms) # Start near 4.8V
    
    # 2. Integration Slope (V/s)
    # Signal current + Shot noise
    # We'll simulate a fine time grid to find the saturation point
    num_pts = 2000
    t = np.linspace(0, T_int, num_pts)
    
    # Ideal ramp
    v_ideal = v_reset - (I_pd * t / C_int)
    
    # Add cumulative shot noise component
    # sigma_v(t) = sqrt(q * I_pd * t) / C
    v_shot_noise = np.random.normal(0, 1) * np.sqrt(q * I_pd * t) / C_int
    
    # Readout noise (applied at sampling points)
    v_readout = np.random.normal(0, v_read_rms, size=num_pts)
    
    v_noisy = v_ideal + v_shot_noise + v_readout
    
    # 3. Physical Saturation (Clipped at V_SAT)
    v_phys = np.maximum(v_noisy, V_SAT)
    
    # 4. Readout Logic
    # Standard Integration: Sample at the end
    v_standard_final = v_phys[-1]
    delta_v_phys = v_reset - v_standard_final
    
    # TTS Logic: Find time to cross 1.0V (our comparator threshold)
    V_REF = 1.0
    trigger_indices = np.where(v_phys <= V_REF)[0]
    
    if len(trigger_indices) > 0:
        idx = trigger_indices[0]
        t_sat = t[idx]
        # Avoid division by zero
        if t_sat < 1e-9: t_sat = 1e-9
        # Reconstruct theoretical drop: Delta_V = (V_start - V_ref) * (T_total / t_sat)
        v_tts_inferred = (v_reset - V_REF) * (T_int / t_sat)
        is_saturated = True
    else:
        v_tts_inferred = v_reset - v_phys[-1]
        is_saturated = False
        
    return delta_v_phys, v_tts_inferred, is_saturated

def perform_tts_noise_analysis():
    print("--- Starting TTS Noise Simulation (WDR Verification) ---")
    T_int = 100e-3 # 100ms integration (Long enough to see saturation)
    
    # Sweep Photodiode Current from 1pA to 10nA
    i_pd_sweep = np.logspace(-12, -8, 40) 
    
    results = []
    
    for i_pd in i_pd_sweep:
        # Run multiple trials for statistics
        num_trials = 50
        trial_phys = []
        trial_tts = []
        for _ in range(num_trials):
            d_phys, d_tts, sat = simulate_pixel_with_tts(i_pd, T_int)
            trial_phys.append(d_phys)
            trial_tts.append(d_tts)
            
        results.append({
            "i_pd": i_pd,
            "v_phys_mean": np.mean(trial_phys),
            "v_tts_mean": np.mean(trial_tts),
            "v_tts_std": np.std(trial_tts),
            "is_saturated": np.mean(trial_phys) > (V_DD - V_SAT - 0.1)
        })

    df = pl.DataFrame(results)
    
    # --- Visualization ---
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Linearity Comparison
    plt.loglog(df["i_pd"], df["v_phys_mean"], 'ro-', label='Physical Output (Saturating)', alpha=0.5)
    plt.loglog(df["i_pd"], df["v_tts_mean"], 'b*-', label='TTS Inferred Output (Linearized)')
    
    # Ideal Line
    plt.loglog(df["i_pd"], (df["i_pd"] * T_int / C_int), 'k--', label='Theoretical Ideal', alpha=0.8)

    plt.xlabel("Photodiode Current (A)")
    plt.ylabel("Output Voltage Drop (V)")
    plt.title(f"Simulation: Dynamic Range Extension via TTS\nIntegration Time = {T_int*1000:.0f}ms | C_int = 11pF")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    # Annotate Saturation
    plt.axhline(V_DD - V_SAT, color='grey', linestyle=':', label="Physical Rail")
    
    plt.savefig("noise_tts_simulation_report.png")
    print("Simulation complete. Report saved: noise_tts_simulation_report.png")

if __name__ == "__main__":
    perform_tts_noise_analysis()
