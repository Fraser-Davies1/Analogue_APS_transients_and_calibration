import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

def simulate_pixel_noise():
    """
    Simulates noise sources for an Analogue APS pixel circuit.
    Assumptions:
    - Integration capacitance (C_int) = 11 pF
    - Temperature (T) = 300 K
    - VDD = 3.3 V
    - Integration time (T_int) = 1 ms
    """
    
    # 1. Physical Constants
    C_int = 11e-12  # 11pF
    T = 300         # Kelvin
    q = sc.e        # 1.602e-19 C
    k_B = sc.k      # 1.38e-23 J/K
    
    # 2. Simulation Parameters
    T_int = 1e-3    # 1ms integration
    num_points = 1000
    t = np.linspace(0, T_int, num_points)
    V_reset_ideal = 3.3
    
    # 3. Photodiode Currents (Light intensities)
    I_pd_list = [1e-9, 5e-9, 10e-9]  # 1nA, 5nA, 10nA
    
    plt.figure(figsize=(12, 8))
    
    for I_pd in I_pd_list:
        # --- Noise Components ---
        
        # A. kTC Noise (Reset Noise)
        # Added once at the start of integration
        v_kTC_rms = np.sqrt(k_B * T / C_int)
        v_reset = V_reset_ideal + np.random.normal(0, v_kTC_rms)
        
        # B. Ideal Signal (Linear discharge)
        v_ideal = v_reset - (I_pd * t / C_int)
        
        # C. Shot Noise (Poisson noise on the charge)
        # Variance in volts: sigma^2 = q * Delta_V / C
        delta_v = (I_pd * t) / C_int
        v_shot_rms = np.sqrt(q * delta_v / C_int)
        v_shot = np.random.normal(0, v_shot_rms)
        
        # D. Readout Noise (White Gaussian Noise from SF/Scope)
        v_readout_rms = 0.5e-3 # 500uV RMS assumption
        v_readout = np.random.normal(0, v_readout_rms, size=num_points)
        
        # Total Signal
        v_total = v_ideal + v_shot + v_readout
        
        plt.plot(t * 1e3, v_total, label=f"I_pd = {I_pd*1e9:.0f}nA (Noisy)")
        plt.plot(t * 1e3, v_ideal, '--', alpha=0.5, label=f"I_pd = {I_pd*1e9:.0f}nA (Ideal)")

    plt.title("Analogue APS Pixel Noise Simulation (11pF Integration Cap)")
    plt.xlabel("Integration Time (ms)")
    plt.ylabel("Output Voltage (V)")
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pixel_noise_simulation.png")
    print("Simulation complete. Plot saved as pixel_noise_simulation.png")

if __name__ == "__main__":
    simulate_pixel_noise()
