import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def derive_currents():
    print("--- Deriving Photodiode Current (I_pd) ---")
    
    # 1. Load data
    try:
        df = pl.read_csv("tts_ultra_fine_results.csv")
    except:
        print("Data not found. Run pixel_tts_ultra_fine.py first.")
        return

    # 2. Constants
    # Adjust C_int to match your physical capacitor (e.g., 10e-12 for 10pF)
    C_INT = 10e-12 
    
    # Calculate I_pd [Amperes] = C * (dV / dt)
    # t_int is in ms, delta_v is in V
    df = df.with_columns([
        (C_INT * pl.col("delta_v") / (pl.col("t_int_ms") / 1000.0)).alias("i_pd_amps")
    ])
    
    # Convert to nanoAmps for readability
    df = df.with_columns([
        (pl.col("i_pd_amps") * 1e9).alias("i_pd_na")
    ])

    # 3. Calculate Coupling Efficiency (nA of PD current per mA of LED current)
    # We use the linear region (> 1mA)
    linear_df = df.filter(pl.col("i_ma") > 1.0)
    if not linear_df.is_empty():
        x = linear_df["i_ma"].to_numpy()
        y = linear_df["i_pd_na"].to_numpy()
        k_eff, intercept = np.polyfit(x, y, 1)
        print(f"System Coupling Efficiency: {k_eff:.4f} nA_pd / mA_led")

    # 4. Save and Plot
    df.write_csv("photodiode_current_results.csv")
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["i_ma"], df["i_pd_na"], 'bo-', markersize=2, label='Derived I_pd')
    plt.xlabel("LED Input Current (mA)")
    plt.ylabel("Derived Photodiode Current (nA)")
    plt.title(f"Photodiode Current vs LED Stimulus\n(Assumed C_int = {C_INT*1e12:.1f} pF)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("pd_current_derivation.png")
    
    print("\n--- Derivation Complete ---")
    print("Results: photodiode_current_results.csv")
    print("Plot: pd_current_derivation.png")

if __name__ == "__main__":
    derive_currents()
