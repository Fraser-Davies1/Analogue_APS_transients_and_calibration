import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_final_iv():
    # 1. Load Data
    df = pl.read_csv("led_iv_final.csv")
    
    # 2. Filter for the active region (ignore zeros/noise)
    active_df = df.filter(pl.col("i_ma") > 0.1)
    
    if active_df.is_empty():
        print("No active current data to plot.")
        return

    x = active_df["v_in"].to_numpy()
    y = active_df["i_ma"].to_numpy()
    
    # 3. Linear Fit for the Ohmic Region
    m, b = np.polyfit(x, y, 1)
    r_sq = np.corrcoef(x, y)[0, 1]**2
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'bs', label='Measured LED Current')
    plt.plot(x, m*x + b, 'r--', label=f'Ohmic Fit (1/220 Ohm, R²={r_sq:.4f})')
    
    plt.xlabel("PSU Input Voltage (V)")
    plt.ylabel("Calculated LED Current (mA)")
    plt.title("LED I-V Characterization: Ohmic Regime\n(Measured via 220 Ohm Shunt)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("led_iv_final_plot.png")
    print(f"Plot saved to led_iv_final_plot.png. R²: {r_sq:.4f}")

if __name__ == "__main__":
    plot_final_iv()
