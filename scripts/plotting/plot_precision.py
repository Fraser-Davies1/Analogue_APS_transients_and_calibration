import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_precision():
    df = pl.read_csv("precision_linearity_results.csv")
    # Clean data (drop NaNs if any edges were missed)
    df = df.drop_nulls()
    active_df = df.filter(pl.col("v_led") > 2.3)
    
    if active_df.is_empty():
        print("No active data found.")
        return

    x = active_df["v_led"].to_numpy()
    y = active_df["v_delta"].to_numpy()
    
    m, b = np.polyfit(x, y, 1)
    r_sq = np.corrcoef(x, y)[0, 1]**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'go', label='Measured Delta V (V_end - V_start)')
    plt.plot(x, m*x + b, 'k--', label=f'Linear Regression (R²={r_sq:.5f})')
    
    plt.xlabel("LED Input Voltage (V)")
    plt.ylabel("Measured Pixel Voltage Drop (V)")
    plt.title("Active Pixel Precision Linearity: VDD=5V, Reset=5V, 500Hz")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("precision_linearity_plot.png")
    print(f"Plot saved. Linearity R²: {r_sq:.5f}")

if __name__ == "__main__":
    plot_precision()
