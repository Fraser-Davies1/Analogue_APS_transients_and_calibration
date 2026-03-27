import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_v_to_v():
    df = pl.read_csv("v_to_v_linearity_v2.csv")
    # Clean data (remove out-of-range measurements)
    df = df.filter(pl.col("v_pixel_drop") < 5.0)
    
    x = df["v_led"].to_numpy()
    y = df["v_pixel_drop"].to_numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', markersize=8, label='Pixel Response (Gated Vpp)')
    
    # Linear fit
    m, b = np.polyfit(x, y, 1)
    r_sq = np.corrcoef(x, y)[0, 1]**2
    plt.plot(x, m*x + b, 'r--', label=f'Linear Fit (R²={r_sq:.4f})')
    
    plt.xlabel("LED Control Voltage (V)")
    plt.ylabel("Measured Pixel Drop (V)")
    plt.title("Active Pixel Circuit Linearity Verification (Fixed 500Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("v_to_v_linearity_plot.png")
    print(f"Plot saved. R-squared: {r_sq:.4f}")

if __name__ == "__main__":
    plot_v_to_v()
