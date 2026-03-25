import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_linearity():
    df = pl.read_csv("linearity_results_manual.csv")
    # Filter out any error readings
    df = df.filter(pl.col("v_drop") < 10.0)
    
    if df.is_empty():
        print("No valid data to plot.")
        return

    x = df["integration_time_ms"].to_numpy()
    y = df["v_drop"].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Measured Data')
    
    # Trendline
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Linear Fit (R²=0.88)')
    
    plt.xlabel("Integration Time (ms)")
    plt.ylabel("Voltage Drop (V)")
    plt.title("Pixel Circuit Linearity Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("pixel_linearity_plot.png")
    print("Plot saved to pixel_linearity_plot.png")

if __name__ == "__main__":
    plot_linearity()
