import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_final():
    df = pl.read_csv("v_to_v_linearity_v3.csv")
    df = df.filter(pl.col("v_drop") < 6.0) # Filter noise
    
    # Analyze the Active Region (LED ON)
    active_df = df.filter(pl.col("v_led") > 2.4)
    
    x = active_df["v_led"].to_numpy()
    y = active_df["v_drop"].to_numpy()
    
    m, b = np.polyfit(x, y, 1)
    r_sq = np.corrcoef(x, y)[0, 1]**2
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df["v_led"], df["v_drop"], label='All Data', alpha=0.3)
    plt.plot(x, y, 'bo', label='Active Region Data')
    plt.plot(x, m*x + b, 'r--', label=f'Linear Fit (R²={r_sq:.4f})')
    
    plt.xlabel("LED Input Voltage (V)")
    plt.ylabel("Pixel Delta V (V)")
    plt.title("Final Pixel Linearity: V_DD=5V, Reset=5V, 500Hz")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("final_pixel_linearity_5v.png")
    print(f"Final R²: {r_sq:.4f}")

if __name__ == "__main__":
    plot_final()
