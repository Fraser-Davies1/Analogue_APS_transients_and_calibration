import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_final_iv():
    df = pl.read_csv("led_iv_high_res.csv")
    
    # Filter out noise floor
    active_df = df.filter(pl.col("v_in") > 2.0)
    
    x = active_df["v_in"].to_numpy()
    y = active_df["i_ma"].to_numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b.', markersize=4, label='Data (100 points)')
    
    # Fit Ohmic region (2.6V - 5.0V)
    ohmic_df = active_df.filter(pl.col("v_in") > 2.6)
    xo, yo = ohmic_df["v_in"].to_numpy(), ohmic_df["i_ma"].to_numpy()
    m, b = np.polyfit(xo, yo, 1)
    r_sq = np.corrcoef(xo, yo)[0, 1]**2
    
    plt.plot(xo, m*xo + b, 'r--', alpha=0.8, label=f'Ohmic Fit (R²={r_sq:.5f})')
    
    # Annotate threshold
    plt.annotate('Exponential Knee', xy=(2.45, 1.0), xytext=(1.5, 15),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel("PSU Input Voltage (V)")
    plt.ylabel("LED Current (mA)")
    plt.title(f"High-Resolution LED Characterization\n(Corrected for 50 Ohm Parallel Load)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("led_iv_high_res_plot.png")
    print(f"Plot saved. Ohmic Region R²: {r_sq:.5f}")

if __name__ == "__main__":
    plot_final_iv()
