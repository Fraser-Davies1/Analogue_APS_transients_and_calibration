import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_loglog_pd():
    print("--- Generating Log-Log WDR Plot (Photodiode Current) ---")
    try:
        # 1. Load Data
        df = pl.read_csv("pixel_wdr_family_data.csv")
        
        # 2. Convert LED current to Photodiode current
        # Using established coupling factor: 0.15 nA (PD) per 1.0 mA (LED)
        df = df.with_columns([
            (pl.col("i_led_ma") * 0.15).alias("i_pd_na")
        ])
        
        # 3. Filter for strictly positive values for log-log scale
        df = df.filter((pl.col("i_pd_na") > 0.001) & (pl.col("v_out_inferred") > 0.01))
        
        plt.figure(figsize=(10, 7))
        
        # 4. Plot family of curves
        for t_ms in sorted(df["t_int_ms"].unique(), reverse=True):
            sub = df.filter(pl.col("t_int_ms") == t_ms).sort("i_pd_na")
            plt.loglog(sub["i_pd_na"], sub["v_out_inferred"], 'o-', markersize=4, label=f"T={t_ms:.1f}ms")
        
        # 5. Formatting
        plt.xlabel("Photodiode Current (nA) [Log Scale]")
        plt.ylabel("Inferred Voltage Drop (V) [Log Scale]")
        plt.title("WDR APS Response: Inferred Voltage vs. Photodiode Current")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(title="Integration Time")
        plt.tight_layout()
        
        plt.savefig("pixel_wdr_loglog_pd.png")
        print("Plot saved to pixel_wdr_loglog_pd.png")
        
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    plot_loglog_pd()
