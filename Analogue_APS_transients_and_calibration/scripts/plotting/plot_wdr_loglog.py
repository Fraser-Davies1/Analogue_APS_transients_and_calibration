import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_loglog():
    print("--- Generating Log-Log WDR Plot ---")
    try:
        # 1. Load Data
        df = pl.read_csv("pixel_wdr_family_data.csv")
        
        # 2. Filter for strictly positive values for log-log scale
        # We also filter for v_out_inferred > 0.01 to avoid noise floor artifacts in log space
        df = df.filter((pl.col("i_led_ma") > 0.01) & (pl.col("v_out_inferred") > 0.01))
        
        plt.figure(figsize=(10, 7))
        
        # 3. Plot family of curves
        # Reverse sort to put longest integration times at the top of the legend
        for t_ms in sorted(df["t_int_ms"].unique(), reverse=True):
            sub = df.filter(pl.col("t_int_ms") == t_ms).sort("i_led_ma")
            plt.loglog(sub["i_led_ma"], sub["v_out_inferred"], 'o-', markersize=4, label=f"T={t_ms:.1f}ms")
        
        # 4. Formatting
        plt.xlabel("LED Current (mA) [Log Scale]")
        plt.ylabel("Inferred Voltage Drop (V) [Log Scale]")
        plt.title("WDR Pixel Response: Log-Log Scale\n(Power-Law Verification)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(title="Integration Time")
        plt.tight_layout()
        
        plt.savefig("pixel_wdr_loglog_family.png")
        print("Plot saved to pixel_wdr_loglog_family.png")
        
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    plot_loglog()
