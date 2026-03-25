import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_semilogx():
    print("--- Generating Semilog-X WDR Plot ---")
    try:
        df = pl.read_csv("tts_ultra_fine_results.csv")
        # Filter out noise and non-positive currents for log scale
        df = df.filter((pl.col("i_ma") > 0.01) & (pl.col("delta_v") >= 0))
        
        plt.figure(figsize=(10, 7))
        
        for t_ms in sorted(df["t_int_ms"].unique(), reverse=True):
            sub = df.filter(pl.col("t_int_ms") == t_ms)
            plt.semilogx(sub["i_ma"], sub["delta_v"], 'o-', markersize=3, label=f"T={t_ms:.1f}ms")
        
        plt.xlabel("LED Current (mA) [Log Scale]")
        plt.ylabel("Inferred Voltage Drop (V) [Linear Scale]")
        plt.title("WDR Pixel Linearity: Semilog-X Scale\n(Inferred Voltage vs. LED Current)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(title="Integration Time")
        plt.tight_layout()
        
        plt.savefig("tts_semilogx_report.png")
        print("Plot saved to tts_semilogx_report.png")
        
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    plot_semilogx()
