import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_dual_tts():
    df = pl.read_csv("hardware_tts_results.csv")
    df = df.filter((pl.col("delta_v") > 0) & (pl.col("delta_v") < 5000)) # Clean outliers
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Linear Scale Plot
    for t_ms in sorted(df["t_int_ms"].unique()):
        sub = df.filter(pl.col("t_int_ms") == t_ms)
        ax1.plot(sub["i_led_ma"], sub["delta_v"], 'o-', markersize=4, label=f"T={t_ms:.1f}ms")
    
    ax1.set_xlabel("LED Current (mA)")
    ax1.set_ylabel("Inferred Voltage (V)")
    ax1.set_title("TTS Linearity: Linear Scale")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Log-Log Scale Plot
    for t_ms in sorted(df["t_int_ms"].unique()):
        sub = df.filter(pl.col("t_int_ms") == t_ms)
        ax2.loglog(sub["i_led_ma"], sub["delta_v"], 'o-', markersize=4, label=f"T={t_ms:.1f}ms")
    
    ax2.set_xlabel("LED Current (mA)")
    ax2.set_ylabel("Inferred Voltage (V)")
    ax2.set_title("TTS Linearity: Log-Log Scale")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.suptitle("Hardware-Emulated Time-to-Saturation (TTS) Dynamic Range Extension", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("tts_dual_report.png")
    print("Dual plot saved to tts_dual_report.png")

if __name__ == "__main__":
    plot_dual_tts()
