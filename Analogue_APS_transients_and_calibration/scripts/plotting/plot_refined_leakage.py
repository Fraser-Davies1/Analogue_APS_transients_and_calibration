import matplotlib.pyplot as plt
import polars as pl
import os

def plot_refined_comparison():
    plt.figure(figsize=(10, 6))
    
    files = [
        ("leakage_data_encased.csv", "Encased (Blu-Tac)", "bo-"),
        ("leakage_data_open.csv", "Open (Ambient)", "rs-")
    ]
    
    for f, label, fmt in files:
        if os.path.exists(f):
            # Remove the first data point (1ms artifact)
            df = pl.read_csv(f).slice(1)
            plt.plot(df["t_int"], df["i_pa"], fmt, label=label)
    
    plt.xlabel("Integration Time (s)")
    plt.ylabel("Derived Current (pA)")
    plt.title("Refined Leakage Comparison: Encased vs. Open\n(Excluding 1ms Switching Artifact)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("leakage_comparison_refined.png")
    print("Refined plot saved to leakage_comparison_refined.png")

if __name__ == "__main__":
    plot_refined_comparison()
