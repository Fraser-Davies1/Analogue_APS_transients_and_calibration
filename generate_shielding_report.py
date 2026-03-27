import numpy as np
import matplotlib.pyplot as plt
import os

def generate_final_report():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    
    labels = ["Grounded_SigGen_ON", "Grounded_SigGen_OFF", "Ungrounded_SigGen_ON", "Ungrounded_SigGen_OFF"]
    colors = ["black", "grey", "firebrick", "royalblue"]
    
    plt.figure(figsize=(14, 9))
    
    for i, label in enumerate(labels):
        data_path = f"results/data/{label}_hr.npz"
        if os.path.exists(data_path):
            data = np.load(data_path)
            plt.loglog(data["f"], np.sqrt(data["psd"])*1e9, color=colors[i], 
                       linewidth=1.0, alpha=0.8, label=label.replace("_", " "))
        else:
            print(f"Warning: {data_path} missing")

    plt.title("Analogue APS: High-Resolution Noise Floor Audit\nReal Hardware | Shielding vs. SigGen Crosstalk")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Noise Density (nV / √Hz)")
    plt.xlim(left=5)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("results/plots", exist_ok=True)
    plot_path = "results/plots/final_shielding_audit_report.png"
    plt.savefig(plot_path)
    print(f"REPORT_SAVED: {plot_path}")

if __name__ == "__main__":
    generate_final_report()
