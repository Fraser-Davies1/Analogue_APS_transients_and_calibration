import numpy as np
import polars as pl
import os
import time
import sys
import matplotlib.pyplot as plt
from pytestlab import Bench

# --- Framework Patch: Register WaveformGeneratorConfig ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass
# ---------------------------------------------------------

def capture_sweep(bench, label, integration_times):
    C_integrator = 11.0e-12
    TTS_THRESHOLD_FRAC = 0.50
    results = []
    
    for t_target in integration_times:
        freq = 1.0 / (2.0 * t_target)
        print(f"[{label}] Measuring T_INT: {t_target}s")
        
        total_window = 4.0 * t_target
        trigger_pos = 0.2 * total_window
        bench.osc.set_time_axis(scale=total_window/10.0, position=trigger_pos)
        bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
        
        bench.siggen.set_frequency(1, freq)
        time.sleep(3.0)
        
        bench.osc._send_command(":SINGle")
        time.sleep(total_window + 1.5)
        
        data = bench.osc.read_channels([1, 2])
        df = data.values
        t = df["Time (s)"].to_numpy()
        v_px = df["Channel 1 (V)"].to_numpy()
        v_rs = df["Channel 2 (V)"].to_numpy()
        
        edges = np.diff((v_rs > 2.5).astype(int))
        falls = np.where(edges == -1)[0]
        rises = np.where(edges == 1)[0]
        
        if len(falls) > 0 and len(rises) > 0:
            f_idx = falls[0]
            r_pts = rises[rises > f_idx]
            if len(r_pts) > 0:
                r_idx = r_pts[0]
                t0 = t[f_idx]
                actual_T = t[r_idx] - t0
                v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                v_final = np.mean(v_px[max(f_idx, r_idx-10):r_idx])
                v_target_50 = v_start * (1.0 - TTS_THRESHOLD_FRAC)
                
                # Delta-V
                i_delta_v = abs((v_start - v_final) / actual_T) * C_integrator * 1e12
                
                # TTS
                window_v = v_px[f_idx:r_idx]
                window_t = t[f_idx:r_idx]
                sat_indices = np.where(window_v <= v_target_50)[0]
                i_tts = None
                if len(sat_indices) > 0:
                    idx_sat = sat_indices[0]
                    t_tts = window_t[idx_sat] - t0
                    i_tts = abs((v_start - v_target_50) / t_tts) * C_integrator * 1e12
                
                results.append({
                    "target": t_target,
                    "i_delta": i_delta_v,
                    "i_tts": i_tts
                })
    return results

def run_combined_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    
    # 1. We already have 'Enclosed' data from previous turn, but for a true 'on-graph' comparison,
    # we'll assume the user wants to see the new 'Open' run and compare with the logic.
    # I'll save the 'Open' results and then create the composite plot.
    
    integration_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Hardcoded Enclosed results from previous successful turn
    enclosed_data = [
        {"target": 1.0, "i_delta": 12.71, "i_tts": None},
        {"target": 2.0, "i_delta": 12.23, "i_tts": None},
        {"target": 3.0, "i_delta": 10.69, "i_tts": 12.32},
        {"target": 4.0, "i_delta": 8.67,  "i_tts": 12.29},
        {"target": 5.0, "i_delta": 6.98,  "i_tts": 12.24}
    ]

    print(">>> RUNNING 'OPEN' AUDIT (AMBIENT LIGHT)...")
    with Bench.open("config/bench.yaml") as bench:
        # SigGen Init
        bench.siggen._send_command("OUTP1:LOAD INF")
        bench.siggen.set_function(1, "SQU")
        bench.siggen.set_amplitude(1, 5.0)
        bench.siggen.set_offset(1, 2.5)
        bench.siggen.set_output_state(1, "ON")

        open_results = capture_sweep(bench, "OPEN", integration_times)
        bench.siggen.set_output_state(1, "OFF")

    # Final Plotting
    plt.figure(figsize=(12, 7))
    
    t_enc = [r['target'] for r in enclosed_data]
    i_d_enc = [r['i_delta'] for r in enclosed_data]
    i_t_enc = [r['i_tts'] for r in enclosed_data]
    
    t_open = [r['target'] for r in open_results]
    i_d_open = [r['i_delta'] for r in open_results]
    i_t_open = [r['i_tts'] for r in open_results]

    # Enclosed Series
    plt.plot(t_enc, i_d_enc, 'k--', label='Enclosed: Delta-V', alpha=0.6)
    valid_t_enc = [t_enc[i] for i in range(len(i_t_enc)) if i_t_enc[i] is not None]
    valid_i_enc = [i for i in i_t_enc if i is not None]
    plt.plot(valid_t_enc, valid_i_enc, 'ks-', label='Enclosed: 50% TTS', linewidth=2)

    # Open Series
    plt.plot(t_open, i_d_open, 'r--', label='Open: Delta-V', alpha=0.6)
    valid_t_open = [t_open[i] for i in range(len(i_t_open)) if i_t_open[i] is not None]
    valid_i_open = [i for i in i_t_open if i is not None]
    plt.plot(valid_t_open, valid_i_open, 'ro-', label='Open: 50% TTS', linewidth=2)

    plt.title("Dark Current vs. Ambient Response: Method Comparison\nTTS Convergence vs. Delta-V Decay (11pF Integrator)")
    plt.xlabel("Integration Window (s)")
    plt.ylabel("Derived Current (pA)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    report_path = "results/plots/comprehensive_leakage_comparison.png"
    plt.savefig(report_path)
    print(f"\n>>> FINAL COMPARISON REPORT SAVED: {report_path}")

if __name__ == "__main__":
    run_combined_audit()
