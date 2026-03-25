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

def run_open_sweep_and_compare():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    TTS_THRESHOLD_FRAC = 0.50 
    integration_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Baseline Enclosed Data (from our previous successful audit)
    enclosed_results = [
        {"target": 1.0, "dv": 12.71, "tts": None},
        {"target": 2.0, "dv": 12.23, "tts": None},
        {"target": 3.0, "dv": 10.69, "tts": 12.32},
        {"target": 4.0, "dv": 8.67,  "tts": 12.29},
        {"target": 5.0, "dv": 6.98,  "tts": 12.24}
    ]

    open_results = []

    print("====================================================")
    print("   ANALOGUE APS: FINAL COMPARATIVE AUDIT            ")
    print("   STATE: UNCOVERED (OPEN)                          ")
    print("====================================================\n")
    sys.stdout.flush()

    try:
        with Bench.open("config/bench.yaml") as bench:
            # Setup Stimulus (Reset Pulse Only, LED Stimulus OFF)
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen.set_output_state(1, "ON")

            # Scope Setup
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
            
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_target in integration_times:
                freq = 1.0 / (2.0 * t_target)
                print(f">>> MEASURING T_INT={t_target}s (OPEN)")
                sys.stdout.flush()

                total_window = 4.0 * t_target
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.2 * total_window)
                bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(3.0) 
                
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.5)
                
                data = bench.osc.read_channels([1, 2])
                t = data.values["Time (s)"].to_numpy()
                v_px = data.values["Channel 1 (V)"].to_numpy()
                v_rs = data.values["Channel 2 (V)"].to_numpy()
                
                # Analysis Logic (Matches Enclosed Run exactly)
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    f_idx = falls[0]
                    r_pts = rises[rises > f_idx]
                    if len(r_pts) > 0:
                        r_idx = r_pts[0]
                        t0, t_end = t[f_idx], t[r_idx]
                        actual_T = t_end - t0
                        v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                        v_final = np.mean(v_px[max(f_idx, r_idx-10):r_idx])
                        v_target_50 = v_start * (1.0 - TTS_THRESHOLD_FRAC)
                        
                        # 1. Delta-V
                        i_dv = abs((v_start - v_final) / actual_T) * C_integrator * 1e12
                        
                        # 2. 50% TTS
                        window_v, window_t = v_px[f_idx:r_idx], t[f_idx:r_idx]
                        sat_indices = np.where(window_v <= v_target_50)[0]
                        i_tts = None
                        if len(sat_indices) > 0:
                            t_tts = window_t[sat_indices[0]] - t0
                            i_tts = abs((v_start - v_target_50) / t_tts) * C_integrator * 1e12
                        
                        print(f"    [OPEN] Delta-V: {i_dv:.2f} pA | TTS: {i_tts if i_tts else 'N/A'}")
                        open_results.append({"target": t_target, "dv": i_dv, "tts": i_tts})

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")

    # --- FINAL PLOTTING ---
    plt.figure(figsize=(12, 7))
    
    # Extract Plotting Arrays
    t_axis = [r['target'] for r in enclosed_results]
    
    # Enclosed Series
    dv_enc = [r['dv'] for r in enclosed_results]
    tts_enc_mask = [r['tts'] is not None for r in enclosed_results]
    t_tts_enc = [t_axis[i] for i, m in enumerate(tts_enc_mask) if m]
    i_tts_enc = [r['tts'] for r in enclosed_results if r['tts'] is not None]

    # Open Series
    dv_open = [r['dv'] for r in open_results]
    tts_open_mask = [r['tts'] is not None for r in open_results]
    t_tts_open = [open_results[i]['target'] for i, m in enumerate(tts_open_mask) if m]
    i_tts_open = [r['tts'] for r in open_results if r['tts'] is not None]

    plt.plot(t_axis, dv_enc, 'k--', alpha=0.5, label='Enclosed: Delta-V')
    plt.plot(t_tts_enc, i_tts_enc, 'ks-', label='Enclosed: 50% TTS', linewidth=2)
    
    plt.plot(t_axis[:len(dv_open)], dv_open, 'r--', alpha=0.5, label='Open: Delta-V')
    plt.plot(t_tts_open, i_tts_open, 'ro-', label='Open: 50% TTS', linewidth=2)

    plt.title("Dark Current Comparison: Enclosed vs. Uncovered\nExact Method Reproducibility | 11pF Integrator")
    plt.xlabel("Target Integration Time (s)")
    plt.ylabel("Derived Current (pA)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    plot_path = "results/plots/final_enclosed_vs_open_audit.png"
    plt.savefig(plot_path)
    print(f"\n>>> AUDIT COMPLETE. REPORT SAVED: {plot_path}")
    sys.stdout.flush()

if __name__ == "__main__":
    run_open_sweep_and_compare()
