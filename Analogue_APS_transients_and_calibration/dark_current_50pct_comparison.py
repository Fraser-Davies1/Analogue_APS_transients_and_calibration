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

def run_50pct_comparison_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    TTS_THRESHOLD_FRAC = 0.50 # 50% depletion
    
    integration_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    results = []

    print("====================================================")
    print("   ANALOGUE APS: 50% TTS vs DELTA-V COMPARISON      ")
    print("   (Targeting Reachable Saturation)                 ")
    print("====================================================\n")

    try:
        with Bench.open("config/bench.yaml") as bench:
            # 1. SigGen Setup
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen._send_command("SOUR1:FUNC:SQU:DCYC 50")
            bench.siggen.set_output_state(1, "ON")

            # 2. Scope Setup
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable() 
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
            
            # Falling Edge Trigger on Reset (CH2)
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_target in integration_times:
                freq = 1.0 / (2.0 * t_target)
                print(f">>> AUDITING T_INT: {t_target}s")
                sys.stdout.flush()

                total_window = 4.0 * t_target
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.2 * total_window)
                bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(3.0) 
                
                print(f"    Capturing {total_window:.1f}s Acquisition...")
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.5)
                
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 2 (V)"].to_numpy()
                
                # --- Analysis ---
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
                        
                        # Target for 50% depletion
                        v_target_50 = v_start * (1.0 - TTS_THRESHOLD_FRAC)
                        
                        window_v = v_px[f_idx:r_idx]
                        window_t = t[f_idx:r_idx]
                        sat_indices = np.where(window_v <= v_target_50)[0]

                        # 1. Delta-V Method
                        i_delta_v = abs((v_start - v_final) / actual_T) * C_integrator * 1e12
                        
                        # 2. TTS Method
                        if len(sat_indices) > 0:
                            idx_sat = sat_indices[0]
                            t_tts = window_t[idx_sat] - t0
                            i_tts = abs((v_start - v_target_50) / t_tts) * C_integrator * 1e12
                            tts_label = f"{i_tts:.2f} pA"
                        else:
                            i_tts = None
                            tts_label = "N/A"

                        print(f"    Delta-V: {i_delta_v:.2f} pA | 50% TTS: {tts_label}")
                        
                        results.append({
                            "target": t_target,
                            "i_delta": i_delta_v,
                            "i_tts": i_tts
                        })
                        
                        # Plot
                        plt.figure(figsize=(10, 4))
                        plt.plot(t - t0, v_px, 'k', label='Sensor')
                        if i_tts:
                            plt.axvline(t_tts, color='r', linestyle='--', label='50% Sat Point')
                            plt.axhline(v_target_50, color='r', alpha=0.3)
                        plt.title(f"50% TTS Comparison: {t_target}s")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(f"results/plots/comparison_50pct_{t_target}s.png")
                        plt.close()
                    else:
                        print("    [!] End of integration not found.")
                else:
                    print("    [!] Edges not found.")

            # Final Table
            print("\n" + "="*70)
            print(f"{'Target (s)':<12} {'I (Delta-V) [pA]':<20} {'I (50% TTS) [pA]':<20}")
            print("-" * 70)
            for r in results:
                tts_str = f"{r['i_tts']:<20.2f}" if r['i_tts'] else f"{'N/A':<20}"
                print(f"{r['target']:<12.1f} {r['i_delta']:<20.2f} {tts_str}")
            print("="*70)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_50pct_comparison_audit()
