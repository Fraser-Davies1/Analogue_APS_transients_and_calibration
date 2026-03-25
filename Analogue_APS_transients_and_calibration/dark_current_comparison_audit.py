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

def run_comparison_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    TTS_THRESHOLD_FRAC = 0.30 # 70% depletion
    
    integration_times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5]
    results = []

    print("====================================================")
    print("   ANALOGUE APS: DARK CURRENT METHOD COMPARISON     ")
    print("   (Delta-V vs. 70% TTS | 11pF Integrator)          ")
    print("====================================================\n")

    try:
        with Bench.open("config/bench.yaml") as bench:
            # SigGen Setup (Reset Pulse)
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen._send_command("SOUR1:FUNC:SQU:DCYC 50")
            bench.siggen.set_output_state(1, "ON")

            # Scope Setup
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable() 
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
            
            # Falling Edge Trigger on Reset (CH2)
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_target in integration_times:
                freq = 1.0 / (2.0 * t_target)
                print(f">>> MEASURING T_INT={t_target}s")
                sys.stdout.flush()

                # Acquisition Window: 4x margin
                total_window = 4.0 * t_target
                trigger_pos = 0.2 * total_window
                bench.osc.set_time_axis(scale=total_window/10.0, position=trigger_pos)
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
                    # Find first rise after fall
                    r_pts = rises[rises > f_idx]
                    
                    if len(r_pts) > 0:
                        r_idx = r_pts[0]
                        t0 = t[f_idx]
                        t_end = t[r_idx]
                        actual_T = t_end - t0
                        
                        v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                        v_final = np.mean(v_px[max(f_idx, r_idx-10):r_idx])
                        
                        # 1. Delta-V Method
                        i_delta_v = abs((v_start - v_final) / actual_T) * C_integrator * 1e12
                        
                        # 2. TTS Method (70% depletion)
                        v_target_70 = v_start * (1.0 - 0.70)
                        window_v = v_px[f_idx:r_idx]
                        window_t = t[f_idx:r_idx]
                        sat_indices = np.where(window_v <= v_target_70)[0]
                        
                        if len(sat_indices) > 0:
                            idx_sat = sat_indices[0]
                            t_tts = window_t[idx_sat] - t0
                            i_tts = abs((v_start - v_target_70) / t_tts) * C_integrator * 1e12
                            tts_label = f"{i_tts:.2f} pA"
                        else:
                            i_tts = None
                            tts_label = "N/A (No Sat)"

                        print(f"    Delta-V: {i_delta_v:.2f} pA | TTS: {tts_label}")
                        
                        results.append({
                            "target": t_target,
                            "i_delta": i_delta_v,
                            "i_tts": i_tts,
                            "mode": "Saturated" if i_tts else "Linear"
                        })
                        
                        # Plot
                        plt.figure(figsize=(10, 4))
                        plt.plot(t - t0, v_px, 'k', label='Sensor Output')
                        plt.axvspan(0, actual_T, color='gray', alpha=0.1, label='Integ Period')
                        if i_tts:
                            plt.axvline(t_tts, color='r', linestyle='--', label='70% Point')
                        plt.title(f"Comparison Audit: {t_target}s (dV={i_delta_v:.1f}pA)")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Voltage (V)")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(f"results/plots/comparison_audit_{t_target}s.png")
                        plt.close()
                    else:
                        print("    [!] Error: End of cycle not found.")
                else:
                    print("    [!] Error: Pulse edges not captured.")

            # --- Final Report ---
            print("\n" + "="*70)
            print(f"{'Target (s)':<12} {'I (Delta-V) [pA]':<20} {'I (TTS) [pA]':<20}")
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
    run_comparison_audit()
