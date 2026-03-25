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

def run_wide_window_dark_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    SAT_THRESHOLD_FRAC = 0.30 
    
    integration_times = [0.5, 1.0, 2.0, 3.0, 5.0]
    results = []

    print("====================================================")
    print("   ANALOGUE APS: WIDE-WINDOW DARK CURRENT AUDIT     ")
    print("   (4x Window Margin | CH2 REF | CH1 SENSOR)        ")
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
                print(f">>> MEASURING T_INT={t_target}s (Freq: {freq:.3f} Hz)")
                sys.stdout.flush()

                # Acquisition Window: 4x t_target for massive margin
                total_window = 4.0 * t_target
                # Position trigger at 2s if window is large, otherwise 20%
                trigger_pos = 0.2 * total_window
                bench.osc.set_time_axis(scale=total_window/10.0, position=trigger_pos)
                
                # Dynamic LAMB timeout
                bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(3.0) 
                
                print(f"    Capturing {total_window:.1f}s Acquisition...")
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.0)
                
                # Fetch Data
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 2 (V)"].to_numpy()
                
                # --- Analysis ---
                # Detect Edges on CH2
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0:
                    f_idx = falls[0]
                    # Find first rising edge AFTER the falling edge
                    r_pts = rises[rises > f_idx]
                    
                    if len(r_pts) > 0:
                        r_idx = r_pts[0]
                        t0 = t[f_idx]
                        t_end = t[r_idx]
                        actual_T = t_end - t0
                        
                        # Use average of 10ms before reset for baseline
                        v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                        v_sat_target = v_start * SAT_THRESHOLD_FRAC
                        
                        window_v = v_px[f_idx:r_idx]
                        window_t = t[f_idx:r_idx]
                        
                        # Check for 70% depletion
                        sat_indices = np.where(window_v <= v_sat_target)[0]
                        
                        if len(sat_indices) > 0:
                            is_sat = True
                            sat_idx = sat_indices[0]
                            t_meas = window_t[sat_idx] - t0
                            slope = (v_start - v_sat_target) / t_meas
                            inferred_delta_v = slope * actual_T
                            mode = "TTS"
                        else:
                            is_sat = False
                            # Mean of last 10ms of integration for final V
                            v_final = np.mean(v_px[max(f_idx, r_idx-10):r_idx])
                            inferred_delta_v = v_start - v_final
                            actual_T_meas = actual_T
                            mode = "Delta-V"

                        i_dark = (inferred_delta_v / actual_T) * C_integrator
                        
                        print(f"    Measured Integration Time: {actual_T:.3f}s")
                        print(f"    Mode: {mode} | I_dark: {i_dark*1e12:.2f} pA")
                        
                        results.append({
                            "target": t_target,
                            "actual": actual_T,
                            "i_pa": i_dark * 1e12,
                            "mode": mode
                        })
                        
                        # Plot Verification
                        plt.figure(figsize=(10, 5))
                        plt.plot(t - t0, v_px, 'k', label='Sensor (CH1)')
                        plt.plot(t - t0, v_rs, 'b--', alpha=0.3, label='Reset (CH2)')
                        plt.axvspan(0, actual_T, color='yellow', alpha=0.1, label='Integ. Period')
                        plt.title(f"Dark Current Audit: {t_target}s (Derived {i_dark*1e12:.1f}pA)")
                        plt.xlabel("Time from Reset Release (s)")
                        plt.ylabel("Voltage (V)")
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.savefig(f"results/plots/wide_window_audit_{t_target}s.png")
                        plt.close()
                    else:
                        print("    [!] Error: Rising edge not captured. Increase window.")
                else:
                    print("    [!] Error: Falling edge not captured.")

            # --- Final Table ---
            print("\n" + "="*65)
            print(f"{'Target (s)':<12} {'Actual (s)':<12} {'I_dark (pA)':<15} {'Mode':<10}")
            print("-" * 65)
            for r in results:
                print(f"{r['target']:<12.1f} {r['actual']:<12.3f} {r['i_pa']:<15.2f} {r['mode']:<10}")
            print("="*65)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_wide_window_dark_audit()
