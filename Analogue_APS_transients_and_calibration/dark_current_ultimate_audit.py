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

def run_robust_dark_current_tts():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    SAT_THRESHOLD_FRAC = 0.30 # 70% depleted
    
    integration_times = [0.5, 1.0, 2.0, 3.0, 5.0]
    results = []

    print("====================================================")
    print("   ANALOGUE APS: ROBUST DARK CURRENT (TTS + ΔV)     ")
    print("   REF: CH3 (Reset) | SENSOR: CH1                   ")
    print("====================================================\n")

    try:
        with Bench.open("config/bench.yaml") as bench:
            # 1. Hardware Initialization
            # PSU CH1: Sensor Power
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            # SigGen CH1: Reset Signal (0-5V Square)
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen._send_command("SOUR1:FUNC:SQU:DCYC 50")
            bench.siggen.set_output_state(1, "ON")

            # 2. Scope Configuration
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable() 
            bench.osc.channel(3).setup(scale=1.0, offset=2.5, coupling="DC").enable() 
            
            # Hardware Trigger on Reset Falling Edge (CH3)
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel3")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel3")

            for t_target in integration_times:
                freq = 1.0 / (2.0 * t_target)
                print(f">>> MEASURING T_INT={t_target}s (Freq: {freq:.3f} Hz)")
                sys.stdout.flush()

                # Set window to capture 1.5 cycles
                total_window = 3.0 * t_target
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.2 * total_window)
                bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(3.0) 
                
                # Single Acquisition
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.0)
                
                # Data Extraction (CH1: Sensor, CH3: Reset)
                data = bench.osc.read_channels([1, 3])
                df = data.values
                t = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 3 (V)"].to_numpy()
                
                # --- Analysis Logic (Using your referenced method) ---
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0:
                    f_idx = falls[0]
                    r_pts = rises[rises > f_idx]
                    
                    if len(r_pts) > 0:
                        r_idx = r_pts[0]
                        t0 = t[f_idx]
                        t_end = t[r_idx]
                        actual_T = t_end - t0
                        
                        # Baseline voltage (average 10ms before falling edge)
                        v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                        v_sat_target = v_start * SAT_THRESHOLD_FRAC
                        
                        # Inspect the integration window
                        window_v = v_px[f_idx:r_idx]
                        window_t = t[f_idx:r_idx]
                        
                        # Check for 70% depletion (TTS threshold)
                        sat_indices = np.where(window_v <= v_sat_target)[0]
                        
                        if len(sat_indices) > 0:
                            # Saturated! Use TTS
                            is_sat = True
                            sat_idx = sat_indices[0]
                            t_to_sat = window_t[sat_idx] - t0
                            # Extrapolate delta_V for the whole integration period
                            slope = (v_start - v_sat_target) / t_to_sat
                            inferred_delta_v = slope * actual_T
                            mode = "TTS"
                        else:
                            # Not saturated. Use Direct Delta-V
                            is_sat = False
                            v_final = np.mean(v_px[max(0, r_idx-10):r_idx])
                            inferred_delta_v = v_start - v_final
                            mode = "Delta-V"

                        # I = C * (Delta_V / T)
                        i_dark = (inferred_delta_v / actual_T) * C_integrator
                        
                        print(f"    V_start: {v_start:.3f}V | actual_T: {actual_T:.3f}s | Mode: {mode}")
                        print(f"    I_dark:  {i_dark*1e12:.2f} pA")
                        
                        results.append({
                            "target_t": t_target,
                            "actual_t": actual_T,
                            "i_pa": i_dark * 1e12,
                            "mode": mode
                        })
                    else:
                        print("    [!] Error: Next rising edge not found.")
                else:
                    print("    [!] Error: No falling edge found in CH3 buffer.")

            # --- Summary ---
            print("\n" + "="*60)
            print(f"{'Target (s)':<10} {'Meas (s)':<10} {'I_dark (pA)':<15} {'Mode':<10}")
            print("-" * 60)
            for r in results:
                print(f"{r['target_t']:<10.1f} {r['actual_t']:<10.3f} {r['i_pa']:<15.2f} {r['mode']:<10}")
            print("="*60)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_robust_dark_current_tts()
