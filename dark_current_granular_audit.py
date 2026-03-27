import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from scipy import stats
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

def run_granular_dark_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    TTS_THRESHOLD_FRAC = 0.20
    
    # Granular sweep between 0.5 and 5s
    integration_times = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: GRANULAR DARK CURRENT AUDIT        ")
    print("   (0.5s to 5.0s | Refined Trigger Sync)            ")
    print("====================================================\n")

    try:
        with Bench.open("config/bench.yaml") as bench:
            # SigGen Setup
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen._send_command("SOUR1:FUNC:SQU:DCYC 50")
            bench.siggen.set_output_state(1, "ON")

            # Scope Setup
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
            
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_int in integration_times:
                freq = 1.0 / (2.0 * t_int)
                print(f">>> MEASURING T_INT: {t_int}s (Freq: {freq:.3f} Hz)")
                sys.stdout.flush()

                total_window = 2.2 * t_int
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.1 * total_window)
                bench.osc._backend.timeout_ms = 60000 
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(3.0) 
                
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.5)
                
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                v_reset = data.values["Channel 2 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                # --- Analysis ---
                # 1. Identify Reset Release (Falling Edge on CH2)
                # We find the exact crossing point of 2.5V
                crossings = np.where(np.diff(np.sign(v_reset - 2.5)) < 0)[0]
                if len(crossings) == 0:
                    print("    [!] Error: Trigger edge not found in buffer.")
                    continue
                
                # Use the first falling edge in the buffer
                idx_trigger = crossings[0]
                
                # Offset Start by 50ms to skip switching noise
                idx_start = idx_trigger + int(0.05 / (t[1]-t[0]))
                
                t_rel = t[idx_start:] - t[idx_start]
                v_ramp = v_out[idx_start:]
                v_start_val = v_ramp[0]

                if t_int >= 5.0:
                    mode = "TTS (80%)"
                    v_target = v_start_val * TTS_THRESHOLD_FRAC
                    sat_indices = np.where(v_ramp <= v_target)[0]
                    if len(sat_indices) > 0:
                        idx_80 = sat_indices[0]
                        actual_dt = t_rel[idx_80]
                        dv = v_ramp[idx_80] - v_start_val
                    else:
                        mode = "Delta-V (Fallback)"
                        idx_end = np.argmin(np.abs(t_rel - t_int))
                        actual_dt = t_rel[idx_end]
                        dv = v_ramp[idx_end] - v_start_val
                else:
                    mode = "Delta-V"
                    idx_end = np.argmin(np.abs(t_rel - t_int))
                    actual_dt = t_rel[idx_end]
                    dv = v_ramp[idx_end] - v_start_val

                slope = dv / actual_dt if actual_dt > 0 else 0
                i_pa = abs(slope * C_integrator) * 1e12
                
                print(f"    V_start: {v_start_val:.3f}V | V_end: {v_out[idx_start + idx_end]:.3f}V")
                print(f"    Mode: {mode} | Slope: {slope*1e3:.2f} mV/s | I: {i_pa:.2f} pA")
                
                results_table.append({
                    "target": t_int,
                    "actual": actual_dt,
                    "i": i_pa,
                    "slope": slope * 1e3
                })

            # Final Summary
            print("\n" + "="*70)
            print(f"{'Target (s)':<12} {'Meas (s)':<12} {'Slope (mV/s)':<15} {'I_dark (pA)':<15}")
            print("-" * 70)
            for r in results_table:
                print(f"{r['target']:<12.2f} {r['actual']:<12.3f} {r['slope']:<15.2f} {r['i']:<15.2f}")
            print("="*70)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_granular_dark_audit()
