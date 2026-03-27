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

def run_tts_70_percent_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    THRESHOLD_DEPLETION = 0.70 # 70% saturation (depletion)
    
    integration_times = [0.5, 1.0, 2.0, 3.0, 5.0]
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: 70% TTS DARK CURRENT AUDIT         ")
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
            
            # Setup Edge Triggering on CH2 (Reset)
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_target in integration_times:
                freq = 1.0 / (2.0 * t_target)
                print(f">>> MEASURING T_INT: {t_target}s (Freq: {freq:.3f} Hz)")
                
                # Window = 2 cycles
                total_window = 2.0 * t_target
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.2 * total_window)
                bench.osc._backend.timeout_ms = 60000 
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(2.0) 
                
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.0)
                
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                v_reset = data.values["Channel 2 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                # --- ROBUST ANALYSIS ---
                # 1. Find the actual falling edge in the buffer
                # Threshold crossing from >2.5V to <2.5V
                edges = np.where((v_reset[:-1] > 2.5) & (v_reset[1:] <= 2.5))[0]
                if len(edges) == 0:
                    print("    [!] Error: No reset falling edge found in scope buffer.")
                    continue
                
                idx_trigger = edges[0]
                # Skip the first 10ms to avoid switching transients
                idx_start = idx_trigger + int(0.010 / (t[1]-t[0]))
                
                t_ramp = t[idx_start:] - t[idx_start]
                v_ramp = v_out[idx_start:]
                v_start_val = v_ramp[0]
                
                # Target for 70% saturation (downwards)
                v_target_70 = v_start_val * (1.0 - THRESHOLD_DEPLETION)
                
                sat_crossings = np.where(v_ramp <= v_target_70)[0]
                
                if len(sat_crossings) > 0:
                    # TTS METHOD
                    idx_70 = sat_crossings[0]
                    actual_dt = t_ramp[idx_70]
                    dv = v_ramp[idx_70] - v_start_val
                    mode = "TTS (70%)"
                else:
                    # DELTA-V METHOD (Linear)
                    # We use the slope over the first t_target duration
                    idx_end = np.argmin(np.abs(t_ramp - t_target))
                    actual_dt = t_ramp[idx_end]
                    dv = v_ramp[idx_end] - v_start_val
                    mode = "Delta-V"

                slope = dv / actual_dt if actual_dt > 0 else 0
                i_pa = abs(slope * C_integrator) * 1e12

                print(f"    Mode: {mode} | dV/dt: {slope*1e3:.2f} mV/s | I: {i_pa:.2f} pA")
                results_table.append({
                    "target": t_target,
                    "actual": actual_dt,
                    "slope": slope * 1e3,
                    "i": i_pa,
                    "mode": mode
                })
                
                # Verification Plot
                plt.figure()
                plt.plot(t_ramp, v_ramp, label='Integrator Output')
                plt.axhline(v_target_70, color='r', linestyle='--', label='70% Threshold')
                plt.title(f"Audit {t_target}s - {mode}")
                plt.xlabel("Time from Reset (s)")
                plt.ylabel("Voltage (V)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig(f"results/plots/audit_70pct_{t_target}s.png")
                plt.close()

            # Final Summary Table
            print("\n" + "="*80)
            print(f"{'Target (s)':<12} {'Meas (s)':<12} {'Slope (mV/s)':<15} {'I_dark (pA)':<15} {'Mode':<10}")
            print("-" * 80)
            for r in results_table:
                print(f"{r['target']:<12.1f} {r['actual']:<12.3f} {r['slope']:<15.2f} {r['i']:<15.2f} {r['mode']:<10}")
            print("="*80)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_tts_70_percent_audit()
