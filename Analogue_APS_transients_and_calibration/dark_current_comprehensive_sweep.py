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

def run_tts_sweep():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/data", exist_ok=True)
    
    C_integrator = 11.0e-12 # 11pF
    V_max = 5.0             # System Rail
    V_threshold = V_max * 0.9 # 90% Saturation point
    
    # Integration times requested
    integration_times = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: DARK CURRENT SWEEP (TTS + DELTA-V) ")
    print("====================================================\n")
    sys.stdout.flush()

    try:
        with Bench.open("config/bench.yaml") as bench:
            # 1. Scope Baseline Configuration
            # CH1: Sensor Output, CH2: Reset Probe
            bench.osc.channel(1).setup(scale=0.5, offset=0, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=0, coupling="DC").enable()
            
            # 2. SigGen Reset Line Preparation
            bench.siggen.set_function(1, "DC")
            bench.siggen.set_output_state(1, "ON")

            for t_target in integration_times:
                print(f"\n>>> TARGET INTEGRATION TIME: {t_target}s")
                
                # Adjust timebase to fit the whole sequence
                # sequence: Reset(1s) + Integration(t_target) + Buffer(1s)
                total_window = t_target + 2.0
                bench.osc.set_time_axis(scale=total_window/10.0, position=total_window/2.0)
                
                # --- EXECUTION SEQUENCE ---
                # A. Hold Reset High
                bench.siggen.set_offset(1, 5.0)
                time.sleep(1.0)
                
                # B. Arm Scope
                bench.osc._send_command(":SINGle")
                time.sleep(0.5)
                
                # C. Release Reset (Start Integration)
                start_time = time.time()
                bench.siggen.set_offset(1, 0.0)
                
                # D. Wait for sequence to complete
                time.sleep(t_target + 0.5)
                
                # E. Capture Data
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                v_reset = data.values["Channel 2 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                # --- ANALYSIS ---
                # 1. Find Reset Falling Edge (where CH2 drops below 2.5V)
                reset_indices = np.where(v_reset < 2.5)[0]
                if len(reset_indices) == 0:
                    print("    [!] Error: Reset falling edge not detected on CH2.")
                    continue
                
                idx_start = reset_indices[0]
                t0 = t[idx_start]
                v0 = v_out[idx_start]
                
                # 2. Check for Saturation (90% of Max)
                sat_indices = np.where(v_out > V_threshold)[0]
                
                if len(sat_indices) > 0 and sat_indices[0] > idx_start:
                    # SATURATED CASE: Use Time-To-Saturate
                    idx_sat = sat_indices[0]
                    t_sat = t[idx_sat] - t0
                    v_sat = v_out[idx_sat]
                    
                    slope = (v_sat - v0) / t_sat
                    i_dark = slope * C_integrator
                    mode = "TTS (Sat)"
                    actual_time = t_sat
                else:
                    # LINEAR CASE: Use Delta-V across target time
                    # Look at end of buffer or end of integration
                    idx_end = np.argmin(np.abs(t - (t0 + t_target)))
                    v_end = v_out[idx_end]
                    t_actual = t[idx_end] - t0
                    
                    slope = (v_end - v0) / t_actual
                    i_dark = slope * C_integrator
                    mode = "Delta-V"
                    actual_time = t_actual

                print(f"    Mode: {mode} | dV/dt: {slope*1e3:.2f} mV/s | I_dark: {i_dark*1e12:.2f} pA")
                results_table.append({
                    "target_t": t_target,
                    "actual_t": actual_time,
                    "i_dark_pa": i_dark * 1e12,
                    "mode": mode
                })

            # 3. Final Summary Table
            print("\n" + "="*60)
            print(f"{'Target (s)':<12} {'Actual (s)':<12} {'I_dark (pA)':<15} {'Mode':<10}")
            print("-" * 60)
            for r in results_table:
                print(f"{r['target_t']:<12.1f} {r['actual_t']:<12.3f} {r['i_dark_pa']:<15.2f} {r['mode']:<10}")
            print("="*60)

            # Cleanup
            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_tts_sweep()
