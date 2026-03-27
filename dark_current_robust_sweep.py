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

def run_tts_robust_sweep():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    V_MAX = 5.0
    V_THRESHOLD_UP = V_MAX * 0.9    # 4.5V
    V_THRESHOLD_DOWN = V_MAX * 0.1  # 0.5V
    
    integration_times = [0.5, 1.0, 2.0, 5.0, 10.0]
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: ROBUST DARK CURRENT TTS SWEEP      ")
    print("====================================================\n")

    try:
        # Increase timeout explicitly in config for the 10s run
        with Bench.open("config/bench.yaml") as bench:
            # Update timeout for this session
            for inst in bench._instrument_instances.values():
                if hasattr(inst._backend, 'timeout_ms'):
                    inst._backend.timeout_ms = 60000 

            # 1. SigGen Configuration (Reset)
            print(">>> CONFIGURING SIGGEN (5Vpp Square, 2.5V Offset, High-Z)")
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen._send_command("SOUR1:FUNC:SQU:DCYC 50")
            bench.siggen.set_output_state(1, "ON")

            # 2. Scope Configuration
            # CH1: Sensor (0-5V), CH2: Reset Probe (0-5V)
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
            
            # Trigger on Falling Edge of Reset (CH2)
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_int in integration_times:
                freq = 1.0 / (2.0 * t_int)
                print(f"\n>>> TARGET T_INT: {t_int}s (Freq: {freq:.3f} Hz)")
                sys.stdout.flush()

                # Window: 1.5 cycles to ensure we see the reset and the ramp
                total_window = 3.0 * t_int
                bench.osc.set_time_axis(scale=total_window/10.0, position=t_int)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(2.0) # Stabilize
                
                print("    Waiting for Falling Edge Trigger...")
                bench.osc._send_command(":SINGle")
                # Wait for acquisition + safety margin
                time.sleep(total_window + 1.0)
                
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                # --- Analysis ---
                # Trigger point is t=0 (the falling edge)
                start_idx = np.argmin(np.abs(t))
                t_ramp = t[start_idx:] - t[start_idx]
                v_ramp = v_out[start_idx:]
                
                v_start = v_ramp[0]
                
                # Detect Ramp Direction (Up/Down)
                # Compare start to end of t_int
                end_idx = np.argmin(np.abs(t_ramp - t_int))
                v_end_val = v_ramp[end_idx]
                
                direction = "UP" if v_end_val > v_start else "DOWN"
                
                # Check for Saturation
                if direction == "UP":
                    sat_mask = v_ramp >= V_THRESHOLD_UP
                else:
                    sat_mask = v_ramp <= V_THRESHOLD_DOWN
                
                if np.any(sat_mask):
                    mode = "TTS (Sat)"
                    idx_sat = np.where(sat_mask)[0][0]
                    actual_t = t_ramp[idx_sat]
                    dv = v_ramp[idx_sat] - v_start
                else:
                    mode = "Delta-V"
                    actual_t = t_ramp[end_idx]
                    dv = v_end_val - v_start

                if actual_t > 0:
                    slope = dv / actual_t
                    i_dark = abs(slope * C_integrator)
                else:
                    slope, i_dark = 0, 0

                print(f"    Direction: {direction} | Mode: {mode}")
                print(f"    Slope: {slope*1e3:.2f} mV/s | I_dark: {i_dark*1e12:.2f} pA")
                
                results_table.append({
                    "target": t_int,
                    "actual": actual_t,
                    "i_pa": i_dark * 1e12,
                    "mode": mode,
                    "dir": direction
                })
                
                # Plot
                plt.figure()
                plt.plot(t_ramp, v_ramp, label=f'Output ({direction})')
                plt.title(f"Dark Current Integration (T_target={t_int}s)")
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage (V)")
                plt.grid(True, alpha=0.3)
                plt.savefig(f"results/plots/dark_ramp_{t_int}s.png")
                plt.close()

            # Final Table
            print("\n" + "="*70)
            print(f"{'Target (s)':<12} {'Meas (s)':<12} {'I_dark (pA)':<15} {'Dir':<6} {'Mode':<10}")
            print("-" * 70)
            for r in results_table:
                print(f"{r['target']:<12.1f} {r['actual']:<12.3f} {r['i_pa']:<15.2f} {r['dir']:<6} {r['mode']:<10}")
            print("="*70)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_tts_robust_sweep()
