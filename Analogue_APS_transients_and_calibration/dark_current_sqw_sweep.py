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

def run_tts_square_wave_sweep():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    V_max = 5.0
    V_threshold = V_max * 0.9 # 4.5V
    
    # Target integration times (s)
    integration_times = [0.5, 1.0, 2.0, 5.0, 10.0]
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: SQUARE-WAVE RESET TTS SWEEP        ")
    print("====================================================\n")

    try:
        with Bench.open("config/bench.yaml") as bench:
            # 1. Setup Signal Generator (CH1 Reset)
            print(">>> CONFIGURING SIGGEN (SQUARE WAVE, HIGH-Z)")
            # Set to High-Z (INFinity)
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            # Use 50% duty cycle; Integration time will be 1/(2*f)
            bench.siggen._send_command("SOUR1:FUNC:SQU:DCYC 50")
            bench.siggen.set_output_state(1, "ON")

            # 2. Scope Setup
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable() # Center at 2.5V
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
            
            # Set trigger to falling edge of Reset (CH2)
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_int in integration_times:
                freq = 1.0 / (2.0 * t_int)
                print(f"\n>>> MEASURING T_INT = {t_int}s (Freq: {freq:.3f} Hz)")
                
                # Adjust timebase: capture 1.5 cycles
                total_window = 3.0 * t_int
                bench.osc.set_time_axis(scale=total_window/10.0, position=t_int)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(1.0) # Allow freq to stabilize
                
                # Force single acquisition
                print("    Waiting for trigger...")
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.0)
                
                # Fetch data
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                v_reset = data.values["Channel 2 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                # --- ANALYSIS ---
                # Find the trigger point (t=0 is the falling edge)
                idx_start = np.argmin(np.abs(t)) 
                t_rel = t[idx_start:] - t[idx_start]
                v_rel = v_out[idx_start:]
                
                # Check for saturation
                sat_mask = v_rel >= V_threshold
                if np.any(sat_mask):
                    mode = "TTS (Sat)"
                    idx_sat = np.where(sat_mask)[0][0]
                    t_meas = t_rel[idx_sat]
                    v_delta = v_rel[idx_sat] - v_rel[0]
                else:
                    mode = "Delta-V"
                    # Measure at the end of the low-cycle
                    idx_end = np.argmin(np.abs(t_rel - t_int))
                    t_meas = t_rel[idx_end]
                    v_delta = v_rel[idx_end] - v_rel[0]

                # Avoid division by zero
                if t_meas > 0:
                    slope = v_delta / t_meas
                    i_dark = slope * C_integrator
                else:
                    slope, i_dark = 0, 0

                print(f"    RESULT: {mode} | I_dark: {i_dark*1e12:.2f} pA | dV/dt: {slope*1e3:.2f} mV/s")
                results_table.append({
                    "target_t": t_int,
                    "actual_t": t_meas,
                    "i_dark_pa": i_dark * 1e12,
                    "mode": mode
                })
                
                # Plot individual ramp for verification
                plt.figure()
                plt.plot(t_rel, v_rel, label='Sensor Output')
                plt.axhline(V_threshold, color='r', linestyle='--', label='90% Sat')
                plt.title(f"Integration Ramp (T_int={t_int}s)")
                plt.xlabel("Time from Reset (s)")
                plt.ylabel("Voltage (V)")
                plt.legend()
                plt.savefig(f"results/plots/ramp_{t_int}s.png")
                plt.close()

            # Final Table
            print("\n" + "="*65)
            print(f"{'Target T (s)':<12} {'Meas T (s)':<12} {'I_dark (pA)':<15} {'Mode':<10}")
            print("-" * 65)
            for r in results_table:
                print(f"{r['target_t']:<12.1f} {r['actual_t']:<12.3f} {r['i_dark_pa']:<15.2f} {r['mode']:<10}")
            print("="*65)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_tts_square_wave_sweep()
