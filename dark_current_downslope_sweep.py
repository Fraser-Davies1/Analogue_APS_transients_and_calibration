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

def run_tts_downward_sweep():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    # Integration times to sweep
    integration_times = [0.5, 1.0, 2.0, 5.0, 10.0]
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: 11pF DARK CURRENT (DOWN-SLOPE)     ")
    print("====================================================\n")

    try:
        with Bench.open("config/bench.yaml") as bench:
            # 1. SigGen Setup: 5V Square Wave, High-Z
            print(">>> RESET CONFIG: 5Vpp Square, 2.5V Offset, High-Z")
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen._send_command("SOUR1:FUNC:SQU:DCYC 50")
            bench.siggen.set_output_state(1, "ON")

            # 2. Scope Setup
            # CH1 (Sensor) and CH2 (Reset Probe) centered on 2.5V
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
            
            # Trigger on Falling Edge of Reset (CH2)
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_target in integration_times:
                freq = 1.0 / (2.0 * t_target)
                print(f"\n>>> MEASURING T_INT: {t_target}s (Freq: {freq:.3f} Hz)")
                sys.stdout.flush()

                # Acquisition: 2.0x t_target to capture reset release + ramp
                total_window = 2.0 * t_target
                # Position trigger 10% from the left to maximize ramp visibility
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.1 * total_window)
                
                # Update frequency and let settle
                bench.siggen.set_frequency(1, freq)
                time.sleep(2.0) 
                
                print(f"    Capturing {total_window}s window...")
                # Increase LAMB timeout dynamically for the long transfer
                bench.osc._backend.timeout_ms = int(total_window * 1000 + 30000)
                
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.0)
                
                # Fetch Data
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                # --- Analysis ---
                # Start index is trigger (t=0)
                idx0 = np.argmin(np.abs(t))
                t_ramp = t[idx0:] - t[idx0]
                v_ramp = v_out[idx0:]
                
                v_start = v_ramp[0]
                # Saturation Threshold: 90% of the way to 0V (depletion)
                # If reset is ~4.5V, threshold is 0.45V
                v_threshold_90 = v_start * 0.1
                
                # Find where it crosses 90% saturation
                sat_crossings = np.where(v_ramp <= v_threshold_90)[0]
                
                if len(sat_crossings) > 0:
                    mode = "TTS (90%)"
                    idx_sat = sat_crossings[0]
                    actual_dt = t_ramp[idx_sat]
                    dv = v_ramp[idx_sat] - v_start
                else:
                    mode = "Delta-V"
                    # Use end of the target integration time
                    idx_end = np.argmin(np.abs(t_ramp - t_target))
                    actual_dt = t_ramp[idx_end]
                    dv = v_ramp[idx_end] - v_start

                # dV/dt (Downward)
                slope = dv / actual_dt if actual_dt > 0 else 0
                # I = C * |slope|
                i_dark = abs(slope * C_integrator)

                print(f"    Result: {mode} | dV/dt: {slope*1e3:.2f} mV/s | I_dark: {i_dark*1e12:.2f} pA")
                results_table.append({
                    "target": t_target,
                    "actual": actual_dt,
                    "slope_mv_s": slope * 1e3,
                    "i_pa": i_dark * 1e12,
                    "mode": mode
                })
                
                # Save individual verification plot
                plt.figure()
                plt.plot(t_ramp, v_ramp, 'k')
                plt.axhline(v_threshold_90, color='r', linestyle='--', label='90% Depletion')
                plt.title(f"Dark Integration: {t_target}s window")
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage (V)")
                plt.grid(True, alpha=0.3)
                plt.savefig(f"results/plots/dark_sweep_{t_target}s.png")
                plt.close()

            # Final Report
            print("\n" + "="*75)
            print(f"{'Target (s)':<12} {'Meas (s)':<12} {'Slope (mV/s)':<15} {'I_dark (pA)':<15} {'Mode':<10}")
            print("-" * 75)
            for r in results_table:
                print(f"{r['target']:<12.1f} {r['actual']:<12.3f} {r['slope_mv_s']:<15.2f} {r['i_pa']:<15.2f} {r['mode']:<10}")
            print("="*75)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_tts_downward_sweep()
