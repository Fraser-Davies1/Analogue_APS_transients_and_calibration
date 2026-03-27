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

def run_tts_final_attempt():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    V_MAX = 5.0
    V_THRESHOLD_UP = V_MAX * 0.9    # 4.5V
    V_THRESHOLD_DOWN = V_MAX * 0.1  # 0.5V
    
    # Sweep requested
    integration_times = [0.5, 1.0, 2.0, 5.0, 10.0]
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: HIGH-PRECISION DARK CURRENT SWEEP  ")
    print("====================================================\n")

    try:
        with Bench.open("config/bench.yaml") as bench:
            # Set absolute maximum timeout for LAMB
            for alias in ['osc', 'siggen', 'psu']:
                if hasattr(bench, alias):
                    getattr(bench, alias)._backend.timeout_ms = 120000

            # 1. SigGen Setup
            print(">>> RESET STIMULUS: Square Wave, 5Vpp, 2.5V Offset, High-Z")
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen._send_command("SOUR1:FUNC:SQU:DCYC 50")
            bench.siggen.set_output_state(1, "ON")

            # 2. Scope Setup
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
            
            # Falling Edge Trigger on Reset
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_int in integration_times:
                freq = 1.0 / (2.0 * t_int)
                print(f"\n>>> TARGET T_INT: {t_int}s (Freq: {freq:.3f} Hz)")
                sys.stdout.flush()

                # Acquisition Window: 2.5x t_int to see Reset High -> Fall -> Low
                total_window = 2.5 * t_int
                # Position trigger 10% from left
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.1 * total_window)
                
                bench.siggen.set_frequency(1, freq)
                print(f"    Stabilizing sensor for 5s...")
                time.sleep(5.0) 
                
                print("    Capturing integration cycle...")
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 2.0)
                
                # Retrieve Data
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                v_reset = data.values["Channel 2 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                # --- Analysis ---
                # Start at falling edge (t=0)
                idx0 = np.argmin(np.abs(t))
                # Look at integration period
                idx_end = np.argmin(np.abs(t - t_int))
                
                t_fit = t[idx0:idx_end]
                v_fit = v_out[idx0:idx_end]
                
                if len(t_fit) < 10:
                    print("    [!] Error: Insufficient data points in ramp.")
                    continue

                # Linear regression on the low-phase integration
                res = stats.linregress(t_fit, v_fit)
                
                # Check for Saturation (90% check)
                sat_mask = v_fit >= V_THRESHOLD_UP
                if np.any(sat_mask):
                    mode = "TTS (Sat)"
                    idx_sat = np.where(sat_mask)[0][0]
                    t_meas = t_fit[idx_sat] - t_fit[0]
                    slope = (v_fit[idx_sat] - v_fit[0]) / t_meas
                else:
                    mode = "Delta-V"
                    t_meas = t_fit[-1] - t_fit[0]
                    slope = res.slope

                i_dark = abs(slope * C_integrator)

                print(f"    Mode: {mode} | dV/dt: {slope*1e3:.2f} mV/s | I_dark: {i_dark*1e12:.2f} pA")
                results_table.append({
                    "target": t_int,
                    "actual": t_meas,
                    "i_pa": i_dark * 1e12,
                    "mode": mode,
                    "r2": res.rvalue**2
                })
                
                # Save trace
                plt.figure()
                plt.plot(t, v_out, 'k', label='Sensor')
                plt.plot(t, v_reset, 'b--', alpha=0.5, label='Reset Signal')
                plt.axvline(t[idx0], color='g', label='Reset Released')
                plt.title(f"Dark Current Integration (T={t_int}s)")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"results/plots/dark_audit_{t_int}s.png")
                plt.close()

            # Final Table
            print("\n" + "="*70)
            print(f"{'Target (s)':<12} {'Meas (s)':<12} {'I_dark (pA)':<15} {'Linearity':<12} {'Mode':<10}")
            print("-" * 70)
            for r in results_table:
                print(f"{r['target']:<12.1f} {r['actual']:<12.3f} {r['i_pa']:<15.2f} {r['r2']:<12.4f} {r['mode']:<10}")
            print("="*70)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_tts_final_attempt()
