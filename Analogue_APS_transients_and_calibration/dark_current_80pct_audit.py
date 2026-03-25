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

def run_tts_80_percent_sweep():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    THRESHOLD_FRAC = 0.80  # 80% depletion
    
    integration_times = [0.5, 1.0, 2.0, 5.0, 10.0]
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: 80% TTS DARK CURRENT AUDIT         ")
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
            
            # Falling Edge Trigger on Reset
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_target in integration_times:
                freq = 1.0 / (2.0 * t_target)
                print(f"\n>>> AUDITING T_INT: {t_target}s")
                sys.stdout.flush()

                total_window = 2.5 * t_target
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.1 * total_window)
                bench.osc._backend.timeout_ms = int(total_window * 1000 + 30000)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(3.0) # Settle
                
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.0)
                
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                # Analysis
                idx0 = np.argmin(np.abs(t))
                t_ramp = t[idx0:] - t[idx0]
                v_ramp = v_out[idx0:]
                
                v_start = v_ramp[0]
                # Target voltage for 80% depletion (downwards)
                v_target_80 = v_start * (1 - THRESHOLD_FRAC)
                
                sat_indices = np.where(v_ramp <= v_target_80)[0]
                
                if len(sat_indices) > 0:
                    # TTS MODE
                    idx_80 = sat_indices[0]
                    t_meas = t_ramp[idx_80]
                    dv = v_ramp[idx_80] - v_start
                    mode = "TTS (80%)"
                else:
                    # LINEAR MODE (Full window)
                    idx_end = np.argmin(np.abs(t_ramp - t_target))
                    t_meas = t_ramp[idx_end]
                    dv = v_ramp[idx_end] - v_start
                    mode = "Delta-V"

                slope = dv / t_meas if t_meas > 0 else 0
                i_pa = abs(slope * C_integrator) * 1e12

                print(f"    Mode: {mode} | Slope: {slope*1e3:.2f} mV/s | I: {i_pa:.2f} pA")
                results_table.append({
                    "target": t_target,
                    "actual_t": t_meas,
                    "slope": slope * 1e3,
                    "i": i_pa,
                    "mode": mode
                })
                
                plt.figure()
                plt.plot(t_ramp, v_ramp, 'k')
                plt.axhline(v_target_80, color='r', linestyle='--', label='80% Threshold')
                plt.title(f"Dark Current Audit ({t_target}s) - {mode}")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"results/plots/audit_80pct_{t_target}s.png")
                plt.close()

            # Final Table
            print("\n" + "="*70)
            print(f"{'Target (s)':<12} {'Meas (s)':<12} {'Slope (mV/s)':<15} {'Current (pA)':<15}")
            print("-" * 70)
            for r in results_table:
                print(f"{r['target']:<12.1f} {r['actual_t']:<12.3f} {r['slope']:<15.2f} {r['i']:<15.2f}")
            print("="*70)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_tts_80_percent_sweep()
