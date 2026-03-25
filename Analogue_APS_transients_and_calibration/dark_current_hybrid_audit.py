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

def run_hybrid_dark_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    TTS_THRESHOLD_FRAC = 0.20 # 80% depletion means 20% of start voltage remains
    
    integration_times = [0.5, 1.0, 2.0, 5.0, 10.0]
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: HYBRID DARK CURRENT AUDIT          ")
    print("   (Delta-V < 5s | TTS >= 5s)                       ")
    print("====================================================\n")

    try:
        with Bench.open("config/bench.yaml") as bench:
            # 1. Reset Signal Setup
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

            for t_int in integration_times:
                freq = 1.0 / (2.0 * t_int)
                print(f"\n>>> TARGET T_INT: {t_int}s")
                sys.stdout.flush()

                # Acquisition window setup
                total_window = 2.5 * t_int
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.1 * total_window)
                bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(3.0) 
                
                print(f"    Capturing {total_window}s window...")
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.0)
                
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                # --- Analysis ---
                idx0 = np.argmin(np.abs(t)) # Reset release at t=0
                t_ramp = t[idx0:] - t[idx0]
                v_ramp = v_out[idx0:]
                v_start = v_ramp[0]

                if t_int >= 5.0:
                    # TTS Method (80% depletion)
                    mode = "TTS (80%)"
                    v_target = v_start * TTS_THRESHOLD_FRAC
                    crossings = np.where(v_ramp <= v_target)[0]
                    
                    if len(crossings) > 0:
                        idx_80 = crossings[0]
                        actual_dt = t_ramp[idx_80]
                        dv = v_ramp[idx_80] - v_start
                    else:
                        print("    [!] Warning: 80% depletion not reached. Falling back to Delta-V.")
                        mode = "Delta-V (Fallback)"
                        idx_end = np.argmin(np.abs(t_ramp - t_int))
                        actual_dt = t_ramp[idx_end]
                        dv = v_ramp[idx_end] - v_start
                else:
                    # Simple Delta-V Method
                    mode = "Delta-V"
                    idx_end = np.argmin(np.abs(t_ramp - t_int))
                    actual_dt = t_ramp[idx_end]
                    dv = v_ramp[idx_end] - v_start

                slope = dv / actual_dt if actual_dt > 0 else 0
                i_pa = abs(slope * C_integrator) * 1e12

                print(f"    Mode: {mode} | Slope: {slope*1e3:.2f} mV/s | I: {i_pa:.2f} pA")
                results_table.append({
                    "target": t_int,
                    "actual_t": actual_dt,
                    "slope": slope * 1e3,
                    "i": i_pa,
                    "mode": mode
                })
                
                # Plot
                plt.figure()
                plt.plot(t_ramp, v_ramp, 'k')
                plt.title(f"Integration Ramp ({t_int}s) - {mode}")
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage (V)")
                plt.grid(True, alpha=0.3)
                plt.savefig(f"results/plots/hybrid_audit_{t_int}s.png")
                plt.close()

            # Final Table
            print("\n" + "="*80)
            print(f"{'Target (s)':<12} {'Meas (s)':<12} {'Slope (mV/s)':<15} {'Current (pA)':<15} {'Mode':<15}")
            print("-" * 80)
            for r in results_table:
                print(f"{r['target']:<12.1f} {r['actual_t']:<12.3f} {r['slope']:<15.2f} {r['i']:<15.2f} {r['mode']:<15}")
            print("="*80)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_hybrid_dark_audit()
