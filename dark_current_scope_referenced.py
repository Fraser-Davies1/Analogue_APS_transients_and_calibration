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

def run_scope_referenced_audit():
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/plots", exist_ok=True)
    
    C_integrator = 11.0e-12 
    SAT_THRESHOLD_PERCENT = 0.70
    
    integration_times = [0.5, 1.0, 2.0, 3.0, 5.0]
    results_table = []

    print("====================================================")
    print("   ANALOGUE APS: SCOPE-REFERENCED DARK CURRENT      ")
    print("   (CH1: SENSOR | CH2: RESET PROBE)                 ")
    print("====================================================\n")

    try:
        with Bench.open("config/bench.yaml") as bench:
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen._send_command("SOUR1:FUNC:SQU:DCYC 50")
            bench.siggen.set_output_state(1, "ON")

            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
            
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            for t_target in integration_times:
                freq = 1.0 / (2.0 * t_target)
                print(f">>> CAPTURING T_INT: {t_target}s (Freq: {freq:.3f} Hz)")
                sys.stdout.flush()

                total_window = 2.5 * t_target
                bench.osc.set_time_axis(scale=total_window/10.0, position=0.1 * total_window)
                bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(3.0) 
                
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.0)
                
                data = bench.osc.read_channels([1, 2])
                v_out = data.values["Channel 1 (V)"].to_numpy()
                v_reset = data.values["Channel 2 (V)"].to_numpy()
                t = data.values["Time (s)"].to_numpy()
                
                edge_indices = np.where((v_reset[:-1] > 2.5) & (v_reset[1:] <= 2.5))[0]
                
                if len(edge_indices) == 0:
                    print(f"    [!] Error: Reset edge not found in CH2.")
                    continue
                
                idx_t0 = edge_indices[0]
                t0 = t[idx_t0]
                
                idx_start = idx_t0 + int(0.010 / (t[1]-t[0]))
                t_ramp = t[idx_start:] - t[idx_start]
                v_ramp = v_out[idx_start:]
                v_reset_level = v_ramp[0]
                
                v_target_sat = v_reset_level * (1.0 - SAT_THRESHOLD_PERCENT)
                sat_crossings = np.where(v_ramp <= v_target_sat)[0]
                
                if len(sat_crossings) > 0:
                    mode = "TTS (70%)"
                    idx_sat = sat_crossings[0]
                    dt = t_ramp[idx_sat]
                    dv = v_ramp[idx_sat] - v_reset_level
                else:
                    mode = "Delta-V"
                    idx_end = np.argmin(np.abs(t_ramp - t_target))
                    dt = t_ramp[idx_end]
                    dv = v_ramp[idx_end] - v_reset_level

                slope = dv / dt if dt > 0 else 0
                i_pa = abs(slope * C_integrator) * 1e12

                print(f"    Mode: {mode} | Slope: {slope*1e3:.2f} mV/s | I: {i_pa:.2f} pA")
                results_table.append({
                    "target": t_target,
                    "actual": dt,
                    "slope": slope * 1e3,
                    "i": i_pa,
                    "mode": mode
                })
                
                plt.figure()
                plt.plot(t - t0, v_out, label='Sensor CH1')
                plt.plot(t - t0, v_reset, '--', alpha=0.4, label='Reset CH2')
                plt.axhline(v_target_sat, color='r', linestyle=':', label='70% Threshold')
                plt.title(f"Dark Current Audit ({t_target}s) - {mode}")
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage (V)")
                plt.xlim(-0.1, t_target + 0.1)
                plt.grid(True, alpha=0.3)
                plt.savefig(f"results/plots/scope_ref_audit_{t_target}s.png")
                plt.close()

            print("\n" + "="*80)
            print(f"{'Target (s)':<12} {'Meas (s)':<12} {'Slope (mV/s)':<15} {'I_dark (pA)':<15} {'Method':<10}")
            print("-" * 80)
            for r in results_table:
                print(f"{r['target']:<12.1f} {r['actual']:<12.3f} {r['slope']:<15.2f} {r['i']:<15.2f} {r['mode']:<10}")
            print("="*80)

            bench.siggen.set_output_state(1, "OFF")

    except Exception as e:
        print(f"ERROR: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    run_scope_referenced_audit()
