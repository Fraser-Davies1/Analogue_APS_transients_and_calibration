import numpy as np
import polars as pl
import os
import time
import sys
import matplotlib.pyplot as plt
from datetime import datetime
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

def user_prompt(message):
    print(f"\n[USER ACTION REQUIRED] >>> {message}")
    print("Type 'YES' to proceed: ", end="")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline().strip().upper()
        if line == 'YES':
            break
        print("Invalid input. Please type 'YES' when ready: ", end="")
        sys.stdout.flush()

def capture_precision_sweep(bench, label, integration_times):
    """
    Captures integration ramps using a stable 4x windowing strategy.
    """
    C_integrator = 11.0e-12 
    TTS_THRESHOLD_FRAC = 0.50 
    sweep_results = []
    
    for t_target in integration_times:
        freq = float(np.round(1.0 / (2.0 * t_target), 4))
        print(f"    -> Measuring {t_target}s window...")
        
        # Scope Config
        total_window = 4.0 * t_target
        trigger_pos = 0.2 * total_window
        bench.osc.set_time_axis(scale=total_window/10.0, position=trigger_pos)
        bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
        
        bench.siggen.set_frequency(1, freq)
        time.sleep(3.0) 
        
        bench.osc._send_command(":SINGle")
        time.sleep(total_window + 1.5)
        
        try:
            data = bench.osc.read_channels([1, 2])
            df = data.values
            t = df["Time (s)"].to_numpy()
            v_px = df["Channel 1 (V)"].to_numpy()
            v_rs = df["Channel 2 (V)"].to_numpy()
            
            # --- Analysis ---
            edges = np.diff((v_rs > 2.5).astype(int))
            falls = np.where(edges == -1)[0]
            rises = np.where(edges == 1)[0]
            
            if len(falls) > 0 and len(rises) > 0:
                f_idx = falls[0]
                r_pts = rises[rises > f_idx]
                if len(r_pts) > 0:
                    r_idx = r_pts[0]
                    t0, t_end = t[f_idx], t[r_idx]
                    actual_T = t_end - t0
                    
                    v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                    v_final = np.mean(v_px[max(f_idx, r_idx-10):r_idx])
                    v_target_50 = v_start * (1.0 - TTS_THRESHOLD_FRAC)
                    
                    # 1. Delta-V
                    i_dv = abs((v_start - v_final) / actual_T) * C_integrator * 1e12
                    
                    # 2. 50% TTS
                    window_v, window_t = v_px[f_idx:r_idx], t[f_idx:r_idx]
                    sat_indices = np.where(window_v <= v_target_50)[0]
                    i_tts = None
                    if len(sat_indices) > 0:
                        t_tts = window_t[sat_indices[0]] - t0
                        i_tts = abs((v_start - v_target_50) / t_tts) * C_integrator * 1e12
                    
                    print(f"       Result: ΔV={i_dv:.2f}pA | TTS={i_tts if i_tts else 'N/A'}")
                    sweep_results.append({"target": t_target, "dv": i_dv, "tts": i_tts})
                else:
                    print("       [!] Rising edge not captured.")
            else:
                print("       [!] Pulse edges not found.")
                
        except Exception as e:
            print(f"       [!] Capture failed: {e}")
            bench.osc.clear_status()
            
    return sweep_results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("====================================================")
    print("   AMBIENT LIGHT AUDIT (STABLE 5s SWEEP)            ")
    print("====================================================\n")
    
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    plot_dir = "final_tests/ambient_light_characterisation/results/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    integration_times = [1.0, 2.0, 3.0, 4.0, 5.0]

    try:
        with Bench.open("config/bench.yaml") as bench:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen.set_output_state(1, "ON")

            # PHASE 1: ENCLOSED
            user_prompt("Ensure photodiode is ENCLOSED (Blu-Tack ON)")
            enclosed_data = capture_precision_sweep(bench, "ENCLOSED", integration_times)
            
            # PHASE 2: OPEN
            user_prompt("Ensure photodiode is UNCOVERED (Blu-Tack OFF)")
            open_data = capture_precision_sweep(bench, "OPEN", integration_times)

            bench.siggen.set_output_state(1, "OFF")

        # Plotting
        plt.figure(figsize=(12, 7))
        def plot_s(data, color, label):
            t = [r['target'] for r in data]
            dv = [r['dv'] for r in data]
            tts = [r['tts'] for r in data]
            plt.plot(t, dv, color=color, linestyle='--', alpha=0.5, label=f'{label}: ΔV')
            vt = [t[i] for i in range(len(tts)) if tts[i] is not None]
            vtts = [i for i in tts if i is not None]
            if vtts: plt.plot(vt, vtts, color=color, marker='o', label=f'{label}: TTS', linewidth=2)

        plot_s(enclosed_data, 'black', 'Enclosed')
        plot_s(open_data, 'red', 'Uncovered')

        plt.title(f"Ambient Response vs Dark Baseline (1s-5s)\nRun ID: {timestamp}")
        plt.xlabel("Integration Window (s)")
        plt.ylabel("Derived Current (pA)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        report_path = os.path.join(plot_dir, f"stable_ambient_report_{timestamp}.png")
        plt.savefig(report_path)
        print(f"\n[DONE] Audit complete. Report saved to: {report_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__": main()
