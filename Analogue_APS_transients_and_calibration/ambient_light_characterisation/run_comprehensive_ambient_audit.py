import numpy as np
import polars as pl
import os
import time
import sys
import matplotlib.pyplot as plt
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
    """Wait for explicit user confirmation in terminal."""
    print(f"\n[USER ACTION REQUIRED] >>> {message}")
    print("Type 'YES' to proceed (or 'Ctrl+C' to abort): ", end="")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline().strip().upper()
        if line == 'YES':
            break
        print("Invalid input. Please type 'YES' when ready: ", end="")
        sys.stdout.flush()

def capture_precision_sweep(bench, label, integration_times):
    """
    Captures integration ramps and returns parsed Delta-V and TTS results.
    """
    C_integrator = 11.0e-12 
    TTS_THRESHOLD_FRAC = 0.50 
    sweep_results = []
    
    for t_target in integration_times:
        freq = 1.0 / (2.0 * t_target)
        print(f"    -> Measuring {t_target}s window...")
        
        # 4x window margin for edge safety
        total_window = 4.0 * t_target
        trigger_pos = 0.2 * total_window
        bench.osc.set_time_axis(scale=total_window/10.0, position=trigger_pos)
        bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
        
        bench.siggen.set_frequency(1, freq)
        time.sleep(3.0) # Hardware stabilization
        
        bench.osc._send_command(":SINGle")
        time.sleep(total_window + 1.5)
        
        data = bench.osc.read_channels([1, 2])
        t = data.values["Time (s)"].to_numpy()
        v_px = data.values["Channel 1 (V)"].to_numpy()
        v_rs = data.values["Channel 2 (V)"].to_numpy()
        
        # Detect Edges
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
                
                sweep_results.append({
                    "target": t_target,
                    "actual": actual_T,
                    "dv": i_dv,
                    "tts": i_tts
                })
    return sweep_results

def main():
    print("====================================================")
    print("   AMBIENT LIGHT CHARACTERISATION SUITE v1.1        ")
    print("   Framework: PyTestLab | Methodology: TTS+DeltaV   ")
    print("====================================================\n")
    
    # 1. Setup absolute paths
    project_root = os.path.abspath("/home/coder/project/Analogue_APS_transients_and_calibration")
    script_dir = os.path.join(project_root, "ambient_light_characterisation")
    plot_dir = os.path.join(script_dir, "results/plots")
    
    os.chdir(project_root)
    os.makedirs(plot_dir, exist_ok=True)
    
    integration_times = [1.0, 2.0, 3.0, 4.0, 5.0]

    try:
        with Bench.open("config/bench.yaml") as bench:
            # Instrument Setup
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen.set_output_state(1, "ON")

            # --- PHASE 1: ENCLOSED ---
            user_prompt("Ensure photodiode is ENCLOSED (add Blu-Tack)")
            print("\n>>> MEASURING ENCLOSED BASELINE...")
            enclosed_data = capture_precision_sweep(bench, "ENCLOSED", integration_times)
            
            # --- PHASE 2: OPEN ---
            user_prompt("Ensure photodiode is UNCOVERED (remove Blu-Tack)")
            print("\n>>> MEASURING AMBIENT RESPONSE...")
            open_data = capture_precision_sweep(bench, "OPEN", integration_times)

            bench.siggen.set_output_state(1, "OFF")

        # --- DATA VISUALIZATION ---
        print("\n>>> GENERATING COMPARATIVE REPORT...")
        plt.figure(figsize=(12, 7))
        
        def plot_series(data, color, label_prefix):
            t = [r['target'] for r in data]
            dv = [r['dv'] for r in data]
            tts = [r['tts'] for r in data]
            
            plt.plot(t, dv, color=color, linestyle='--', alpha=0.5, label=f'{label_prefix}: Delta-V')
            
            valid_t = [t[i] for i in range(len(tts)) if tts[i] is not None]
            valid_tts = [i for i in tts if i is not None]
            if valid_tts:
                plt.plot(valid_t, valid_tts, color=color, marker='o', label=f'{label_prefix}: 50% TTS', linewidth=2)

        plot_series(enclosed_data, 'black', 'Enclosed')
        plot_series(open_data, 'red', 'Uncovered')

        plt.title("Ambient Light Response vs. Dark Baseline\n11pF Integrator | Dual-Method Consistency Check")
        plt.xlabel("Target Integration Time (s)")
        plt.ylabel("Derived Current (pA)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        
        report_path = os.path.join(plot_dir, "ambient_characterisation_report.png")
        plt.savefig(report_path)
        print(f"\n[DONE] Audit complete.")
        print(f"REPORT SAVED TO: {report_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
