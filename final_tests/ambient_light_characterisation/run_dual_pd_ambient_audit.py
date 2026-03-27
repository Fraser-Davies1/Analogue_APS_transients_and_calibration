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
    """
    Blocks execution until the user manually types 'YES'.
    """
    print(f"\n{'='*60}")
    print(f"  [USER ACTION REQUIRED]")
    print(f"  >>> {message}")
    print(f"{'='*60}")
    
    while True:
        try:
            # use input() for better terminal interaction handling
            user_in = input("\nType 'YES' and press ENTER to proceed: ").strip().upper()
            if user_in == 'YES':
                print("Confirmed. Proceeding...\n")
                break
            else:
                print(f"Received '{user_in}'. Please type exactly 'YES' to continue.")
        except EOFError:
            # Handle non-interactive environments
            time.sleep(1)
            continue

def capture_precision_sweep(bench, label, integration_times):
    """
    Captures integration ramps using a stable 4x windowing strategy.
    """
    C_integrator = 11.0e-12 
    TTS_THRESHOLD_FRAC = 0.50 
    sweep_results = []
    
    print(f"  --- Characterizing: {label} ---")
    for t_target in integration_times:
        freq = float(np.round(1.0 / (2.0 * t_target), 4))
        print(f"    -> Measuring {t_target}s window...")
        
        # Scope Config
        total_window = 4.0 * t_target
        trigger_pos = 0.2 * total_window
        bench.osc.set_time_axis(scale=total_window/10.0, position=trigger_pos)
        
        # Update timeout for long acquisitions
        if hasattr(bench.osc._backend, "set_timeout"):
            bench.osc._backend.set_timeout(int(total_window * 1000 + 40000))
        
        bench.siggen.set_frequency(1, freq)
        time.sleep(1.0) # Reduced from 3.0 for demo
        
        bench.osc._send_command(":SINGle")
        time.sleep(total_window + 0.5)
        
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
    print("   DUAL PHOTODIODE AMBIENT LIGHT AUDIT             ")
    print("====================================================\n")
    
    project_root = os.getcwd()
    plot_dir = os.path.join(project_root, "final_tests/ambient_light_characterisation/results/plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Dynamic integration times for the audit
    integration_times = [0.2, 0.5, 1.0, 2.0, 5.0]

    try:
        with Bench.open("bench.yaml") as bench:
            # Global Init
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen.set_output_state(1, "ON")

            # --- CONFIGURATION 1: SINGLE PHOTODIODE ---
            print("\n>>> CONFIGURATION 1: SINGLE PHOTODIODE")
            user_prompt("Setup SINGLE photodiode (ensure it is the only one in circuit)")
            
            user_prompt("Ensure photodiode is ENCLOSED (Dark Baseline)")
            pd1_enclosed = capture_precision_sweep(bench, "1PD_ENCLOSED", integration_times)
            
            user_prompt("Ensure photodiode is UNCOVERED (Ambient Stimulus)")
            pd1_open = capture_precision_sweep(bench, "1PD_OPEN", integration_times)

            # --- CONFIGURATION 2: DUAL PHOTODIODE ---
            print("\n>>> CONFIGURATION 2: DUAL PHOTODIODE")
            user_prompt("ADD the SECOND photodiode in parallel to the first")
            
            user_prompt("Ensure BOTH photodiodes are ENCLOSED (Dark Baseline)")
            pd2_enclosed = capture_precision_sweep(bench, "2PD_ENCLOSED", integration_times)
            
            user_prompt("Ensure BOTH photodiodes are UNCOVERED (Ambient Stimulus)")
            pd2_open = capture_precision_sweep(bench, "2PD_OPEN", integration_times)

            bench.siggen.set_output_state(1, "OFF")
            bench.psu.channel(1).off()

        # Plotting Results
        plt.figure(figsize=(12, 8))
        def plot_s(data, color, label, marker='o'):
            if not data: return
            t = [r['target'] for r in data]
            dv = [r['dv'] for r in data]
            tts = [r['tts'] for r in data]
            
            plt.plot(t, dv, color=color, linestyle='--', alpha=0.4, label=f'{label} (ΔV)')
            vt = [t[i] for i in range(len(tts)) if tts[i] is not None]
            vtts = [i for i in tts if i is not None]
            if vtts:
                plt.plot(vt, vtts, color=color, marker=marker, label=f'{label} (TTS)', linewidth=2)

        plot_s(pd1_enclosed, 'gray', '1PD Enclosed', marker='x')
        plot_s(pd1_open, 'red', '1PD Open', marker='o')
        plot_s(pd2_enclosed, 'black', '2PD Enclosed', marker='s')
        plot_s(pd2_open, 'blue', '2PD Open', marker='D')

        plt.title(f"Dual PD Ambient Response Characterization\nRun ID: {timestamp}")
        plt.xlabel("Integration Window (s)")
        plt.ylabel("Derived Current (pA)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        report_path = os.path.join(plot_dir, f"dual_pd_ambient_report_{timestamp}.png")
        plt.savefig(report_path)
        plt.close()
        
        print(f"\n[DONE] Audit complete.")
        print(f"Report saved to: {report_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
