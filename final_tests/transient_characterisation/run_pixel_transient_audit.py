import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
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
    print("Type 'YES' to proceed: ", end="")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline().strip().upper()
        if line == 'YES':
            break
        print("Invalid input. Please type 'YES' when ready: ", end="")
        sys.stdout.flush()

def main():
    print("====================================================")
    print("   PIXEL TRANSIENT CHARACTERISATION SUITE v1.0      ")
    print("   Reference: CH2 (Reset) | Sensor: CH1             ")
    print("====================================================\n")
    
    # 1. Setup absolute paths
    project_root = os.path.abspath("/home/coder/project/Analogue_APS_transients_and_calibration")
    plot_dir = os.path.join(project_root, "transient_characterisation/results/plots")
    
    os.chdir(project_root)
    os.makedirs(plot_dir, exist_ok=True)
    
    integration_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    C_integrator = 11.0e-12 
    TTS_THRESHOLD_FRAC = 0.50 # 50% depletion

    try:
        with Bench.open("config/bench.yaml") as bench:
            # Setup Stimulus
            print(">>> Configuring Instruments...")
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen.set_output_state(1, "ON")

            # Setup Scope
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
            
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            user_prompt("Ensure photodiode is in your desired test state (Enclosed or Open)")

            for t_target in integration_times:
                freq = 1.0 / (2.0 * t_target)
                print(f"\n>>> AUDITING TRANSIENT: {t_target}s (Freq: {freq:.3f} Hz)")
                
                # 4x Margin for edge safety
                total_window = 4.0 * t_target
                trigger_pos = 0.2 * total_window
                bench.osc.set_time_axis(scale=total_window/10.0, position=trigger_pos)
                bench.osc._backend.timeout_ms = int(total_window * 1000 + 40000)
                
                bench.siggen.set_frequency(1, freq)
                time.sleep(3.0) # Hardware stabilization
                
                print(f"    Capturing {total_window:.1f}s integration frame...")
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 1.5)
                
                # Fetch Data
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_raw = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 2 (V)"].to_numpy()
                
                # --- Analysis Logic ---
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    f_idx = falls[0]
                    r_pts = rises[rises > f_idx]
                    
                    if len(r_pts) > 0:
                        r_idx = r_pts[0]
                        t0, t_end = t_raw[f_idx], t_raw[r_idx]
                        actual_T = t_end - t0
                        
                        v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                        v_final = np.mean(v_px[max(f_idx, r_idx-10):r_idx])
                        v_target_50 = v_start * (1.0 - TTS_THRESHOLD_FRAC)
                        
                        # 1. Delta-V Method
                        i_dv = abs((v_start - v_final) / actual_T) * C_integrator * 1e12
                        
                        # 2. 50% TTS Method
                        window_v, window_t = v_px[f_idx:r_idx], t_raw[f_idx:r_idx]
                        sat_indices = np.where(window_v <= v_target_50)[0]
                        i_tts = None
                        if len(sat_indices) > 0:
                            t_tts = window_t[sat_indices[0]] - t0
                            i_tts = abs((v_start - v_target_50) / t_tts) * C_integrator * 1e12

                        print(f"    - Measured T: {actual_T:.3f}s")
                        print(f"    - Delta-V I:  {i_dv:.2f} pA")
                        print(f"    - 50% TTS I:  {i_tts if i_tts else 'N/A'}")

                        # Plotting
                        plt.figure(figsize=(10, 5))
                        plt.plot(t_raw - t0, v_px, 'k', label='Sensor Output (CH1)')
                        plt.plot(t_raw - t0, v_rs, 'b--', alpha=0.3, label='Reset Signal (CH2)')
                        plt.axvspan(0, actual_T, color='green', alpha=0.1, label='Active Integration')
                        if i_tts:
                            plt.axvline(t_raw[f_idx + sat_indices[0]] - t0, color='r', linestyle='--', label='TTS 50% Point')
                        
                        plt.title(f"Transient Response: {t_target}s Window\nDerived I: {i_dv:.1f}pA (Delta-V)")
                        plt.xlabel("Time from Reset Release (s)")
                        plt.ylabel("Integrator Voltage (V)")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        save_name = os.path.join(plot_dir, f"transient_audit_{t_target}s.png")
                        plt.savefig(save_name)
                        plt.close()
                    else:
                        print("    [!] End of integration cycle not found in buffer.")
                else:
                    print("    [!] Reset edges not found. Increase window or check wiring.")

            bench.siggen.set_output_state(1, "OFF")
            print(f"\n[DONE] Transient audit complete. Plots saved to: {plot_dir}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
