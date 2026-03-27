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

def main():
    # --- USER PARAMETER ---
    FREQ = 30.0  # Set your desired Reset Frequency here
    # ----------------------

    T_INT_PERIOD = 1.0 / FREQ
    T_ACTIVE_INTEG = T_INT_PERIOD / 2.0
    
    print("====================================================")
    print(f"   ACTIVE PIXEL: CLEAN OPTICAL FAMILY SWEEP        ")
    print(f"   Reset Frequency: {FREQ} Hz | Step: 0.2V          ")
    print("====================================================\n")
    
    project_root = os.path.abspath("/home/coder/project/Analogue_APS_transients_and_calibration")
    plot_dir = os.path.join(project_root, "optical_linearity_sweep/results/plots")
    os.chdir(project_root)
    os.makedirs(plot_dir, exist_ok=True)
        # LED Voltage Sweep (PSU CH2) 
        # Starting from 2.7V per calibration results (Linear Region)
    v_led_steps = np.arange(2.7, 5.1, 0.2) 
    
    waveforms = []
    reference_reset = None

    try:
        with Bench.open("config/bench.yaml") as bench:
            print(">>> INITIALIZING HARDWARE...")
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            
            # SigGen CH1: Reset Pulse
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen.set_frequency(1, FREQ)
            bench.siggen.set_output_state(1, "ON")

            # Adaptive Scope Setup
            total_window = 2.0 / FREQ
            time_div = total_window / 10.0
            trigger_pos = 0.2 * total_window 
            
            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()
            bench.osc.set_time_axis(scale=time_div, position=trigger_pos) 
            
            # Trigger on Reset Fall (CH2)
            bench.osc._send_command(":TRIGger:MODE EDGE")
            bench.osc._send_command(":TRIGger:EDGE:SOURce CHANnel2")
            bench.osc._send_command(":TRIGger:EDGE:SLOPe NEGative")
            bench.osc._send_command(":TRIGger:LEVel 2.5, CHANnel2")

            print(f">>> CAPTURING CLEAN FAMILY ({len(v_led_steps)} Traces)...")
            
            for v_led in v_led_steps:
                print(f"    - LED Stimulus: {v_led:.2f} V", end="\r")
                bench.psu.channel(2).set(voltage=v_led).on()
                time.sleep(0.4) 
                
                bench.osc._send_command(":SINGle")
                time.sleep(total_window + 0.2) 
                
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_raw = df["Time (s)"].to_numpy()
                v_px = df["Channel 1 (V)"].to_numpy()
                v_rs = df["Channel 2 (V)"].to_numpy()
                
                # Alignment Logic
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                
                if len(falls) > 0:
                    f_idx = falls[0]
                    t0 = t_raw[f_idx]
                    dt = t_raw[1] - t_raw[0]
                    points_to_keep = int((T_ACTIVE_INTEG * 1.1) / dt)
                    end_idx = min(f_idx + points_to_keep, len(t_raw))
                    
                    waveforms.append({
                        "label": f"{v_led:.1f}V",
                        "t": t_raw[f_idx : end_idx] - t0,
                        "v": v_px[f_idx : end_idx]
                    })
                    
                    if reference_reset is None:
                        reference_reset = {
                            "t": t_raw[f_idx : end_idx] - t0,
                            "v": v_rs[f_idx : end_idx]
                        }

            bench.psu.channel(2).off()
            bench.siggen.set_output_state(1, "OFF")

        # --- FINAL OVERLAY PLOTTING ---
        print("\n\n>>> GENERATING CLEAN OVERLAY REPORT...")
        plt.figure(figsize=(10, 6))
        
        # Scale units
        t_scale, t_unit = (1e3, "ms") if T_ACTIVE_INTEG < 1.0 else (1.0, "s")

        if reference_reset is not None:
            plt.plot(reference_reset["t"]*t_scale, reference_reset["v"], 'k--', alpha=0.15, label='Reset Ref')

        # Use discrete colors for high clarity
        colors = plt.cm.viridis(np.linspace(0, 1, len(waveforms)))
        for i, wave in enumerate(waveforms):
            plt.plot(wave["t"]*t_scale, wave["v"], color=colors[i], label=wave["label"], linewidth=1.2)

        plt.title(f"Active Pixel: Clean Family of Curves (Step: 0.2V)\nReset: {FREQ}Hz | Integration: {T_ACTIVE_INTEG*t_scale:.1f}{t_unit}")
        plt.xlabel(f"Time from Reset Release ({t_unit})")
        plt.ylabel("Sensor Output (V)")
        plt.grid(True, alpha=0.3)
        plt.legend(title="LED Voltage", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        report_path = os.path.join(plot_dir, "clean_family_curves.png")
        plt.savefig(report_path)
        print(f"[DONE] Sweep complete. Report: {report_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
