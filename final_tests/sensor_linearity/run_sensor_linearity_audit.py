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

def run_linearity_sweep(bench, label, get_led_current, integration_times):
    C_integrator = 11.0e-12 
    v_led_steps = np.arange(2.7, 5.1, 0.1) 
    sweep_results = []

    for t_int in integration_times:
        freq = 1.0 / (2.0 * t_int)
        print(f"    - Sweeping T_int: {t_int*1000:.1f}ms...")
        
        # 1. Scope Recovery & Stabilization
        bench.osc.clear_status()
        time.sleep(1.0)
        
        # 2. Configure Timing (Rounded to avoid buffer bloat)
        total_window = float(np.round(4.0 * t_int, 4))
        time_div = float(np.round(total_window / 10.0, 5))
        trigger_pos = float(np.round(0.2 * total_window, 5))
        
        # Set individual parameters with slight delays for LAMB stability
        bench.osc._send_command(f":TIMebase:SCALe {time_div}")
        time.sleep(0.2)
        bench.osc._send_command(f":TIMebase:POSition {trigger_pos}")
        time.sleep(0.5)
        
        # 3. Configure Stimulus
        bench.siggen.set_frequency(1, float(np.round(freq, 3)))
        time.sleep(1.0)

        for v_led in v_led_steps:
            led_ma = get_led_current(v_led)
            bench.psu.channel(2).set(voltage=float(np.round(v_led, 2))).on()
            time.sleep(0.3)
            
            bench.osc._send_command(":SINGle")
            # Wait for hardware acquisition to finish physically
            time.sleep(total_window + 0.5)
            
            # Fetch Data
            try:
                data = bench.osc.read_channels([1, 2])
                df = data.values
                t_raw, v_px, v_rs = df["Time (s)"].to_numpy(), df["Channel 1 (V)"].to_numpy(), df["Channel 2 (V)"].to_numpy()
                
                # Edge detection
                edges = np.diff((v_rs > 2.5).astype(int))
                falls = np.where(edges == -1)[0]
                rises = np.where(edges == 1)[0]
                
                if len(falls) > 0 and len(rises) > 0:
                    f_idx, r_idx = falls[0], rises[rises > falls[0]][0]
                    v_start = np.mean(v_px[max(0, f_idx-10):f_idx])
                    v_final = np.mean(v_px[max(f_idx, r_idx-10):r_idx])
                    dt_actual = t_raw[r_idx] - t_raw[f_idx]
                    i_pd_total_pa = (C_integrator * (abs(v_start - v_final) / dt_actual)) * 1e12
                    
                    sweep_results.append({
                        "config": label,
                        "t_int_ms": round(t_int*1000, 2),
                        "i_led_ma": led_ma,
                        "i_pd_pa": i_pd_total_pa
                    })
            except Exception as e:
                print(f"      [!] Capture Error at {v_led}V: {e}")
                bench.osc.clear_status()
                time.sleep(1.0)

    return sweep_results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    I_BASELINE_CORRECTION = 16.0 
    
    print("====================================================")
    print("   LOW-LIGHT SENSOR SCALING AUDIT (HARDENED v1.3)   ")
    print(f"   Run ID: {timestamp}                             ")
    print("====================================================\n")
    
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    
    iv_path = "final_tests/led_iv_characterisation/results/data/led_iv_data_manual_mean.csv"
    plot_dir = "final_tests/sensor_linearity/results/plots"
    data_dir = "final_tests/sensor_linearity/results/data"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    iv_data = pl.read_csv(iv_path)
    def get_led_current(v): return np.interp(v, iv_data["v_supply"], iv_data["i_ma"])

    # 6 Integration Times: 1ms to 12ms
    integration_times = np.linspace(0.001, 0.012, 6)
    
    try:
        with Bench.open("config/bench.yaml") as bench:
            # Set high global timeout
            for inst in bench._instrument_instances.values():
                inst._backend.timeout_ms = 90000

            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            bench.siggen._send_command("OUTP1:LOAD INF")
            bench.siggen.set_function(1, "SQU")
            bench.siggen.set_amplitude(1, 5.0)
            bench.siggen.set_offset(1, 2.5)
            bench.siggen.set_output_state(1, "ON")

            bench.osc.channel(1).setup(scale=0.5, offset=2.5, coupling="DC").enable()
            bench.osc.channel(2).setup(scale=1.0, offset=2.5, coupling="DC").enable()

            user_prompt("Ensure exactly ONE photodiode is connected.")
            single_data = run_linearity_sweep(bench, "Single PD", get_led_current, integration_times)

            user_prompt("Now ADD the SECOND photodiode (Parallel).")
            dual_data = run_linearity_sweep(bench, "Dual PD", get_led_current, integration_times)

            bench.psu.channel(2).off()
            bench.siggen.set_output_state(1, "OFF")

        print("\n>>> GENERATING FINAL SCALING REPORT...")
        all_data = single_data + dual_data
        df = pl.DataFrame(all_data)
        df.write_csv(os.path.join(data_dir, f"pd_robust_data_{timestamp}.csv"))
        
        plt.figure(figsize=(14, 9))
        t_unique = sorted(df["t_int_ms"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(t_unique)))
        
        for config_idx, config in enumerate(["Single PD", "Dual PD"]):
            marker = 'o' if config == "Single PD" else 'x'
            ls = '-' if config == "Single PD" else '--'
            
            for i, t_ms in enumerate(t_unique):
                subset = df.filter((pl.col("config") == config) & (pl.col("t_int_ms") == t_ms))
                corrected_i = subset["i_pd_pa"] - I_BASELINE_CORRECTION
                plt.plot(subset["i_led_ma"], corrected_i, 
                         marker=marker, color=colors[i], linestyle=ls, markersize=4,
                         label=f"{config} (Tint={t_ms:.1f}ms)")

        plt.title(f"Distant-Stimulus Scaling Audit (1ms-12ms)\nCorrected for {I_BASELINE_CORRECTION}pA Baseline")
        plt.xlabel("LED Input Current (mA)")
        plt.ylabel("Photo-Current (pA)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(plot_dir, f"robust_scaling_report_{timestamp}.png")
        plt.savefig(save_path)
        print(f"\n[DONE] Audit complete. Report: {save_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__": main()
