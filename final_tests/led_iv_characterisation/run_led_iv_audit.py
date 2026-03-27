import numpy as np
import polars as pl
from pytestlab import Bench
import time
import matplotlib.pyplot as plt
import os
import sys

# --- Framework Patch: Register WaveformGeneratorConfig ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass
# ---------------------------------------------------------

def run_led_iv_manual_mean():
    print("====================================================")
    print("   LED I-V CHARACTERISATION (MANUAL BUFFER MEAN)    ")
    print("   Stimulus: PSU CH2 | Resistor Measurement: CH3    ")
    print("====================================================\n")
    
    # 1. Setup absolute paths
    project_root = os.path.abspath("/home/coder/project/Analogue_APS_transients_and_calibration")
    script_dir = os.path.join(project_root, "final_tests/led_iv_characterisation")
    plot_dir = os.path.join(script_dir, "results/plots")
    data_dir = os.path.join(script_dir, "results/data")
    
    os.chdir(project_root)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    try:
        with Bench.open("config/bench.yaml") as bench:
            # 1. Hardware Initialization
            print(">>> Initializing PSU Rails...")
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            
            # 2. Scope Configuration
            # Set a clear scale (e.g. 500mV/div) for the resistor drop
            print(">>> Configuring Scope CH3...")
            bench.osc.channel(3).setup(scale=0.5, offset=1.0, coupling="DC").enable()
            bench.osc.set_time_axis(scale=500e-6, position=2.5e-3) # 5ms window
            time.sleep(1)

            # Sweep LED Supply Voltage
            v_supply_sweep = np.arange(0.0, 5.1, 0.1) 
            results = []
            R_SENSE = 220.0 # Ohm

            print(f">>> STARTING SWEEP...")
            for v_supply in v_supply_sweep:
                bench.psu.channel(2).set(voltage=v_supply).on()
                time.sleep(0.4) 
                
                # Capture frame
                bench.osc._send_command(":SINGle")
                time.sleep(0.1)
                
                try:
                    # Read the raw buffer and calculate mean manually
                    data = bench.osc.read_channels([3])
                    v_buffer = data.values["Channel 3 (V)"].to_numpy()
                    v_resistor = np.mean(v_buffer)
                except:
                    v_resistor = 0.0
                
                # I = V_res / R_sense
                i_ma = (v_resistor / R_SENSE) * 1000.0
                
                results.append({
                    "v_supply": v_supply, 
                    "v_resistor": v_resistor,
                    "i_ma": i_ma
                })
                print(f"    - Supply: {v_supply:.1f}V | V_mean: {v_resistor:.3f}V | I: {i_ma:.3f}mA", end="\r")

            bench.psu.channel(2).off()
            
            # 3. Save Data
            df = pl.DataFrame(results)
            csv_path = os.path.join(data_dir, "led_iv_data_manual_mean.csv")
            df.write_csv(csv_path)
            
            # 4. Plot Results
            print("\n\n>>> GENERATING MANUAL MEAN I-V PLOT...")
            plt.figure(figsize=(10, 6))
            plt.plot(df["v_supply"], df["i_ma"], 'bo-', linewidth=1.5, markersize=4, label='Manual Buffer Mean')
            plt.title(f"LED I-V Characteristic (Manual Buffer Mean)\nMeasured via 220Ω Resistor (CH3)")
            plt.xlabel("LED Supply Voltage (V)")
            plt.ylabel("Inferred LED Current (mA)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plot_path = os.path.join(plot_dir, "led_iv_report_manual_mean.png")
            plt.savefig(plot_path)
            
            print(f"[DONE] Audit complete.")
            print(f"REPORT SAVED TO: {plot_path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_led_iv_manual_mean()
