"""
Generator Amplitude Sweep - Vout vs Gen Amplitude

Purpose:
  - Sweep WFG CH1 amplitude from 1V to 9V (Vpp).
  - Record Vout (RMS or Vpp) on OSC CH1.
  - Plot the results.
"""

import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pytestlab import Bench, MeasurementSession

# --- Configuration Constants ---
BENCH_CONFIG = "bench.yaml"
RESULT_IMAGE = "gen_sweep_osc.png"

# Sweep Parameters
AMP_START, AMP_STOP, AMP_STEP = 1.0, 9.0, 0.5
FREQUENCY = 1000 # 1kHz

def setup_instruments(bench: Bench):
    """Initializes hardware for amplitude sweep."""
    wfg = bench.wfg
    osc = bench.osc
    
    print(f"⚙️  Initializing Hardware...")
    print(f"  WFG: {wfg.id()}")
    print(f"  OSC: {osc.id()}")
    
    # 1. Setup Waveform Generator
    # Initialize at 1Vpp, 1kHz
    wfg.channel(1).setup_sine(frequency=FREQUENCY, amplitude=AMP_START).enable()
    
    # 2. Setup Oscilloscope
    # For a 1V-9V range, we might need to adjust scale dynamically, 
    # but starting with a scale that fits most of it.
    # 9Vpp means +/- 4.5V. 1V/div gives 8V range. 
    osc.set_channel_axis(1, scale=1.5, offset=0)
    osc.auto_scale()
    
    time.sleep(1.0)
    print("✅ Setup complete.")

def run_amplitude_sweep(bench: Bench) -> pl.DataFrame:
    """Executes the WFG amplitude sweep."""
    amp_steps = np.arange(AMP_START, AMP_STOP + 0.1, AMP_STEP)
    
    print(f"📈 Capturing amplitude sweep ({len(amp_steps)} points)...")
    
    with MeasurementSession(bench=bench, name="gen_amplitude_sweep") as session:
        session.parameter("amplitude", amp_steps, unit="Vpp")
        
        @session.acquire
        def capture_point(amplitude, wfg, osc):
            # Set WFG amplitude
            wfg.channel(1).setup_sine(frequency=FREQUENCY, amplitude=float(amplitude))
            
            # Settle
            time.sleep(0.5)
            
            # Measure Vout
            res_vpp = osc.measure_voltage_peak_to_peak(1)
            v_out_vpp = res_vpp.values.nominal_value if hasattr(res_vpp.values, 'nominal_value') else float(res_vpp.values)
            
            res_rms = osc.measure_rms_voltage(1)
            v_out_rms = res_rms.values.nominal_value if hasattr(res_rms.values, 'nominal_value') else float(res_rms.values)
            
            return {
                "v_out_vpp": v_out_vpp,
                "v_out_rms": v_out_rms
            }
        
        experiment = session.run(show_progress=True)
        return experiment.data

def plot_results(df: pl.DataFrame):
    """Generates the sweep plot."""
    print(f"📊 Rendering plot to '{RESULT_IMAGE}'...")
    
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(10, 6), dpi=120)
    
    plt.plot(df["amplitude"], df["v_out_vpp"], marker='o', linestyle='-', label="Vout (Vpp)")
    plt.plot(df["amplitude"], df["v_out_rms"], marker='s', linestyle='--', label="Vout (RMS)")
    
    plt.title("WFG Amplitude Sweep (OSC CH1 Capture)", fontsize=14, pad=15)
    plt.xlabel("WFG Amplitude [Vpp]", fontsize=12)
    plt.ylabel("Measured Voltage [V]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(RESULT_IMAGE)
    print(f"✅ Plot saved to {RESULT_IMAGE}.")

def main():
    try:
        with Bench.open(BENCH_CONFIG) as bench:
            setup_instruments(bench)
            data = run_amplitude_sweep(bench)
            plot_results(data)
            
            print("🔌 Disabling outputs...")
            bench.wfg.channel(1).disable()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
