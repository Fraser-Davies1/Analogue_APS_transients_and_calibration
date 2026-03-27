import numpy as np
import polars as pl
from pytestlab import Bench
import time

# --- Patch for registry ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except: pass
# --------------------------

def vertical_search():
    print("--- Metrology Vertical Search ---")
    with Bench.open("bench.yaml") as bench:
        # 1. Ensure PSU is correct
        print("  Setting PSU VDD=5.0V...")
        bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
        time.sleep(0.5)
        v_act = bench.psu.read_voltage(1)
        i_act = bench.psu.read_current(1)
        print(f"  PSU CH1: {v_act:.3f}V, {i_act*1000:.2f}mA")
        
        # 2. Ensure SigGen is resetting
        print("  Setting SigGen Reset Pulse (500Hz, 0-5V)...")
        bench.siggen._send_command("SOUR1:FUNC SQU")
        bench.siggen._send_command("SOUR1:FREQ 500")
        bench.siggen._send_command("SOUR1:VOLT 5.0")
        bench.siggen._send_command("SOUR1:VOLT:OFFS 2.5")
        bench.siggen._send_command("OUTP1:STAT ON")
        
        # 3. Find Signal on Scope
        print("  Probing Scope CH1...")
        # Start wide
        bench.osc.channel(1).setup(scale=1.0, offset=2.5, coupling="DC").enable()
        bench.osc.channel(3).setup(scale=1.0, offset=2.5, coupling="DC").enable() # Trigger ref
        bench.osc.set_time_axis(scale=500e-6, position=2.5e-3)
        
        time.sleep(1.0)
        data = bench.osc.read_channels([1, 3])
        v1 = data.values["Channel 1 (V)"].to_numpy()
        v3 = data.values["Channel 3 (V)"].to_numpy()
        
        print(f"  CH1 (Pixel) Range: {np.min(v1):.3f}V to {np.max(v1):.3f}V")
        print(f"  CH1 (Pixel) Mean:  {np.mean(v1):.3f}V")
        print(f"  CH3 (Reset) Range: {np.min(v3):.3f}V to {np.max(v3):.3f}V")
        
        if np.max(v3) < 1.0:
            print("  [WARN] No reset pulse detected on CH3. Check siggen cable.")
            
        if np.std(v1) < 0.001:
            print("  [WARN] Pixel output is perfectly flat. Is the board biased?")
            
        # Refine for Noise
        if np.mean(v1) > 0.1:
            target_offset = np.mean(v1)
            print(f"  Refining to target offset: {target_offset:.3f}V")
            bench.osc.channel(1).setup(scale=0.01, offset=target_offset).enable()
            time.sleep(0.5)
            data_noise = bench.osc.read_channels([1])
            v_noise = data_noise.values["Channel 1 (V)"].to_numpy()
            print(f"  Refined RMS: {np.std(v_noise)*1e3:.3f} mV")
            print(f"  Unique ADC levels: {len(np.unique(v_noise))}")
        else:
            print("  [ERROR] Signal is near ground. Noise characterization aborted.")

if __name__ == "__main__":
    vertical_search()
