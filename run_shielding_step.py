import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from pytestlab import Bench
import time
import sys
import os

# --- Framework Patch: Register WaveformGeneratorConfig ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except Exception: pass
# ---------------------------------------------------------

def capture_psd(bench, num_frames, siggen_on):
    if siggen_on:
        bench.siggen._send_command("SOUR1:FUNC SQU")
        bench.siggen._send_command("SOUR1:FREQ 500")
        bench.siggen._send_command("SOUR1:VOLT 5.0")
        bench.siggen._send_command("SOUR1:VOLT:OFFS 2.5")
        bench.siggen._send_command("OUTP1:STAT ON")
    else:
        bench.siggen._send_command("OUTP1:STAT OFF")
    
    bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="AC").enable()
    bench.osc.set_time_axis(scale=20e-3, position=100e-3)
    time.sleep(0.5)
    
    psd_accumulator = []
    
    for i in range(num_frames):
        bench.osc._send_command(":DIGitize CHANnel1")
        data = bench.osc.read_channels([1])
        v = data.values["Channel 1 (V)"].to_numpy()
        t = data.values["Time (s)"].to_numpy()
        fs_actual = 1.0 / (t[1] - t[0])
        
        n_fft = 524288 if len(v) >= 524288 else len(v)
        f, psd = signal.welch(v, fs_actual, 
                              window='blackmanharris',
                              nperseg=n_fft, 
                              detrend=False, 
                              scaling='density')
        psd_accumulator.append(psd)
    
    return f, np.mean(psd_accumulator, axis=0)

def run_step(label, siggen_on):
    project_root = "/home/coder/project/Analogue_APS_transients_and_calibration"
    os.chdir(project_root)
    os.makedirs("results/data", exist_ok=True)
    
    print(f"MEASURING: {label}")
    sys.stdout.flush()
    
    try:
        with Bench.open("config/bench.yaml") as bench:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.1).on()
            
            f, psd = capture_psd(bench, num_frames=5, siggen_on=siggen_on)
            
            data_path = f"results/data/{label}_hr.npz"
            np.savez(data_path, f=f, psd=psd)
            
            rms = np.sqrt(integrate.trapezoid(psd, f))
            print(f"SUCCESS: Integrated RMS = {rms*1e3:.3f} mV")
            sys.stdout.flush()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <label> <siggen_bool>")
    else:
        run_step(sys.argv[1], sys.argv[2].lower() == 'true')
