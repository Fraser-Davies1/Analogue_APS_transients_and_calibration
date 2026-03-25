from pytestlab import Bench
import numpy as np

# --- Patch for registry ---
from pytestlab.config.loader import get_model_registry
from pytestlab.config import WaveformGeneratorConfig
try:
    registry = get_model_registry()
    if "waveform_generator" not in registry:
        registry["waveform_generator"] = WaveformGeneratorConfig
except: pass
# --------------------------

def audit_hardware():
    print("--- Metrology Hardware Audit ---")
    try:
        with Bench.open("bench.yaml") as bench:
            print(f"Bench connected: {bench.config.bench_name}")
            print(f"Simulation Mode: {bench.config.simulate}")
            
            for alias, inst in bench._instrument_instances.items():
                try:
                    idn = inst.id()
                    backend_type = type(inst._backend).__name__
                    print(f"  [{alias}] IDN: {idn}")
                    print(f"  [{alias}] Backend: {backend_type}")
                    
                    if "Sim" in backend_type:
                        print(f"  [WARN] {alias} is using SIMULATION backend.")
                except Exception as e:
                    print(f"  [{alias}] Communication Failed: {e}")
                    
            # Check OSC noise floor directly
            print("\n--- OSC Noise Floor Probe ---")
            try:
                bench.osc.channel(1).setup(scale=0.01, offset=0, coupling="DC").enable()
                data = bench.osc.read_channels([1])
                v = data.values["Channel 1 (V)"].to_numpy()
                print(f"  OSC CH1 RMS: {np.std(v)*1e3:.3f} mV")
                print(f"  OSC CH1 Mean: {np.mean(v)*1e3:.3f} mV")
                print(f"  Unique ADC levels: {len(np.unique(v))}")
            except Exception as e:
                print(f"  OSC Probe Failed: {e}")
                
    except Exception as e:
        print(f"Failed to open bench: {e}")

if __name__ == "__main__":
    audit_hardware()
