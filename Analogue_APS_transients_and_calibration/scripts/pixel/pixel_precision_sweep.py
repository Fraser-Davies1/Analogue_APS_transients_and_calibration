from pytestlab import Bench, MeasurementSession
from pytestlab.measurements.steps import step
import numpy as np
import time
import polars as pl

def run_precision_experiment():
    print("--- Opening Bench (Precision Timing Mode) ---")
    with Bench.open("bench.yaml") as bench:
        # 1. Hardware Initialization
        print("--- Configuring VDD=5V, Reset=5V (0-5V) @ 500Hz ---")
        try:
            bench.psu.channel(1).set(voltage=5.0, current_limit=0.05).on()
            # 0 to 5V Square Wave
            bench.siggen.channel(1).setup_square(frequency=500, amplitude=5.0, offset=2.5).enable()
            
            # OSC Setup: 500us/div -> 5ms total window (enough for 2.5 cycles of 500Hz)
            bench.osc.set_time_axis(scale=500e-6, position=2.0e-3)
            bench.osc.channel(1).setup(scale=0.5, offset=2.5).enable()
            bench.osc.channel(3).setup(scale=1.0, offset=2.5).enable()
            bench.osc.trigger.setup_edge(source="CH3", level=2.5)
            print("  [OK] Hardware timing and rails set.")
        except Exception as e:
            print(f"  [ERROR] Setup failed: {e}")

        print("--- Starting Sweep ---")
        with MeasurementSession(bench=bench, name="Pixel_Precision_Linearity") as session:
            session.parameter("v_led", step.linear(2.3, 4.5, 20), unit="V")
            
            @session.acquire
            def measure(v_led, psu, osc):
                try:
                    psu.channel(2).set_voltage(v_led).on()
                except: pass
                
                time.sleep(1.0) # Settle
                
                # Capture raw waveform data
                reading = osc.read_channels([1, 3])
                df = reading.values # Polars DataFrame
                
                # 1. Identify Edges on CH3 (Reset Signal)
                # Find falling edge (Integration Start) and rising edge (Integration End)
                ch3 = df["Channel 3 (V)"].to_numpy()
                ch1 = df["Channel 1 (V)"].to_numpy()
                
                # Simple thresholding
                high_mask = ch3 > 2.5
                edges = np.diff(high_mask.astype(int))
                
                falling_indices = np.where(edges == -1)[0]
                rising_indices = np.where(edges == 1)[0]
                
                if len(falling_indices) > 0 and len(rising_indices) > 0:
                    # Find a pair where falling occurs before rising
                    start_idx = falling_indices[0]
                    # Find the first rising edge after this falling edge
                    end_indices = rising_indices[rising_indices > start_idx]
                    
                    if len(end_indices) > 0:
                        end_idx = end_indices[0]
                        
                        # Sample "just before" (e.g., 5 samples prior)
                        v_initial = ch1[max(0, start_idx - 5)]
                        v_final = ch1[max(0, end_idx - 5)]
                        delta_v = v_final - v_initial
                        
                        return {
                            "v_led": v_led,
                            "v_pixel_start": v_initial,
                            "v_pixel_end": v_final,
                            "v_delta": delta_v
                        }
                
                return {"v_led": v_led, "v_delta": np.nan}
            
            experiment = session.run(show_progress=True)
            
        filename = "precision_linearity_results.csv"
        experiment.data.write_csv(filename)
        print(f"\n--- Success! Results saved to {filename} ---")

if __name__ == "__main__":
    run_precision_experiment()
