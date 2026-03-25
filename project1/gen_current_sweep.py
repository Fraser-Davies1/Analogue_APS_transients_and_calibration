import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pytestlab import Bench, MeasurementSession

# --- Configuration ---
BENCH_CONFIG = "bench.yaml"
RESULT_IMAGE = "iv_characteristic_set_current.png"
RESISTANCE = 220.0  # Ohms

# Sweep Parameters
AMP_START, AMP_STOP, STEPS = 0.0, 9.0, 4

def setup_instruments(bench: Bench):
    wfg, osc = bench.wfg, bench.osc
    print(f"⚙️  Initializing Hardware (Ideal Stimulus Mode)...")
    
    wfg.channel(1).set_load_impedance("INFinity")
    wfg.channel(1).setup_dc(offset=0.0).enable()
    
    # Setup OSC for 0-10V range
    osc.set_channel_axis(1, scale=1.25, offset=5.0)
    time.sleep(1.0)

def run_sweep(bench: Bench) -> pl.DataFrame:
    dc_steps = np.linspace(AMP_START, AMP_STOP, STEPS)
    
    print(f"📈 Capturing IV curve: X-axis = Vgen/220...")
    
    with MeasurementSession(bench=bench, name="iv_set_current") as session:
        session.parameter("v_gen_set", dc_steps, unit="V")
        
        @session.acquire
        def capture(v_gen_set, wfg, osc):
            # 1. Apply Set Voltage
            set_val = float(v_gen_set)
            wfg.channel(1).setup_dc(offset=set_val)
            time.sleep(0.8) 
            
            # 2. Measure Live Output
            res = osc.measure_rms_voltage(1)
            v_out = res.values.nominal_value if hasattr(res.values, 'nominal_value') else float(res.values)
            
            # 3. Calculate Current from SETPOINT
            i_ma_ideal = (set_val / RESISTANCE) * 1000 
            
            return {
                "v_out": v_out,
                "i_ma_set": i_ma_ideal,
                "v_gen_target": set_val
            }
        
        return session.run(show_progress=True).data

def plot_results(df: pl.DataFrame):
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(10, 6), dpi=120)
    
    # Plot: X = Calculated from Setpoint, Y = Measured Output
    plt.plot(df["i_ma_set"], df["v_out"], marker='o', markersize=12, 
             linestyle='-', linewidth=3, color='#16a085', label=f"R_ext = {RESISTANCE}Ω")
    
    plt.title(f"I-V Characteristic: Output Voltage vs. Set Current", fontsize=14, pad=15)
    plt.xlabel(f"Set Current $I_{{set}}$ [mA] ($V_{{gen}}/220\Omega$)", fontsize=12)
    plt.ylabel("Measured Output $V_{out}$ (OSC CH1) [V]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Annotate points with the exact Set Voltage vs Measured Output
    for row in df.iter_rows(named=True):
        plt.annotate(f"Set: {row['v_gen_target']:.1f}V\nOut: {row['v_out']:.2f}V", 
                     (row['i_ma_set'], row['v_out']), 
                     textcoords="offset points", xytext=(0,15), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULT_IMAGE)
    print(f"✅ Success. Plot: {RESULT_IMAGE}")

def main():
    try:
        with Bench.open(BENCH_CONFIG) as bench:
            setup_instruments(bench)
            data = run_sweep(bench)
            plot_results(data)
            bench.wfg.channel(1).disable()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
