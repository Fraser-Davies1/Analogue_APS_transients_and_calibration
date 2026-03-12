import polars as pl
import numpy as np

def get_92_percent_voltage():
    df = pl.read_csv("fine_linearity_results.csv")
    df = df.sort("v_led_in")
    
    v_in = df["v_led_in"].to_numpy()
    v_out = df["v_pixel_drop"].to_numpy()

    active_mask = v_in >= 2.3
    v_out_active = v_out[active_mask]
    v_in_active = v_in[active_mask]

    # Target: 92% of the range between min and max active drop
    target_drop = v_out_active.min() + 0.92 * (v_out_active.max() - v_out_active.min())
    v_92 = np.interp(target_drop, v_out_active, v_in_active)
    
    print(f"92% Intensity Voltage: {v_92:.4f}V")
    return v_92

if __name__ == "__main__":
    get_92_percent_voltage()
