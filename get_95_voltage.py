import polars as pl
import numpy as np

def get_95_percent_voltage():
    df = pl.read_csv("fine_linearity_results.csv")
    df = df.sort("v_led_in")
    
    v_in = df["v_led_in"].to_numpy()
    v_out = df["v_pixel_drop"].to_numpy()

    active_mask = v_in >= 2.3
    v_out_active = v_out[active_mask]
    v_in_active = v_in[active_mask]

    # Target: 95% of the range between min and max active drop
    target_drop = v_out_active.min() + 0.95 * (v_out_active.max() - v_out_active.min())
    v_95 = np.interp(target_drop, v_out_active, v_in_active)
    
    # Update LUT or just return
    print(f"95% Intensity Voltage: {v_95:.4f}V")
    return v_95

if __name__ == "__main__":
    get_95_percent_voltage()
