import polars as pl
import numpy as np

def generate_calibration_numpy():
    df = pl.read_csv("fine_linearity_results.csv")
    df = df.sort("v_led_in")
    
    v_in = df["v_led_in"].to_numpy()
    v_out = df["v_pixel_drop"].to_numpy()

    # Define 10 linear steps of "Light Output" (Integration Drop)
    # We ignore the noise floor below 2.3V
    active_mask = v_in >= 2.3
    v_out_active = v_out[active_mask]
    v_in_active = v_in[active_mask]

    target_drops = np.linspace(v_out_active.min(), v_out_active.max(), 11)
    
    # Interp v_in as a function of v_out
    required_v_in = np.interp(target_drops, v_out_active, v_in_active)

    print("--- LED Linearization Table (Calculated from Pixel Data) ---")
    print(f"{'Light Level %':<15} | {'Required PSU V_in':<20}")
    print("-" * 40)
    for i, volt in enumerate(required_v_in):
        print(f"{i*10:14.0f}% | {volt:.4f} V")

    pl.DataFrame({
        "light_percent": np.arange(0, 110, 10),
        "v_led_control": required_v_in
    }).write_csv("led_lin_lut.csv")

if __name__ == "__main__":
    generate_calibration_numpy()
