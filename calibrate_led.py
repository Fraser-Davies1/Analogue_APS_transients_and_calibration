import polars as pl
import numpy as np
from scipy.interpolate import interp1d

def generate_calibration():
    # 1. Load the high-res data we just captured
    df = pl.read_csv("fine_linearity_results.csv")
    
    # Clean data: remove noise and ensure strictly increasing for interpolation
    df = df.sort("v_led_in")
    v_in = df["v_led_in"].to_numpy()
    v_out = df["v_pixel_drop"].to_numpy()

    # 2. Create the Inverse Mapping (Linearization)
    # We want to know: What V_in gives me a specific Delta_V (Light Level)?
    # We use a cubic spline for the knee and linear for the rest
    interp_func = interp1d(v_out, v_in, kind='linear', fill_value="extrapolate")

    # 3. Create a normalized "Light Intensity" scale (0 to 1)
    max_drop = v_out.max()
    min_drop = v_out.min()
    
    intensities = np.linspace(0.1, 1.0, 10) # 10% to 100% brightness
    target_drops = min_drop + intensities * (max_drop - min_drop)
    required_psu_voltages = interp_func(target_drops)

    print("--- LED Linearization Table ---")
    print("Target Brightness | Required PSU Voltage (V)")
    print("------------------------------------------")
    for level, volt in zip(intensities, required_psu_voltages):
        print(f"{level*100:15.0f}% | {volt:.4f} V")

    # Save the LUT for use in your measurement scripts
    lut_df = pl.DataFrame({
        "normalized_intensity": intensities,
        "v_psu_setting": required_psu_voltages
    })
    lut_df.write_csv("led_calibration_lut.csv")
    print("\nCalibration table saved to led_calibration_lut.csv")

if __name__ == "__main__":
    generate_calibration()
