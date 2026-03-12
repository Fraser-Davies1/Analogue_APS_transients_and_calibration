import polars as pl
import numpy as np

df = pl.read_csv("automated_pixel_report.csv")
# Filter for typical active LED range
active_df = df.filter((pl.col("v_led_in") > 2.5) & (pl.col("v_led_in") < 4.0))

if not active_df.is_empty():
    x = active_df["v_led_in"].to_numpy()
    y = active_df["delta_v_out"].to_numpy()
    r_sq = np.corrcoef(x, y)[0, 1]**2
    print(f"Active Region (2.5V - 4.0V) R²: {r_sq:.5f}")
    print(f"Voltage Drops in Active Region: {y}")
else:
    print("No data in requested range.")
