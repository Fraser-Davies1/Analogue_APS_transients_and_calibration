import polars as pl
import numpy as np

df = pl.read_csv("v_to_v_linearity_v2.csv")
# Filter for only the region where the LED is actually ON
active_df = df.filter(pl.col("v_led") > 2.4)

if not active_df.is_empty():
    x = active_df["v_led"].to_numpy()
    y = active_df["v_pixel_drop"].to_numpy()
    r_sq = np.corrcoef(x, y)[0, 1]**2
    print(f"Active Region Linearity (V_led > 2.4V): R² = {r_sq:.5f}")
else:
    print("No data found in the active region.")
