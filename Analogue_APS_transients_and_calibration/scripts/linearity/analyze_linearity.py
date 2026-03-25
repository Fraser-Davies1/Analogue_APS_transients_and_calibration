import polars as pl
import numpy as np

def analyze():
    try:
        df = pl.read_csv("linearity_results.csv")
        if df.is_empty():
            print("CSV is empty.")
            return

        x = df["integration_time_ms"].to_numpy()
        y = df["v_drop"].to_numpy()

        # Linear Regression: y = mx + c
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        # Calculate R-squared
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        print(f"--- Linearity Analysis ---")
        print(f"Slope (V/ms):   {slope:.6f}")
        print(f"Intercept (V):  {intercept:.6f}")
        print(f"R-squared:      {r_squared:.6f}")
        print(f"\nData Preview:")
        print(df.head(5))

        if r_squared < 0.9:
            print("\nWARNING: Poor linearity detected. (Likely due to PSU being offline)")
        else:
            print("\nSUCCESS: High linearity confirmed.")

    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    analyze()
