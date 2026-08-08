import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# 1. Load the processed hourly data
print("Loading processed data...")
df = pd.read_csv("data/processed/appliance_hourly.csv", index_col="date", parse_dates=True)
target = df["Appliances"]

# 2. Perform the Augmented Dickey-Fuller (ADF) test for stationarity
print("\n--- Augmented Dickey-Fuller Test ---")
adf_result = adfuller(target)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value:.4f}")

if adf_result[1] < 0.05:
    print("\nConclusion: The p-value is < 0.05. The time series is STATIONARY.")
else:
    print("\nConclusion: The p-value is > 0.05. The time series is NON-STATIONARY.")

# 3. Plot the Autocorrelation Function (ACF) to check for seasonality
print("\nGenerating ACF plot...")
plt.figure(figsize=(12, 5))
plot_acf(target, lags=72, ax=plt.gca()) # Looking at 3 days of lags (72 hours)
plt.title("Autocorrelation of Appliance Energy Use (72 Lags)")
plt.xlabel("Lags (Hours)")
plt.ylabel("Correlation")
plt.show()