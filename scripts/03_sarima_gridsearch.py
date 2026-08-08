import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
import itertools
import warnings

# Ignore harmless statistical warnings during the grid search
warnings.filterwarnings("ignore")

# 1. Load Data
print("Loading data...")
df = pd.read_csv("data/processed/appliance_hourly.csv", index_col="date", parse_dates=True)
target = df["Appliances"]

# We use the last 14 days as the test set (336 hours).
# To make this loop run in minutes instead of hours, we'll train the grid search 
# on a recent 3-week subset of the training data.
train = target.iloc[-840:-336] 
test = target.iloc[-336:-312]  # The assignment asks for a 24-hour forecast horizon

# 2. Define the Grid (p=0-6, d=0-2, q=0-6)
p = range(0, 7)
d = range(0, 3)
q = range(0, 7)
pdq_combinations = list(itertools.product(p, d, q))

best_aic = float("inf")
best_pdq = None
best_model = None

print(f"Running Grid Search for {len(pdq_combinations)} combinations...")
print("This will test every parameter (grab a coffee, this takes a few minutes!)...")

for combo in pdq_combinations:
    try:
        model = ARIMA(train, order=combo)
        model_fit = model.fit()
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_pdq = combo
            best_model = model_fit
    except Exception:
        continue

print(f"\n--- GRID SEARCH COMPLETE ---")
print(f"Best Parameters (p, d, q): {best_pdq}")
print(f"Lowest AIC Score: {best_aic:.2f}")

# 3. Residual Diagnostics
print("\nGenerating Residual Plots...")
residuals = best_model.resid

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(residuals, lags=40, ax=ax[0])
ax[0].set_title("ACF of Model Residuals")

ax[1].hist(residuals, bins=50, edgecolor='black')
ax[1].set_title("Distribution of Residuals")
plt.suptitle("SARIMA Residual Diagnostics")
plt.show()

# 4. Forecast and Confidence Intervals
print("\nForecasting next 24 hours...")
forecast_obj = best_model.get_forecast(steps=24)
forecast_values = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int()

# Calculate RMSE
rmse_val = np.sqrt(mean_squared_error(test, forecast_values))
print(f"24-Hour Forecast RMSE: {rmse_val:.3f}")

# Plotting the forecast with Confidence Intervals
plt.figure(figsize=(12, 6))
# Only plot the last few days of training data so we can see the forecast clearly
plt.plot(train.index[-100:], train.iloc[-100:], label="Training Data")
plt.plot(test.index, test, label="Actual Energy Use (Next 24h)")
plt.plot(forecast_values.index, forecast_values, color='red', label="Model Forecast")

# Add the shaded confidence intervals!
plt.fill_between(
    conf_int.index, 
    conf_int.iloc[:, 0], 
    conf_int.iloc[:, 1], 
    color='pink', alpha=0.4, label="Confidence Interval"
)

plt.title(f"SARIMA {best_pdq} - 24-Hour Forecast with Confidence Intervals")
plt.ylabel("Energy Use")
plt.legend()
plt.show()