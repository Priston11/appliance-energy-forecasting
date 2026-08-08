import pandas as pd
import matplotlib.pyplot as plt
import ssl
import urllib.request

# Bypass Mac SSL Certificate error
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Download data from UCI
print("Downloading data from UCI (SSL check bypassed)...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
df = pd.read_csv(url)

# 2. Parse timestamp and set index
print("Cleaning data...")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

# Clean up numeric columns and drop missing target values
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["Appliances"])

# 3. Resample to hourly data
print("Resampling to hourly intervals...")
hourly = df.resample("h").mean()
hourly = hourly.interpolate("time")
hourly = hourly.dropna()

# 4. Save processed data to your data folder
print("Saving processed data...")
hourly.to_csv("data/processed/appliance_hourly.csv")
print("Data saved successfully!")

# 5. Initial plot
print("Generating plot...")
plt.figure(figsize=(15, 5))
hourly['Appliances'].plot()
plt.title("Hourly Appliance Energy Consumption")
plt.ylabel("Energy Use")
plt.show()