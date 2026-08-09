# Appliance Energy Forecasting

This repository contains a reproducible time-series forecasting pipeline for modelling and forecasting household appliance energy use. 

The project uses the **Appliances Energy Prediction** dataset, which contains appliance energy consumption, indoor temperature and humidity sensor measurements, outdoor weather variables, and timestamp information. The aim is to forecast short-term household appliance energy use over a 24-hour horizon and evaluate whether increasingly complex models improve on simple benchmark methods.

## Models Evaluated
This project compares multiple forecasting approaches:
1. **Benchmark Models:** Mean, Naive, Daily Seasonal Naive, Weekly Seasonal Naive, and Drift forecasts.
2. **SARIMAX:** A statistical model capturing short-term autocorrelation and daily seasonality, including exogenous weather variables.
3. **Feature-Based Machine Learning:** An XGBoost model utilizing lagged features, rolling statistics, time-based features, and weather covariates.
4. **Time-Series Foundation Model:** The Chronos AI model by Amazon, utilized for zero-shot forecasting.

## Repository Structure
```text
appliance-energy-forecasting/

├── README.md
├── requirements.txt
├── data/
│   └── processed/
├── notebooks/
├── scripts/
│   └── pipeline.py
└── outputs/
    ├── figures/
    ├── forecasts/
    └── metrics/