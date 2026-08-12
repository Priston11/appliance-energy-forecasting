# Appliance Energy Forecasting Case Study

This repository contains the complete time-series coding case study and analysis for predicting household appliance energy consumption. The project explores, models, and forecasts energy demand over a 14-day test horizon using a variety of statistical, machine learning, and deep learning foundation models.

## Project Structure

```text
appliance-energy-forecasting/
├── data/
│   └── appliance_hourly.csv          # Resampled hourly dataset
├── outputs/
│   ├── figures/
│   │   └── forecast_comparison.png   # Multi-model comparison plot
│   ├── forecasts/
│   │   └── all_forecasts.csv         # Raw predictions for all models
│   └── metrics/
│       └── model_comparison.csv      # Evaluation metrics table
├── scripts/
│   ├── 01_download.py                # Data retrieval and hourly binning
│   ├── 02_exploratory_analysis.py    # EDA and stationarity checks
│   ├── 03_sarima_gridsearch.py       # SARIMA grid search and diagnostics
│   └── pipeline.py                   # Main end-to-end execution script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation






Dataset Overview
Source: UCI Machine Learning Repository (energydata_complete.csv)  
PY

Original Resolution: 10-minute intervals  
PY

Resampled Resolution: Hourly means (3,290 rows)

Target Variable: Appliances (Energy consumption in Wh)

Exogenous Weather Variables: T_out, RH_out, Windspeed, Visibility, Tdewpoint

[cite: 1]

Model Comparison Results
Evaluated on a 14-day test period (336 hours)[cite: 1]:

Model	MAE	RMSE	MASE	Bias
feature_model	34.228	55.766	0.641	6.471
sarimax	36.630	64.165	0.686	-2.388
foundation_model	38.118	77.156	0.714	-31.940
seasonal_naive_weekly	42.634	79.290	0.798	-10.818
mean	50.319	74.906	0.942	-3.109
seasonal_naive_daily	86.959	129.232	1.628	64.013
naive	250.640	258.820	4.692	247.763
drift	266.373	274.611	4.986	264.501




Visualizations
Multi-Model Forecast Comparison
Installation & Usage
Bash
git clone [https://github.com/Priston11/appliance-energy-forecasting.git](https://github.com/Priston11/appliance-energy-forecasting.git)
cd appliance-energy-forecasting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/pipeline.py
