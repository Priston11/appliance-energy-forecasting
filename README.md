# Appliance Energy Forecasting Case Study

This repository contains the complete time-series coding case study and analysis for predicting household appliance energy consumption. The project explores, models, and forecasts energy demand over a 14-day test horizon using a variety of statistical, machine learning, and deep learning foundation models.

---

## Project Structure

```text
appliance-energy-forecasting/
│
├── data/
│   └── appliance_hourly.csv          
├── outputs/
│   ├── figures/
│   │   └── forecast_comparison.png   
│   ├── forecasts/
│   │   └── all_forecasts.csv        
│   └── metrics/
│       └── model_comparison.csv      
├── scripts/
│   ├── 01_download.py                
│   ├── 02_exploratory_analysis.py    
│   ├── 03_sarima_gridsearch.py      
│   └── pipeline.py                   
├── requirements.txt                 
└── README.md                         
DatasetSource: UCI Machine Learning Repository (energydata_complete.csv)Original Resolution: 10-minute intervalsResampled Resolution: Hourly means (3,290 rows)Target Variable: Appliances (Energy consumption in Wh)Exogenous Weather Variables: T_out, RH_out, Windspeed, Visibility, TdewpointModel Comparison ResultsAll models were evaluated on a 14-day test period (336 hours). Below are the final quantitative metrics:ModelMAERMSEMASEBiasFeature-Based Model (XGBoost/RF)34.22855.7660.6416.471SARIMAX36.63064.1650.686-2.388Foundation Model (Chronos)38.11877.1560.714-31.940Seasonal Naive (Weekly)42.63479.2900.798-10.818Mean Benchmark50.31974.9060.942-3.109Seasonal Naive (Daily)86.959129.2321.62864.013Naive Benchmark250.640258.8204.692247.763Drift Benchmark266.373274.6114.986264.501VisualizationsModel Forecast ComparisonThe chart below illustrates how each model tracked the actual energy usage over the 14-day test period:Installation & Usage1. Clone the RepositoryBashgit clone [https://github.com/Priston11/appliance-energy-forecasting.git](https://github.com/Priston11/appliance-energy-forecasting.git)
cd appliance-energy-forecasting
2. Set Up Virtual EnvironmentPythonpython -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
3. Run the PipelineExecute the main modeling and forecasting script:Bashpython scripts/pipeline.py
