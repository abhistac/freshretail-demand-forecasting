# 📈 FreshRetailNet Demand Forecasting — Real-Time Retail Sales Prediction

## Overview

This project focuses on building a **retail demand forecasting pipeline** using the [FreshRetailNet-50K dataset](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K).\
The dataset contains **4.5M+ transaction records** across multiple cities, stores, and product categories, along with contextual information such as promotions, holidays, and weather conditions.

The objective is to:

- Analyze sales patterns and seasonality.
- Engineer meaningful features (lags, rolling averages, calendar & stock features).
- Build baseline and machine learning forecasting models.
- Simulate **real-time forecasting**.
- Create an **interactive Tableau dashboard** for business insights.

---

## 📂 Project Structure

```
freshretail-demand-forecasting/
│
├── data/
│   ├── raw/                # Raw data (from Hugging Face, small sample only)
│   ├── interim/            # Processed sample parquet files
│   └── processed/          # Future: full cleaned & aggregated datasets
│
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_baselines.ipynb  # Naive & Moving Average baselines
│   ├── 03_feature_xgb.ipynb# Feature engineering + XGBoost / HistGradientBoosting
│   ├── 04_prophet.ipynb    # Prophet model
│
├── reports/
│   ├── model_comparison.csv           # Model evaluation results
│   ├── stream_results.csv             # Real-time simulation logs
│   ├── forecast_next_14d.csv          # 14-day overall forecast
│   ├── forecast_next_14d_by_store.csv # store-level 14-day forecasts
│   └── figures/                       # Dashboard screenshots
│
├── src/
│   ├── config.py
│   └── scripts/
│       ├── load_data.py
│       ├── make_samples.py
│       └── forecast_next_14d.py
│
├── Makefile                # Reproducible pipeline commands
└── README.md               # Project documentation
```

---

## 🔍 Exploratory Data Analysis

- Distribution of sales volumes.
- Seasonality & daily sales trends (90-day range).
- Impact of **holidays, promotions, and weather** (temperature, precipitation, wind).
- Stock availability vs sales.

---

## ⚙️ Feature Engineering

- **Lag features**: previous 1-day and 7-day sales.
- **Rolling features**: 7-day & 30-day moving averages.
- **Calendar features**: day of week, weekend, holiday flag.
- **Stock features**: fraction in-stock, hourly sales statistics.

---

## 📊 Forecasting Models

### Baselines

- Naïve forecast (yesterday = today).
- 7-day moving average.
- 30-day moving average.

### Machine Learning

- **XGBoost Regressor** → best model (RMSE ≈ **10.4**, MAPE ≈ **3.6%**).
- HistGradientBoosting (RMSE ≈ 21.9).

### Time Series

- **Prophet** (RMSE ≈ 288).

---

## 📈 Real-Time Simulation

- Scripted pipeline (`make simulate`) streams hourly sales and updates forecasts.
- Outputs:
  - `reports/stream_results.csv` (validation with RMSE/MAPE trend).
  - `reports/forecast_next_14d.csv` (total 14-day forecast).
  - `reports/forecast_next_14d_by_store.csv` (store-level 14-day forecasts).

---

## 📊 Interactive Dashboard (Tableau Public)

Explore the results live:\
👉 [**Retail Demand Forecasting Explorer**](https://public.tableau.com/views/Book1_17558033052330/RetailDemandForecastingExplorer?\:language=en-US\&publish=yes&\:sid=&\:redirect=auth&\:display_count=n&\:origin=viz_share_link)

### Dashboard sections:

- **Model Comparison** — RMSE, MAPE, SMAPE across models.
- **Validation Results** — Actual vs Predicted sales + error metrics over time.
- **History + Forecast** — 90 days of historical sales with 14-day forecast.
- **Store-Level Explorer** — Interactive drill-down by store, view top stores by demand.

📷 See screenshots in [`reports/figures/`](reports/figures)

---

## 🛠 Tech Stack

- **Python**: pandas, scikit-learn, XGBoost, Prophet, matplotlib
- **Data**: Hugging Face `FreshRetailNet-50K`, PyArrow, Fastparquet
- **Visualization**: Tableau Public (interactive dashboards)
- **Workflow**: VS Code, Makefile, virtualenv

---

## ▶️ How to Reproduce

```bash
# 1. Create virtual environment
make env

# 2. Download & prepare data
make data

# 3. Generate sample subset
make sample

# 4. Run notebooks
make eda
make baselines
make xgb
make prophet

# 5. Run real-time simulation & 14-day forecast
make simulate
make forecast14
```

---

## 🌟 Key Results

- XGBoost achieved the **lowest RMSE (≈10.4)** and **MAPE (≈3.6%)**, outperforming baselines and Prophet.
- Built a **real-time simulation** that updates forecasts as new data streams in.
- Delivered a **Tableau Public dashboard** with interactive model comparison, validation results, and forward forecasts by store.

---
