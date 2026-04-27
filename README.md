# FreshRetail Demand Forecasting

End-to-end retail demand forecasting pipeline built on the [FreshRetailNet-50K dataset](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) — 4.5M+ transaction records across multiple cities, stores, and product categories with promotions, holidays, and weather data.

**XGBoost achieved RMSE 10.4 and MAPE 3.6% — outperforming Prophet (RMSE 288) and all naive baselines by a wide margin.**

---

## Live dashboard

🔗 [**Retail Demand Forecasting Explorer on Tableau Public →**](https://public.tableau.com/views/Book1_17558033052330/RetailDemandForecastingExplorer)

Includes: model comparison (RMSE/MAPE/SMAPE), actual vs predicted validation, 14-day forward forecasts, and store-level explorer with interactive drill-down.

---

## Results

| Model | RMSE | MAPE | Notes |
|-------|------|------|-------|
| **XGBoost** | **10.4** | **3.6%** | Best — lag + rolling + calendar features |
| HistGradientBoosting | 21.9 | — | Strong but overfits short-window lags |
| Naïve (yesterday) | — | — | Baseline: assumes today = yesterday |
| 7-day moving average | — | — | Smooths noise, misses trend shifts |
| **Prophet** | **288** | — | Handles seasonality poorly on this dataset |

The Prophet gap (288 vs 10.4) is not a surprise — Prophet is designed for datasets with strong, stable seasonality and long histories. FreshRetailNet has high per-SKU volatility and promotion-driven spikes that tree-based models handle better through feature engineering.

---

## Architecture

```
HuggingFace dataset (4.5M+ records)
         │
         ▼
┌─────────────────────────┐
│  01_eda.ipynb           │  Sales distribution, seasonality, holiday/weather impact
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature Engineering    │  Lag-1, Lag-7 · 7d/30d rolling avg
│                         │  Day of week, weekend, holiday flag
│                         │  Stock fraction, hourly sales stats
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  02_baselines.ipynb     │  Naïve · 7-day MA · 30-day MA
│  03_feature_xgb.ipynb   │  XGBoost · HistGradientBoosting
│  04_prophet.ipynb       │  Prophet
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Real-time simulation   │  Streams hourly, updates RMSE/MAPE rolling
│  (make simulate)        │  Outputs: stream_results.csv
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  14-day forecast        │  Overall + store-level forward projections
│  (make forecast14)      │  Tableau dashboard
└─────────────────────────┘
```

---

## Quick start

```bash
git clone https://github.com/abhistac/freshretail-demand-forecasting.git
cd freshretail-demand-forecasting

# 0. Optional: clean notebook diffs
pip install nbstripout && nbstripout --install

# 1. Create environment
make env

# 2. Download and prepare data
make data && make sample

# 3. Run notebooks in order
make eda && make baselines && make xgb && make prophet

# 4. Real-time simulation + 14-day forecast
make simulate && make forecast14
```

---

## Reload the trained model

```python
import xgboost as xgb

booster = xgb.Booster()
booster.load_model("models/xgb_baseline.json")
```

---

## Stack

Python · XGBoost · Prophet · Pandas · Scikit-learn · Matplotlib · Tableau Public · PyArrow · Fastparquet · HuggingFace Datasets · Makefile

---

## Author

**Abhista Atchutuni** — AI & Data Engineer  
[linkedin.com/in/abhistac](https://linkedin.com/in/abhistac) · [abhistaca@gmail.com](mailto:abhistaca@gmail.com) · [abhistac.github.io](https://abhistac.github.io)
