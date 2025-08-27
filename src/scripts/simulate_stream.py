"""
Simulate real-time demand forecasting.
Reveals validation data one day at a time, trains on history, forecasts next day, and logs metrics.
"""

import os, sys, pathlib, datetime
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import INTERIM_DIR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ------------------ Helpers ------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape_safe(y_true, y_pred, eps=1e-3):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / denom)))

# ------------------ Load latest sample ------------------
ps = list(pathlib.Path(INTERIM_DIR).glob("train_sample_*.parquet"))
assert ps, "No parquet samples found. Run `make sample` first."
p = max(ps, key=lambda x: x.stat().st_mtime)

df = pd.read_parquet(p, engine="pyarrow")
df["dt"] = pd.to_datetime(df["dt"])
df = df.sort_values("dt").reset_index(drop=True)

# ------------------ Features ------------------
def add_time_features(d):
    d = d.copy()
    d["dayofweek"] = d["dt"].dt.dayofweek
    d["is_weekend"] = d["dayofweek"].isin([5,6]).astype(int)
    d["month"] = d["dt"].dt.month
    return d

def _is_listlike(v):
    return isinstance(v, (list, tuple, np.ndarray))

def add_stock_and_hour_features(d):
    d = d.copy()
    d["hours_sale_mean"] = d["hours_sale"].apply(lambda x: float(np.nanmean(x)) if _is_listlike(x) else np.nan)
    d["hours_sale_max"]  = d["hours_sale"].apply(lambda x: float(np.nanmax(x)) if _is_listlike(x) else np.nan)
    def frac_instock(arr):
        if not _is_listlike(arr) or len(arr) == 0:
            return np.nan
        a = np.array(arr, dtype=float)
        return float((a > 0).mean())
    d["instock_frac"] = d["hours_stock_status"].apply(frac_instock)
    return d

def add_lag_rolling(d, group_cols=("store_id","product_id"), target="sale_amount"):
    d = d.sort_values(["dt"]).copy()
    g = d.groupby(list(group_cols), group_keys=False)
    d["lag_1"]  = g[target].shift(1)
    d["lag_7"]  = g[target].shift(7)
    d["roll_7_mean"]  = g[target].rolling(7, min_periods=1).mean().reset_index(level=list(range(len(group_cols))), drop=True)
    d["roll_30_mean"] = g[target].rolling(30, min_periods=1).mean().reset_index(level=list(range(len(group_cols))), drop=True)
    return d

FEATURES = [
    "dayofweek","is_weekend","month",
    "lag_1","lag_7","roll_7_mean","roll_30_mean",
    "discount","holiday_flag","activity_flag",
    "precpt","avg_temperature","avg_humidity","avg_wind_level",
    "instock_frac","hours_sale_mean","hours_sale_max"
]
TARGET = "sale_amount"

# ------------------ Build features ------------------
df_feat = add_time_features(df)
df_feat = add_stock_and_hour_features(df_feat)
df_feat = add_lag_rolling(df_feat)
df_feat = df_feat.dropna(subset=["lag_1","lag_7","roll_7_mean","roll_30_mean"]).copy()

# ------------------ Train/Valid split ------------------
cutoff_date = df_feat["dt"].max() - pd.Timedelta(days=30)
train_full = df_feat[df_feat["dt"] < cutoff_date].copy()
valid_full = df_feat[df_feat["dt"] >= cutoff_date].copy()

print("Train:", train_full["dt"].min().date(), "→", train_full["dt"].max().date())
print("Valid (to simulate stream):", valid_full["dt"].min().date(), "→", valid_full["dt"].max().date())

# ------------------ Streaming simulation ------------------
out_path = pathlib.Path("reports"); out_path.mkdir(exist_ok=True)
log_path = out_path / "stream_results.csv"

rows = []
hgb = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, max_iter=200)

history = train_full.copy()
metrics = []

for day, chunk in valid_full.groupby("dt"):
    # train on all history so far
    X_h, y_h = history[FEATURES], history[TARGET]
    if len(X_h) == 0:
        continue
    hgb.fit(X_h, y_h)

    # predict this day's sales
    X_c, y_c = chunk[FEATURES], chunk[TARGET]
    y_pred = hgb.predict(X_c)

    # aggregate to daily
    true_daily = y_c.sum()
    pred_daily = y_pred.sum()

    # metrics up to this day (cumulative)
    metrics.append({"dt": day, "y_true": true_daily, "y_pred": pred_daily})
    df_m = pd.DataFrame(metrics)
    rmse_val = rmse(df_m["y_true"], df_m["y_pred"])
    mape_val = mape_safe(df_m["y_true"], df_m["y_pred"])

    rows.append({"dt": day, "y_true": true_daily, "y_pred": pred_daily,
                 "rmse_to_date": rmse_val, "mape_to_date": mape_val})

    # expand training history with this day
    history = pd.concat([history, chunk], ignore_index=True)

# save results
df_log = pd.DataFrame(rows)
df_log.to_csv(log_path, index=False)
print("Simulation complete →", log_path)
print(df_log.head())