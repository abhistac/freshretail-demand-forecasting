"""
Make 14-day forward forecasts using the saved model (XGBoost if available, else HistGB).
Writes:
  - reports/forecast_next_14d.csv         (aggregate daily forecast)
  - reports/forecast_next_14d_by_store.csv (optional per-store breakdown)
"""

import os, sys, pathlib, json, datetime
import numpy as np
import pandas as pd

# ---------- Repo paths ----------
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import INTERIM_DIR  # uses your existing config

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

HORIZON = 14

# ---------- Load latest parquet sample ----------
parqs = list(pathlib.Path(INTERIM_DIR).glob("train_sample_*.parquet"))
assert parqs, "No parquet samples found. Run `make data` first."
p = max(parqs, key=lambda x: x.stat().st_mtime)

df = pd.read_parquet(p, engine="pyarrow")
df["dt"] = pd.to_datetime(df["dt"])
df = df.sort_values("dt").reset_index(drop=True)

# ---------- Feature builders (same as app) ----------
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

# ---------- Prep history with lags already computed ----------
df_feat = add_time_features(df)
df_feat = add_stock_and_hour_features(df_feat)
df_feat = add_lag_rolling(df_feat)
df_feat = df_feat.sort_values(["store_id","product_id","dt"])
df_feat = df_feat.reset_index(drop=True)

# Keep only rows where lags exist (for stable model input)
hist = df_feat.dropna(subset=["lag_1","lag_7","roll_7_mean","roll_30_mean"]).copy()
last_date = hist["dt"].max()
print(f"Loaded history through: {last_date.date()} | rows: {len(hist):,}")

# ---------- Load model artifact ----------
use_xgb = False
xgb_model = None
hgb = None
model_type = None

xgb_json = MODELS_DIR / "xgb_baseline.json"
if xgb_json.exists():
    try:
        import xgboost as xgb
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(xgb_json))
        use_xgb = True
        model_type = "xgboost"
        print("Loaded XGBoost model:", xgb_json.name)
    except Exception as e:
        print("Failed to load XGBoost model:", e)

if not use_xgb:
    import joblib
    hgb_path = MODELS_DIR / "histgb_baseline.pkl"
    if hgb_path.exists():
        hgb = joblib.load(hgb_path)
        model_type = "hist_gradient_boosting"
        print("Loaded HistGB model:", hgb_path.name)
    else:
        # Fallback: quick train on full history (portfolio-friendly)
        from sklearn.ensemble import HistGradientBoostingRegressor
        hgb = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, max_iter=300)
        hgb.fit(hist[FEATURES], hist[TARGET])
        model_type = "hist_gradient_boosting"
        print("No saved model found. Trained a quick HistGB on the fly.")

def predict_batch(X):
    if use_xgb and xgb_model is not None:
        import xgboost as xgb
        return xgb_model.predict(xgb.DMatrix(X))
    return hgb.predict(X)

# ---------- Helper: build next-day feature frame ----------
def build_next_day_frame(history_df: pd.DataFrame, next_day: pd.Timestamp) -> pd.DataFrame:
    """
    For each (store_id, product_id), take the last known row and update to next_day:
      - time features from next_day
      - lags/rollings recomputed from history
      - exogenous (discount/holiday/weather/instock/hour stats) frozen at last known values
    Returns a frame with all FEATURE columns ready for prediction.
    """
    # last available per group
    idx = history_df.groupby(["store_id","product_id"])["dt"].idxmax()
    last_rows = history_df.loc[idx, ["store_id","product_id","dt",TARGET,
                                     "discount","holiday_flag","activity_flag",
                                     "precpt","avg_temperature","avg_humidity","avg_wind_level",
                                     "instock_frac","hours_sale_mean","hours_sale_max"]].copy()

    # compute lags & rollings from history
    g = history_df.groupby(["store_id","product_id"], group_keys=False)
    # last day's actual/pred becomes lag_1
    lag1 = g[TARGET].apply(lambda s: s.iloc[-1] if len(s) > 0 else np.nan)
    # 7 days ago if exists, else fallback to last
    def last_7(s):
        if len(s) >= 7:
            return s.iloc[-7]
        return s.iloc[-1] if len(s) > 0 else np.nan
    lag7 = g[TARGET].apply(last_7)
    # rolling means
    roll7 = g[TARGET].apply(lambda s: float(s.tail(7).mean()) if len(s) > 0 else np.nan)
    roll30 = g[TARGET].apply(lambda s: float(s.tail(30).mean()) if len(s) > 0 else np.nan)

    key = last_rows.set_index(["store_id","product_id"]).index
    # assemble next-day rows
    next_rows = last_rows.set_index(["store_id","product_id"])
    next_rows["dt"] = next_day
    next_rows["dayofweek"] = next_day.dayofweek
    next_rows["is_weekend"] = int(next_day.dayofweek in [5,6])
    next_rows["month"] = next_day.month

    next_rows["lag_1"] = lag1.reindex(key).values
    next_rows["lag_7"] = lag7.reindex(key).values
    next_rows["roll_7_mean"] = roll7.reindex(key).values
    next_rows["roll_30_mean"] = roll30.reindex(key).values

    # Keep only needed columns in right order
    next_X = next_rows.reset_index()[["store_id","product_id","dt"] + FEATURES].copy()
    return next_X

# ---------- Roll forward HORIZON days ----------
agg_rows = []
by_store_rows = []

current_hist = hist.copy()
for i in range(HORIZON):
    next_day = current_hist["dt"].max() + pd.Timedelta(days=1)
    next_X = build_next_day_frame(current_hist, next_day)

    # Predict per-row and aggregate
    yhat = predict_batch(next_X[FEATURES])
    next_X["pred"] = yhat

    # Aggregate for outputs
    agg_rows.append({"dt": next_day, "pred": float(np.sum(yhat))})
    bys = next_X.groupby("store_id")["pred"].sum().reset_index()
    bys["dt"] = next_day
    by_store_rows.append(bys)

    # Append pseudo-observations so lags/rollings advance
    # (use prediction as proxy for next day actuals)
    append_cols = [
        "store_id","product_id","dt",TARGET,
        "discount","holiday_flag","activity_flag","precpt",
        "avg_temperature","avg_humidity","avg_wind_level",
        "instock_frac","hours_sale_mean","hours_sale_max"
    ]

    # We already have the correct exogenous (last known) in next_X and the new 'dt'
    # so just take them directly from next_X and set sale_amount = pred
    pseudo = next_X[[
        "store_id","product_id","dt",
        "discount","holiday_flag","activity_flag","precpt",
        "avg_temperature","avg_humidity","avg_wind_level",
        "instock_frac","hours_sale_mean","hours_sale_max"
    ]].copy()
    pseudo[TARGET] = next_X["pred"].values

    # ensure column order
    pseudo = pseudo[append_cols]

    # extend history
    current_hist = pd.concat([current_hist, pseudo], ignore_index=True)

# ---------- Save outputs ----------
fc = pd.DataFrame(agg_rows).sort_values("dt")
fc.to_csv(REPORTS_DIR / "forecast_next_14d.csv", index=False)
print("Wrote:", REPORTS_DIR / "forecast_next_14d.csv")

by_store = pd.concat(by_store_rows, ignore_index=True)[["dt","store_id","pred"]].sort_values(["dt","store_id"])
by_store.to_csv(REPORTS_DIR / "forecast_next_14d_by_store.csv", index=False)
print("Wrote:", REPORTS_DIR / "forecast_next_14d_by_store.csv")