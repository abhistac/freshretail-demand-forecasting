# --- macOS OpenMP preload for XGBoost ---
import os, ctypes
LIBOMP = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
if os.path.exists(LIBOMP):
    os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/opt/libomp/lib")
    try:
        ctypes.CDLL(LIBOMP)
    except OSError:
        pass

import os, sys, pathlib, json, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Paths / Imports ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import INTERIM_DIR

st.set_page_config(page_title="FreshRetail Demand Forecasts", layout="wide")

# ---------- Load data ----------
def load_latest_parquet():
    ps = list(pathlib.Path(INTERIM_DIR).glob("train_sample_*.parquet"))
    if not ps:
        st.error("No parquet samples found in data/interim/. Run the loader script first.")
        st.stop()
    p = max(ps, key=lambda x: x.stat().st_mtime)
    df = pd.read_parquet(p, engine="pyarrow")
    df["dt"] = pd.to_datetime(df["dt"])
    df.sort_values("dt", inplace=True)
    return df, p

df, parquet_path = load_latest_parquet()
st.sidebar.success(f"Loaded sample: {parquet_path.name} ({len(df):,} rows)")

# ---------- Feature engineering ----------
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

df_feat = add_time_features(df)
df_feat = add_stock_and_hour_features(df_feat)
df_feat = add_lag_rolling(df_feat)

# keep rows where lags exist
df_model = df_feat.dropna(subset=["lag_1","lag_7","roll_7_mean","roll_30_mean"]).copy()

# ---------- Split (last 30 days valid) ----------
cutoff_date = df_model["dt"].max() - pd.Timedelta(days=30)
train = df_model[df_model["dt"] < cutoff_date]
valid = df_model[df_model["dt"] >= cutoff_date]

FEATURES = [
    "dayofweek","is_weekend","month",
    "lag_1","lag_7","roll_7_mean","roll_30_mean",
    "discount","holiday_flag","activity_flag",
    "precpt","avg_temperature","avg_humidity","avg_wind_level",
    "instock_frac","hours_sale_mean","hours_sale_max"
]
TARGET = "sale_amount"

X_train, y_train = train[FEATURES], train[TARGET]
X_valid, y_valid = valid[FEATURES], valid[TARGET]

# ---------- Load model artifact if available ----------
models_dir = ROOT / "models"
models_dir.mkdir(exist_ok=True)
use_xgb = False
model_type = None
xgb_model = None
hgb = None

def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    return float(mean_absolute_percentage_error(y_true, y_pred))

# Try XGBoost JSON first
xgb_json = models_dir / "xgb_baseline.json"
xgb_meta = models_dir / "xgb_meta.json"
if xgb_json.exists():
    try:
        import xgboost as xgb
        use_xgb = True
        model_type = "xgboost"
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(xgb_json))
    except Exception as e:
        st.warning(f"Found {xgb_json.name} but failed to load XGBoost model: {e}")
        use_xgb = False

# Else try HistGB pickle
if not use_xgb:
    import joblib
    hgb_path = models_dir / "histgb_baseline.pkl"
    if hgb_path.exists():
        try:
            hgb = joblib.load(hgb_path)
            model_type = "hist_gradient_boosting"
        except Exception as e:
            st.warning(f"Found {hgb_path.name} but failed to load: {e}")

# If no models found, train a quick HistGB on the fly
if xgb_model is None and hgb is None:
    st.info("No saved model found. Training a quick HistGradientBoosting model on the fly…")
    from sklearn.ensemble import HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, max_iter=300)
    hgb.fit(X_train, y_train)
    model_type = "hist_gradient_boosting"

# ---------- Sidebar controls ----------
st.sidebar.header("Filters")
date_min = valid["dt"].min().date()
date_max = valid["dt"].max().date()
date_range = st.sidebar.date_input("Validation date range",
                                   value=(date_min, date_max),
                                   min_value=date_min, max_value=date_max)

# ---------- Predictions ----------
if use_xgb and xgb_model is not None:
    import xgboost as xgb
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    preds = xgb_model.predict(dvalid)
else:
    preds = hgb.predict(X_valid)

valid_view = valid.copy()
valid_view["preds"] = preds

# Apply date filter
start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
valid_view = valid_view[(valid_view["dt"] >= start_d) & (valid_view["dt"] <= end_d)]

# Aggregate by day
daily = valid_view.groupby("dt")[["sale_amount","preds"]].sum().reset_index()

# ---------- Layout ----------
st.title("FreshRetail Demand Forecasts")
st.caption("Aggregated actual vs predicted sales on validation window (last 30 days by default).")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Actual vs Predicted (Daily)")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(daily["dt"], daily["sale_amount"], label="Actual")
    ax.plot(daily["dt"], daily["preds"], label="Predicted")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales"); ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Validation Metrics")
    st.metric("RMSE", f"{rmse(y_valid, preds):,.3f}")
    st.metric("MAPE", f"{mape(y_valid, preds):.2%}")
    st.write(f"**Model**: {model_type or 'unknown'}")
    st.write(f"**Features** ({len(FEATURES)}):")
    st.code(", ".join(FEATURES), language="text")

# ---------- Feature Importance ----------
st.subheader("Feature Importance")
try:
    if model_type == "xgboost" and xgb_model is not None:
        import xgboost as xgb
        score = xgb_model.get_score(importance_type="gain") or xgb_model.get_score(importance_type="weight")
        if all(k.startswith("f") and k[1:].isdigit() for k in score.keys()):
            fmap = {f"f{i}": f for i, f in enumerate(FEATURES)}
            imp = pd.Series({fmap[k]: v for k, v in score.items() if k in fmap})
        else:
            imp = pd.Series(score)
        imp = imp.reindex(FEATURES).fillna(0.0).sort_values(ascending=False)
        st.bar_chart(imp.head(15))
    else:
        # try impurity-based; if all zeros/None -> permutation importance
        base_imp = getattr(hgb, "feature_importances_", None)
        if base_imp is not None and np.nanmax(base_imp) > 0:
            imp = pd.Series(base_imp, index=FEATURES).sort_values(ascending=False)
        else:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(hgb, X_valid, y_valid, n_repeats=5, random_state=42, n_jobs=-1)
            imp = pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)
        st.bar_chart(imp.head(15))
except Exception as e:
    st.write("Feature importance unavailable:", e)

# ---------- Model Comparison table (if available) ----------
cmp_path = ROOT / "reports" / "model_comparison.csv"
st.subheader("Model Comparison (if available)")
if cmp_path.exists():
    cmp_df = pd.read_csv(cmp_path)
    st.dataframe(cmp_df)
else:
    st.info("Run the comparison cell to create reports/model_comparison.csv")

# ---------- Real-Time Simulation (if available) ----------
st.header("📡 Real-Time Simulation")

sim_path = ROOT / "reports" / "stream_results.csv"
if not sim_path.exists():
    st.info(
        "No stream results found yet. Run the simulation first:\n\n"
        "  make simulate\n\n"
        "This will create reports/stream_results.csv"
    )
else:
    sim = pd.read_csv(sim_path)
    # basic hygiene
    if "dt" in sim.columns:
        sim["dt"] = pd.to_datetime(sim["dt"])
    sim = sim.sort_values("dt")
    # if metrics not present for some reason, compute cumulative ones
    if not {"rmse_to_date","mape_to_date"}.issubset(sim.columns):
        # (Re)compute cumulative metrics just in case
        def _rmse(y, yhat):
            y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
            return float(np.sqrt(np.mean((yhat - y)**2)))
        def _mape_safe(y, yhat, eps=1e-3):
            y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
            denom = np.maximum(np.abs(y), eps)
            return float(np.mean(np.abs((yhat - y)/denom)))

        rmses, mapes = [], []
        y_c, yhat_c = [], []
        for _, r in sim.iterrows():
            y_c.append(float(r["y_true"]))
            yhat_c.append(float(r["y_pred"]))
            rmses.append(_rmse(y_c, yhat_c))
            mapes.append(_mape_safe(y_c, yhat_c))
        sim["rmse_to_date"] = rmses
        sim["mape_to_date"] = mapes

    # Sidebar filter for sim dates
    smin, smax = sim["dt"].min().date(), sim["dt"].max().date()
    d1, d2 = st.sidebar.date_input(
        "Real-time view range",
        value=(smin, smax),
        min_value=smin, max_value=smax
    )
    mask = (sim["dt"] >= pd.to_datetime(d1)) & (sim["dt"] <= pd.to_datetime(d2))
    sim_view = sim.loc[mask].copy()

    colA, colB = st.columns([2,1])

    with colA:
        st.subheader("Daily Actual vs Predicted (Simulated)")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(sim_view["dt"], sim_view["y_true"], label="Actual")
        ax.plot(sim_view["dt"], sim_view["y_pred"], label="Predicted")
        ax.set_xlabel("Date"); ax.set_ylabel("Sales"); ax.legend()
        st.pyplot(fig)

    with colB:
        st.subheader("Cumulative Metrics")
        if not sim_view.empty:
            st.metric("RMSE (to date)", f"{sim_view['rmse_to_date'].iloc[-1]:,.3f}")
            st.metric("MAPE (to date)", f"{sim_view['mape_to_date'].iloc[-1]:.2%}")
        else:
            st.metric("RMSE (to date)", "—")
            st.metric("MAPE (to date)", "—")
        st.caption("Metrics recomputed cumulatively through the selected end date.")

    st.subheader("RMSE / MAPE Progression")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(sim_view["dt"], sim_view["rmse_to_date"], label="RMSE (to date)")
    ax2.set_ylabel("RMSE")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(sim_view["dt"], sim_view["mape_to_date"], label="MAPE (to date)")
    ax3.set_ylabel("MAPE")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

    st.subheader("Recent Simulation Steps")
    st.dataframe(sim.tail(10).rename(columns={
        "dt":"Date", "y_true":"Actual", "y_pred":"Predicted",
        "rmse_to_date":"RMSE_to_date", "mape_to_date":"MAPE_to_date"
    }))