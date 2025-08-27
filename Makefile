# ================================
# FreshRetail Demand Forecasting
# Makefile (Mac-friendly)
# ================================

# ---------- Config ----------
PYTHON      := python
VENV_DIR    := .venv
ACTIVATE    := source $(VENV_DIR)/bin/activate
PIP         := $(VENV_DIR)/bin/pip
PY_BIN      := $(VENV_DIR)/bin/python
PYTHONPATH  := PYTHONPATH=.

# Optional: Homebrew libomp path for XGBoost on macOS (no-op if not present)
LIBOMP      := /opt/homebrew/opt/libomp/lib/libomp.dylib
export DYLD_LIBRARY_PATH := /opt/homebrew/opt/libomp/lib:$(DYLD_LIBRARY_PATH)

# ---------- Phony ----------
.PHONY: help env install install-kernel freeze data sample baselines feature_xgb prophet simulate app app-dev lint fmt clean forecast14 check

# ---------- Help ----------
help:
	@echo "Usage:"
	@echo "  make env           - Create virtual environment (.venv)"
	@echo "  make install       - Install project dependencies (from requirements.txt)"
	@echo "  make freeze        - Freeze current environment to requirements.txt"
	@echo "  make data          - Download HF dataset + write sample parquet (data/interim/*)"
	@echo "  make baselines     - (Open notebook) 02_baselines.ipynb"
	@echo "  make feature_xgb   - (Open notebook) 03_feature_xgb.ipynb"
	@echo "  make simulate      - Run streaming simulation → reports/stream_results.csv"
	@echo "  make app           - Launch Streamlit app"
	@echo "  make app-dev       - Launch Streamlit app with autoreload"
	@echo "  make lint          - Run basic lint (ruff) if installed"
	@echo "  make fmt           - Run formatting (ruff format / black) if installed"
	@echo "  make clean         - Remove caches and temporary files"
	@echo "  make forecast14    - Generate 14-day forward forecast CSVs"
	@echo "  make check         - Verify key imports (xgboost/streamlit/prophet) in venv"
	@echo "  make install-kernel- Register the venv as a Jupyter kernel (freshretail)"

# ---------- Environment ----------
env:
	@echo ">>> Creating venv in $(VENV_DIR)"
	$(PYTHON) -m venv $(VENV_DIR)
	@echo ">>> Upgrading pip"
	$(ACTIVATE) && python -m pip install --upgrade pip

install: env
	@echo ">>> Installing requirements"
	@if [ -f requirements.txt ]; then \
	  $(ACTIVATE) && $(PIP) install -r requirements.txt ; \
	else \
	  echo "requirements.txt not found. Installing minimal deps..."; \
	  $(ACTIVATE) && $(PIP) install pandas pyarrow numpy scikit-learn fastparquet streamlit xgboost prophet ; \
	fi

install-kernel: env
	@echo ">>> Installing IPython kernel for this venv (freshretail)"
	$(ACTIVATE) && $(PY_BIN) -m ipykernel install --user --name freshretail --display-name "Python (freshretail)"

freeze:
	@echo ">>> Freezing environment to requirements.txt"
	$(ACTIVATE) && $(PIP) freeze | sed '/^-e /d' > requirements.txt
	@echo "Wrote requirements.txt"

# ---------- Data ----------
data:
	@echo ">>> Loading dataset + writing sample parquet"
	@if [ -f $(LIBOMP) ]; then echo "libomp found: $(LIBOMP)"; else echo "libomp not found (ok)."; fi
	$(ACTIVATE) && $(PYTHONPATH) $(PY_BIN) src/scripts/load_data.py

# (Alias; load_data.py already writes the sample parquet)
sample: data

# ---------- Notebooks (opens in Jupyter if installed) ----------
baselines:
	@echo ">>> Opening notebooks/02_baselines.ipynb"
	$(ACTIVATE) && jupyter notebook notebooks/02_baselines.ipynb

feature_xgb:
	@echo ">>> Opening notebooks/03_feature_xgb.ipynb"
	$(ACTIVATE) && jupyter notebook notebooks/03_feature_xgb.ipynb

prophet:
	@echo ">>> Opening Prophet section in 03_feature_xgb.ipynb"
	$(ACTIVATE) && jupyter notebook notebooks/03_feature_xgb.ipynb

# ---------- Streaming Simulation ----------
simulate:
	@echo ">>> Running real-time simulation"
	@if [ -f $(LIBOMP) ]; then echo "Using libomp at $(LIBOMP)"; fi
	$(ACTIVATE) && $(PYTHONPATH) $(PY_BIN) src/scripts/simulate_stream.py

# ---------- Streamlit App ----------
app:
	@echo ">>> Launching Streamlit app (app/streamlit_app.py)"
	@if [ -f $(LIBOMP) ]; then echo "Using libomp at $(LIBOMP)"; fi
	$(ACTIVATE) && $(PYTHONPATH) $(PY_BIN) -m streamlit run app/streamlit_app.py

# Dev mode (enables streamlit's file watcher to auto-reload)
app-dev:
	@echo ">>> Launching Streamlit app in dev mode"
	@if [ -f $(LIBOMP) ]; then echo "Using libomp at $(LIBOMP)"; fi
	$(ACTIVATE) && export STREAMLIT_SERVER_RUN_ON_SAVE=true && $(PYTHONPATH) $(PY_BIN) -m streamlit run app/streamlit_app.py

# ---------- Quality ----------
lint:
	@echo ">>> Running ruff (if installed)"
	-$(ACTIVATE) && ruff check src || true

fmt:
	@echo ">>> Formatting with ruff/black (if installed)"
	-$(ACTIVATE) && ruff format src || true
	-$(ACTIVATE) && black src || true

# ---------- Clean ----------
clean:
	@echo ">>> Cleaning caches"
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

forecast14:
	$(ACTIVATE) && $(PYTHONPATH) $(PY_BIN) src/scripts/forecast_next_14d.py

check:
	@echo ">>> Verifying imports inside venv"
	$(ACTIVATE) && $(PY_BIN) -c "import xgboost, streamlit, prophet; print('OK: xgboost', xgboost.__version__, '| streamlit', streamlit.__version__, '| prophet', getattr(prophet, '__version__', 'n/a'))"