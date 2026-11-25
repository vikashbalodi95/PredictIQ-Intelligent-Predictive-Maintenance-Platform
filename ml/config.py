"""
ml/config.py
------------
Machine-learning configuration: dataset schema, feature lists, model paths,
and the sensor thresholds used for health scoring and recommendations.

This is intentionally kept SEPARATE from core/settings.py:
  * core/settings.py  -> infrastructure (database, API, logging)
  * ml/config.py      -> the data-science details

The numbers here are unchanged from the original project, so the ML pipeline
behaves exactly as before. Only the file *paths* now come from the shared
`settings` object so every part of the app agrees on where things live.
"""

from core.settings import settings

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = str(settings.DATA_DIR)
MODEL_DIR = str(settings.MODEL_DIR)

# Dataset
DATASET_FILENAME = "ai4i2020.csv"
DATASET_PATH     = str(settings.DATA_DIR / DATASET_FILENAME)
DATASET_URL      = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

# Saved model artifacts
MODEL_PATH   = str(settings.MODEL_DIR / "best_model.pkl")
SCALER_PATH  = str(settings.MODEL_DIR / "scaler.pkl")
ENCODER_PATH = str(settings.MODEL_DIR / "encoder.pkl")
META_PATH    = str(settings.MODEL_DIR / "model_meta.json")

# ─── ML Config ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ─── Feature Definitions ──────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN = "Machine failure"

# Columns that exist in the raw dataset but are NOT used as model features
RAW_DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

# ─── Sensor Thresholds (for recommendations & health score) ───────────────────
THRESHOLDS = {
    "tool_wear":        200,    # minutes
    "torque":            60,    # Nm
    "process_temp":     313,    # K
    "rotational_speed": 2500,   # rpm
    "temp_diff_min":    8.6,    # K  (process_temp - air_temp minimum)
}

# Tool wear (minutes) at which the tool is considered fully worn out.
# Used by the Remaining-Useful-Life estimator in ml/rul.py.
TOOL_WEAR_LIMIT = 240

# ─── Status thresholds ────────────────────────────────────────────────────────
STATUS_THRESHOLDS = {
    "healthy": 0.30,   # prob < 0.30  → Healthy
    "warning": 0.60,   # prob < 0.60  → Warning
                       # prob >= 0.60 → Critical
}

COLORS = {
    "primary": "#1E88E5",
    "success": "#43A047",
    "warning": "#FB8C00",
    "danger":  "#E53935",
    "info":    "#00ACC1",
    "purple":  "#8E24AA",
}
