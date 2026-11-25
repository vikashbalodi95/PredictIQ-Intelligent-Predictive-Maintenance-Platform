"""
preprocess.py
-------------
Data loading, cleaning, encoding, and scaling pipeline.
Handles both the real AI4I 2020 dataset and a synthetic fallback.
"""

import os
import numpy as np
import pandas as pd
import joblib
import requests

from sklearn.preprocessing import StandardScaler, LabelEncoder

from ml.config import (
    DATA_DIR, MODEL_DIR, DATASET_PATH, DATASET_URL,
    SCALER_PATH, ENCODER_PATH,
    FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN,
    RAW_DROP_COLS, RANDOM_STATE,
)


# ─── Dataset Loading ──────────────────────────────────────────────────────────

def _generate_synthetic_data(n_samples: int = 10_000) -> pd.DataFrame:
    """
    Generate synthetic data that mirrors the AI4I 2020 schema.
    Used as fallback if the real dataset cannot be downloaded.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    types        = rng.choice(["L", "M", "H"], size=n_samples, p=[0.60, 0.30, 0.10])
    air_temp     = rng.normal(300.0, 2.0, n_samples)
    process_temp = air_temp + rng.normal(10.0, 1.0, n_samples)
    rot_speed    = rng.normal(1538.0, 179.0, n_samples).astype(int)
    torque       = np.abs(rng.normal(40.0, 10.0, n_samples))
    tool_wear    = rng.integers(0, 251, n_samples)

    # Simulate failure sub-modes (mirrors the real dataset logic)
    power = torque * rot_speed * (2 * np.pi / 60)
    twf = (tool_wear > 200).astype(int)
    hdf = ((process_temp - air_temp < 8.6) & (rot_speed * torque < 3500)).astype(int)
    pwf = ((power < 3500) | (power > 9000)).astype(int)
    type_limit = np.where(types == "L", 11_000, np.where(types == "M", 12_000, 13_000))
    osf = (tool_wear * torque > type_limit).astype(int)
    rnf = (rng.random(n_samples) < 0.001).astype(int)
    failure = np.clip(twf + hdf + pwf + osf + rnf, 0, 1)

    df = pd.DataFrame({
        "UDI":                       range(1, n_samples + 1),
        "Product ID":                [f"{t}{i}" for t, i in zip(types, range(1, n_samples + 1))],
        "Type":                      types,
        "Air temperature [K]":       air_temp,
        "Process temperature [K]":   process_temp,
        "Rotational speed [rpm]":    rot_speed,
        "Torque [Nm]":               torque,
        "Tool wear [min]":           tool_wear,
        "Machine failure":           failure,
        "TWF": twf, "HDF": hdf, "PWF": pwf, "OSF": osf, "RNF": rnf,
    })
    return df


def _download_dataset() -> pd.DataFrame:
    """Attempt to download the AI4I 2020 dataset from the UCI repository."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        print(f"  Downloading dataset from UCI …")
        resp = requests.get(DATASET_URL, timeout=30)
        resp.raise_for_status()
        with open(DATASET_PATH, "wb") as f:
            f.write(resp.content)
        print(f"  Saved to {DATASET_PATH}")
        return pd.read_csv(DATASET_PATH)
    except Exception as exc:
        print(f"  Download failed ({exc}). Generating synthetic data instead …")
        df = _generate_synthetic_data()
        df.to_csv(DATASET_PATH, index=False)
        print(f"  Synthetic dataset saved to {DATASET_PATH}")
        return df


def load_dataset() -> pd.DataFrame:
    """Return the dataset: local file → UCI download → synthetic fallback."""
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return _download_dataset()


# ─── Preprocessing Pipeline ───────────────────────────────────────────────────

def preprocess_data(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: StandardScaler | None = None,
    encoder: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None, StandardScaler, LabelEncoder]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df      : Raw dataframe (may include target column).
    fit     : If True, fit new scaler/encoder and save them to disk.
              If False, use the provided scaler and encoder (inference mode).
    scaler  : Pre-fitted StandardScaler (required when fit=False).
    encoder : Pre-fitted LabelEncoder  (required when fit=False).

    Returns
    -------
    X       : Preprocessed feature DataFrame ready for model input.
    y       : Target array (or None if target column is absent).
    scaler  : The (fitted) StandardScaler.
    encoder : The (fitted) LabelEncoder.
    """
    df = df.copy()

    # Drop columns not used for modelling
    df.drop(columns=[c for c in RAW_DROP_COLS if c in df.columns], inplace=True)

    # ── Handle missing values ──────────────────────────────────────────────
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if "Type" in df.columns:
        df["Type"] = df["Type"].fillna("L")

    # ── Encode categorical feature ─────────────────────────────────────────
    if fit:
        encoder = LabelEncoder()
        df["Type"] = encoder.fit_transform(df["Type"].astype(str))
    else:
        df["Type"] = encoder.transform(df["Type"].astype(str))

    # ── Separate features and target ───────────────────────────────────────
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].values if TARGET_COLUMN in df.columns else None

    # ── Scale numeric features ─────────────────────────────────────────────
    if fit:
        scaler = StandardScaler()
        X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])
        # Persist artifacts for inference
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler,  SCALER_PATH)
        joblib.dump(encoder, ENCODER_PATH)
    else:
        X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])

    return X, y, scaler, encoder


def preprocess_single(input_dict: dict, scaler: StandardScaler, encoder: LabelEncoder) -> pd.DataFrame:
    """Convenience wrapper: preprocess a single prediction input dict."""
    df = pd.DataFrame([input_dict])
    _, _, _, _ = (None, None, None, None)  # silence linters
    return preprocess_data(df, fit=False, scaler=scaler, encoder=encoder)[0]
