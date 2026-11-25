"""
ml/predictor.py
---------------
CNCPredictor: loads the trained model artifacts and provides a clean
inference API. This is the single object the backend uses to score sensor
readings.

Compared to the original project, the ONLY additions are:
  * it now also returns a Remaining-Useful-Life estimate (ml/rul.py), and
  * a tiny `predict_with_rul` convenience used by the API.
The core model logic, health score and status are unchanged.
"""

import json
import os

import joblib
import pandas as pd

from ml.config import (
    MODEL_PATH, SCALER_PATH, ENCODER_PATH, META_PATH,
    FEATURE_COLUMNS, NUMERIC_FEATURES, THRESHOLDS, STATUS_THRESHOLDS,
)
from ml.rul import estimate_rul


class CNCPredictor:
    """
    Encapsulates the trained model plus its preprocessing artifacts.

    Usage
    -----
    predictor = CNCPredictor()
    result = predictor.predict({
        'Type': 'L',
        'Air temperature [K]': 300.0,
        'Process temperature [K]': 310.5,
        'Rotational speed [rpm]': 1500,
        'Torque [Nm]': 42.0,
        'Tool wear [min]': 120,
    })
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained model not found. Run `python -m ml.train` first."
            )
        self.model   = joblib.load(MODEL_PATH)
        self.scaler  = joblib.load(SCALER_PATH)
        self.encoder = joblib.load(ENCODER_PATH)

        with open(META_PATH) as f:
            self.meta = json.load(f)

    # ─── Core Prediction ──────────────────────────────────────────────────────

    def _preprocess(self, input_dict: dict) -> pd.DataFrame:
        """Transform a single raw input dict into a model-ready DataFrame."""
        df = pd.DataFrame([input_dict])
        df["Type"] = self.encoder.transform(df["Type"].astype(str))
        df[NUMERIC_FEATURES] = self.scaler.transform(df[NUMERIC_FEATURES])
        return df[FEATURE_COLUMNS]

    def predict(self, input_dict: dict) -> dict:
        """
        Full prediction pipeline for a single sample.

        Returns
        -------
        dict with keys: prediction, probability, health_score, status, rul,
                        input_data, processed_features
        """
        X = self._preprocess(input_dict)

        prediction  = int(self.model.predict(X.values)[0])
        probability = float(self.model.predict_proba(X.values)[0][1])

        health_score = self._compute_health_score(input_dict, probability)
        status       = self._get_status(probability)

        # NEW: remaining useful life (simple heuristic, see ml/rul.py)
        rul = estimate_rul(
            tool_wear=input_dict.get("Tool wear [min]", 0),
            failure_probability=probability,
            health_score=health_score,
        )

        return {
            "prediction":         prediction,
            "probability":        probability,
            "health_score":       health_score,
            "status":             status,
            "rul":                rul,
            "input_data":         input_dict,
            "processed_features": X,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run predictions on every row of a DataFrame.
        Columns must match FEATURE_COLUMNS (Type + 5 numeric sensors).
        """
        records = []
        for _, row in df.iterrows():
            input_dict = {col: row[col] for col in FEATURE_COLUMNS if col in row.index}
            if "Type" not in input_dict:
                input_dict["Type"] = "L"
            r = self.predict(input_dict)
            records.append({
                **input_dict,
                "Failure Prediction": r["prediction"],
                "Failure Probability (%)": round(r["probability"] * 100, 2),
                "Health Score": r["health_score"],
                "Status": r["status"],
                "RUL (min)": r["rul"]["rul_minutes"],
            })
        return pd.DataFrame(records)

    # ─── Health Score ─────────────────────────────────────────────────────────

    def _compute_health_score(self, data: dict, failure_prob: float) -> int:
        """
        Compute an integer health score in [0, 100].

        Base score = (1 − failure_probability) × 100.
        Additional penalties are applied for individual sensor threshold
        breaches so that a 'borderline' machine is penalised even when the
        overall probability is moderate.
        """
        base = (1.0 - failure_prob) * 100.0

        penalties = 0
        if data.get("Tool wear [min]", 0)         > THRESHOLDS["tool_wear"]:
            penalties += 10
        if data.get("Torque [Nm]", 0)             > THRESHOLDS["torque"]:
            penalties += 8
        if data.get("Process temperature [K]", 0) > THRESHOLDS["process_temp"]:
            penalties += 7
        if data.get("Rotational speed [rpm]", 0)  > THRESHOLDS["rotational_speed"]:
            penalties += 5
        temp_diff = (
            data.get("Process temperature [K]", 310)
            - data.get("Air temperature [K]", 300)
        )
        if temp_diff < THRESHOLDS["temp_diff_min"]:
            penalties += 5

        return max(0, min(100, int(base - penalties)))

    # ─── Status ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_status(probability: float) -> str:
        if probability < STATUS_THRESHOLDS["healthy"]:
            return "Healthy"
        elif probability < STATUS_THRESHOLDS["warning"]:
            return "Warning"
        return "Critical"

    # ─── Metadata Accessors ───────────────────────────────────────────────────

    @property
    def best_model_name(self) -> str:
        return self.meta.get("best_model", "Unknown")

    @property
    def model_results(self) -> dict:
        return self.meta.get("results", {})

    @property
    def roc_data(self) -> dict:
        return self.meta.get("roc_data", {})
