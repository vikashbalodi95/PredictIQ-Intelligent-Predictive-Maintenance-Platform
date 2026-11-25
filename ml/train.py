"""
train.py
--------
Training pipeline for the CNC Predictive Maintenance Platform.

Run directly:
    python train.py

Trains Logistic Regression, Random Forest, and XGBoost; evaluates all
three; saves the best model (by ROC-AUC) and a JSON metadata file with
all metrics and ROC curve data for the analytics dashboard.
"""

import os
import json
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

from ml.config import (
    MODEL_DIR, DATA_DIR, MODEL_PATH, META_PATH,
    RANDOM_STATE, TEST_SIZE, FEATURE_COLUMNS,
)
from ml.preprocess import load_dataset, preprocess_data


# ─── Model Definitions ────────────────────────────────────────────────────────

def _build_models(scale_pos_weight: float) -> dict:
    """Return a dict of untrained models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute all classification metrics + ROC curve data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Downsample ROC arrays to ≤ 200 points (keeps JSON small)
    if len(fpr) > 200:
        idx = np.linspace(0, len(fpr) - 1, 200).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]

    return {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":          float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "fpr":              fpr.tolist(),
        "tpr":              tpr.tolist(),
    }


# ─── Main Training Function ───────────────────────────────────────────────────

def train_all_models() -> tuple[dict, str, object]:
    """
    Full training pipeline.

    Returns
    -------
    results    : dict  – per-model evaluation metrics
    best_name  : str   – name of the winning model
    best_model : object – trained model instance
    """
    print("\n═══════════════════════════════════════")
    print("  CNC Predictive Maintenance — Training")
    print("═══════════════════════════════════════\n")

    # ── Load & preprocess ─────────────────────────────────────────────────
    print("📂 Loading dataset …")
    df = load_dataset()
    print(f"   {len(df):,} rows | {df[' Machine failure' if ' Machine failure' in df.columns else 'Machine failure'].sum():,} failures")

    print("⚙️  Preprocessing …")
    X, y, scaler, encoder = preprocess_data(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Class imbalance ratio for XGBoost
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,} | Imbalance ratio: {scale_pos_weight:.1f}x\n")

    # ── Train & evaluate ──────────────────────────────────────────────────
    models = _build_models(scale_pos_weight)
    results: dict       = {}
    trained_models: dict = {}

    for name, model in models.items():
        print(f"🤖 Training {name} …", end="", flush=True)
        model.fit(X_train.values, y_train)
        metrics = _evaluate(model, X_test.values, y_test)
        results[name]        = metrics
        trained_models[name] = model
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}  |  F1: {metrics['f1']:.4f}  |  Recall: {metrics['recall']:.4f}")

    # ── Choose best model ─────────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]
    print(f"\n🏆 Best model: {best_name}  (ROC-AUC = {results[best_name]['roc_auc']:.4f})")

    # ── Save artifacts ────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    # Separate large ROC arrays from summary metrics to keep meta readable
    meta = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "class_ratio":   float(scale_pos_weight),
        "results": {
            name: {k: v for k, v in m.items() if k not in ("fpr", "tpr")}
            for name, m in results.items()
        },
        "roc_data": {
            name: {"fpr": m["fpr"], "tpr": m["tpr"]}
            for name, m in results.items()
        },
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📊 Metadata saved → {META_PATH}\n")

    print("✅ Training complete!\n")
    return results, best_name, best_model


if __name__ == "__main__":
    train_all_models()
