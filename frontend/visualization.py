"""
visualization.py
----------------
All chart and plot functions for the CNC Predictive Maintenance dashboard.
Returns Plotly figures (for st.plotly_chart) or Matplotlib figures (for st.pyplot).
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import shap

from ml.config import NUMERIC_FEATURES, TARGET_COLUMN, COLORS

# ─── Shared Plotly Layout ─────────────────────────────────────────────────────

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,0.6)",
    font=dict(color="#E5E7EB", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
    margin=dict(t=60, b=40, l=40, r=30),
)

_MODEL_PALETTE = {
    "Logistic Regression": "#1E88E5",
    "Random Forest":       "#43A047",
    "XGBoost":             "#FB8C00",
}


def _apply_base_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply the shared dark layout to any Plotly figure."""
    fig.update_layout(**_BASE_LAYOUT, title=dict(text=title, font=dict(size=17)), height=height)
    return fig


# ─── Data Analysis Charts ─────────────────────────────────────────────────────

def failure_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Normal", "Failure"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.58,
        marker=dict(colors=[COLORS["success"], COLORS["danger"]],
                    line=dict(color="#0E1117", width=3)),
        textinfo="label+percent+value",
        textfont=dict(size=13),
        pull=[0, 0.04],
    ))
    fig.add_annotation(
        text=f"<b>{sum(values):,}</b><br>Total",
        x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#E5E7EB"
    )
    return _apply_base_layout(fig, "Failure Distribution", height=400)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = NUMERIC_FEATURES + [TARGET_COLUMN]
    corr = df[cols].corr().round(3)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="r", thickness=14),
    ))
    return _apply_base_layout(fig, "Feature Correlation Heatmap", height=480)


def feature_distributions(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=NUMERIC_FEATURES,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    shown_legend = set()

    for i, feat in enumerate(NUMERIC_FEATURES):
        row, col = divmod(i, 3)
        for label, val, color in [("Normal", 0, COLORS["success"]), ("Failure", 1, COLORS["danger"])]:
            series = df[df[TARGET_COLUMN] == val][feat]
            show = label not in shown_legend
            fig.add_trace(
                go.Histogram(
                    x=series, name=label,
                    marker_color=color, opacity=0.72,
                    showlegend=show, legendgroup=label,
                    nbinsx=30,
                ),
                row=row + 1, col=col + 1,
            )
            shown_legend.add(label)

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="overlay",
        title=dict(text="Feature Distributions — Normal vs Failure", font=dict(size=17)),
        height=560,
        legend=dict(orientation="h", y=-0.06),
    )
    return fig


def feature_vs_failure_boxplot(df: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.box(
        df, x=TARGET_COLUMN, y=feature,
        color=TARGET_COLUMN,
        color_discrete_map={0: COLORS["success"], 1: COLORS["danger"]},
        labels={TARGET_COLUMN: "Machine Failure", feature: feature},
        category_orders={TARGET_COLUMN: [0, 1]},
    )
    fig.update_traces(boxmean=True)
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"{feature} — Distribution by Failure Status", font=dict(size=17)),
        showlegend=False,
        height=400,
        xaxis=dict(
            ticktext=["Normal (0)", "Failure (1)"],
            tickvals=[0, 1],
            **_BASE_LAYOUT["xaxis"],
        ),
    )
    return fig


# ─── Model Performance Charts ─────────────────────────────────────────────────

def model_comparison_chart(results: dict) -> go.Figure:
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    fig = go.Figure()

    for name, metrics in results.items():
        values = [metrics.get(k, 0) for k in metric_keys]

        fig.add_trace(
            go.Bar(
                name=name,
                x=metric_labels,
                y=values,
                marker_color=_MODEL_PALETTE.get(name, COLORS["primary"]),
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                textfont=dict(size=11),
            )
        )

    layout = _BASE_LAYOUT.copy()
    layout["yaxis"] = {
        **_BASE_LAYOUT["yaxis"],
        "range": [0, 1.18],
    }

    fig.update_layout(
        **layout,
        barmode="group",
        title=dict(text="Model Comparison — All Metrics", font=dict(size=17)),
        legend=dict(orientation="h", y=-0.12),
        height=460,
    )

    return fig

def roc_curve_chart(roc_data: dict) -> go.Figure:
    """
    roc_data: {model_name: {'fpr': [...], 'tpr': [...], 'roc_auc': float}}
    """
    fig = go.Figure()

    for name, data in roc_data.items():
        auc = data.get("roc_auc", 0)
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            name=f"{name}  (AUC = {auc:.3f})",
            mode="lines",
            line=dict(color=_MODEL_PALETTE.get(name, COLORS["primary"]), width=2.5),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random Classifier",
        mode="lines",
        line=dict(color="gray", dash="dash", width=1.5),
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="ROC Curves — All Models", font=dict(size=17)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.08, bgcolor="rgba(0,0,0,0.4)"),
        height=460,
    )
    return fig


def confusion_matrix_chart(cm: list, model_name: str) -> go.Figure:
    cm_arr = np.array(cm)
    labels = ["Normal", "Failure"]

    # Normalised percentages for text
    total_per_row = cm_arr.sum(axis=1, keepdims=True)
    pct = (cm_arr / total_per_row * 100).round(1)

    text = [[f"<b>{cm_arr[r][c]}</b><br>({pct[r][c]}%)" for c in range(2)] for r in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=16),
        showscale=False,
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=17)),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=380,
    )
    return fig


def feature_importance_chart(model, feature_names: list) -> go.Figure:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        source = "Gini / Gain Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        source = "|Coefficient| — Logistic Regression"
    else:
        return go.Figure()

    idx = np.argsort(importances)
    sorted_names  = [feature_names[i] for i in idx]
    sorted_values = importances[idx]

    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_names,
        orientation="h",
        marker=dict(
            color=sorted_values,
            colorscale="Blues",
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        text=[f"{v:.4f}" for v in sorted_values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"Feature Importance  ({source})", font=dict(size=17)),
        xaxis_title="Importance Score",
        height=380,
    )
    return fig


# ─── SHAP Visualizations ──────────────────────────────────────────────────────

def _get_shap_values(model, X: pd.DataFrame):
    """
    Return (shap_values_array, expected_value) for the positive class.
    Handles tree, linear, and kernel explainers automatically.
    """
    model_type = type(model).__name__

    if model_type in ("XGBClassifier", "RandomForestClassifier",
                      "GradientBoostingClassifier", "ExtraTreesClassifier"):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Random Forest returns a list [class_0_vals, class_1_vals]
        if isinstance(shap_values, list):
            ev = explainer.expected_value
            return shap_values[1], (ev[1] if hasattr(ev, "__len__") else ev)
        return shap_values, explainer.expected_value

    elif model_type == "LogisticRegression":
        bg          = shap.maskers.Independent(X, max_samples=min(100, len(X)))
        explainer   = shap.LinearExplainer(model, bg)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            ev = explainer.expected_value
            return shap_values[1], (ev[1] if hasattr(ev, "__len__") else ev)
        return shap_values, explainer.expected_value

    else:
        bg          = shap.sample(X, min(50, len(X)))
        explainer   = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X[:50])
        if isinstance(shap_values, list):
            return shap_values[1], explainer.expected_value[1]
        return shap_values, explainer.expected_value


def _dark_mpl_fig(w=10, h=5) -> tuple:
    """Return a dark-background Matplotlib figure and axis."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    return fig, ax


def shap_summary_plot(model, X: pd.DataFrame, feature_names: list):
    """Beeswarm SHAP summary plot (Matplotlib). Shows global feature impact."""
    warnings.filterwarnings("ignore")
    plt.style.use("dark_background")
    try:
        shap_vals, _ = _get_shap_values(model, X)

        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_facecolor("#0E1117")
        shap.summary_plot(
            shap_vals, X,
            feature_names=feature_names,
            plot_type="dot",
            max_display=10,
            show=False,
            color_bar_label="Feature Value",
        )
        fig = plt.gcf()
        fig.patch.set_facecolor("#0E1117")
        plt.tight_layout()
        return fig

    except Exception as exc:
        fig, ax = _dark_mpl_fig(8, 4)
        ax.text(0.5, 0.5, f"SHAP summary unavailable.\n{type(exc).__name__}: {exc}",
                ha="center", va="center", color="white", fontsize=11,
                transform=ax.transAxes, wrap=True)
        ax.axis("off")
        return fig
    finally:
        warnings.filterwarnings("default")


def shap_waterfall_plot(model, X_sample: pd.DataFrame, feature_names: list):
    """SHAP waterfall plot for a single sample (Matplotlib)."""
    warnings.filterwarnings("ignore")
    plt.style.use("dark_background")
    try:
        model_type = type(model).__name__

        # Use the new Explanation API for waterfall (needs TreeExplainer / LinearExplainer)
        if model_type in ("XGBClassifier", "RandomForestClassifier",
                          "GradientBoostingClassifier"):
            explainer = shap.TreeExplainer(model)
            shap_exp  = explainer(X_sample)

            # RF returns 3-D (samples, features, classes) — extract class 1
            if len(np.array(shap_exp.values).shape) == 3:
                vals  = np.array(shap_exp.values)[:, :, 1]
                base  = np.array(shap_exp.base_values)
                base  = base[:, 1] if base.ndim > 1 else base
                shap_exp = shap.Explanation(
                    values=vals, base_values=base,
                    data=np.array(shap_exp.data),
                    feature_names=feature_names,
                )

        elif model_type == "LogisticRegression":
            bg        = shap.maskers.Independent(X_sample, max_samples=len(X_sample))
            explainer = shap.LinearExplainer(model, bg)
            shap_exp  = explainer(X_sample)

        else:
            # Fallback: bar summary for the single sample
            shap_vals, ev = _get_shap_values(model, X_sample)
            fig = plt.figure(figsize=(10, 5))
            fig.patch.set_facecolor("#0E1117")
            shap.summary_plot(shap_vals[:1], X_sample[:1],
                              feature_names=feature_names,
                              plot_type="bar", show=False)
            fig = plt.gcf()
            fig.patch.set_facecolor("#0E1117")
            plt.tight_layout()
            return fig

        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor("#0E1117")
        shap.plots.waterfall(shap_exp[0], show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor("#0E1117")
        plt.tight_layout()
        return fig

    except Exception as exc:
        fig, ax = _dark_mpl_fig(8, 4)
        ax.text(0.5, 0.5,
                f"Waterfall plot unavailable.\n{type(exc).__name__}: {exc}",
                ha="center", va="center", color="white", fontsize=11,
                transform=ax.transAxes, wrap=True)
        ax.axis("off")
        return fig
    finally:
        warnings.filterwarnings("default")
