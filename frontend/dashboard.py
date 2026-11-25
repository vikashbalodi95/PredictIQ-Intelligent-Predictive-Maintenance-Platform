"""
frontend/dashboard.py
---------------------
The Streamlit dashboard — the human-facing front end.

Run it (with the backend already running) from the project root:
    streamlit run frontend/dashboard.py

Design
======
The dashboard talks to the FastAPI backend through `api_client` for everything
"live" (machines, readings, predictions, alerts, reports). For the offline
data-science pages (Analytics, Explainability) it loads the dataset and model
directly through the ML package, exactly like the original project did.

Pages (chosen in the sidebar):
    Live Monitor   – fleet overview + real-time charts for one machine
    Predict        – manual "what-if" prediction form
    Alerts         – alert feed with acknowledge buttons
    Analytics      – model metrics + dataset insights
    Explainability – SHAP feature attributions
    Reports        – download a PDF maintenance report

Everything is heavily commented so each Streamlit widget's purpose is clear.
"""

import time

import pandas as pd
import streamlit as st

from core.settings import settings
from frontend import api_client as api

# ─── Page setup (must be the first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="CNC Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A little CSS to make the status cards look like a real monitoring console.
st.markdown("""
<style>
.block-container { padding: 1.5rem 2.2rem 2rem; }
.tile {
    border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: .6rem;
    border: 1px solid rgba(255,255,255,.08);
    background: linear-gradient(135deg,#1e2130,#252840);
}
.tile h4 { margin: 0 0 .3rem 0; font-size: 1rem; }
.tile .v { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
.tile .s { font-size: .8rem; color:#9ca3af; }
.badge { display:inline-block; padding:.15rem .7rem; border-radius:999px;
         font-weight:600; font-size:.8rem; }
.b-Healthy  { background:#1b4332; color:#6ee7b7; }
.b-Warning  { background:#451a03; color:#fcd34d; }
.b-Critical { background:#450a0a; color:#fca5a5; }
.sec { font-size:1.25rem; font-weight:700; border-bottom:2px solid #1e88e5;
       padding-bottom:.4rem; margin:.5rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

STATUS_EMOJI = {"Healthy": "🟢", "Warning": "🟡", "Critical": "🔴"}


# ─── Small reusable UI helpers ────────────────────────────────────────────────

def status_badge(status: str) -> str:
    """Return an HTML pill for a status string."""
    return f'<span class="badge b-{status}">{STATUS_EMOJI.get(status,"")} {status}</span>'


def tile(title: str, value: str, sub: str = "") -> str:
    """Return one HTML metric tile."""
    return f'<div class="tile"><h4>{title}</h4><div class="v">{value}</div><div class="s">{sub}</div></div>'


# ─── Cached, OFFLINE resources (dataset + model) for analytics pages ──────────
# These mirror the original app: analytics and SHAP read the dataset/model
# directly rather than going through the API.

@st.cache_resource(show_spinner=False)
def _load_predictor():
    from ml.predictor import CNCPredictor
    return CNCPredictor()


@st.cache_data(show_spinner=False)
def _load_dataset():
    from ml.preprocess import load_dataset
    return load_dataset()


@st.cache_data(show_spinner=False)
def _load_meta():
    import json
    from ml.config import META_PATH
    with open(META_PATH) as f:
        return json.load(f)


# ─── Sidebar: navigation + backend status + auto-refresh ──────────────────────

def sidebar() -> tuple[str, bool, int]:
    with st.sidebar:
        st.markdown("## ⚙️ CNC Predictive\nMaintenance Platform")
        st.caption(f"v{settings.APP_VERSION} · ML + Software Engineering")
        st.markdown("---")

        pages = ["📡 Live Monitor", "🔮 Predict", "🚨 Alerts",
                 "📊 Analytics", "🔍 Explainability", "📄 Reports"]
        choice = st.radio("Navigate", pages, label_visibility="collapsed")

        st.markdown("---")
        # Backend connection indicator — vital for a client/server app.
        if api.health():
            st.success("✅ Backend connected")
        else:
            st.error("❌ Backend offline")
            st.caption(f"Expected at {settings.API_BASE_URL}")
            st.caption("Start it with:\n`uvicorn backend.main:app`")

        st.markdown("---")
        # Auto-refresh controls (used by the live pages).
        auto = st.checkbox("🔄 Auto-refresh", value=True)
        every = st.slider("Refresh every (sec)", 2, 30, 5)
        st.markdown("---")
        st.caption("Streamlit · FastAPI · PostgreSQL · XGBoost")

    return choice.split(" ", 1)[1], auto, every


# ─── Page: Live Monitor ───────────────────────────────────────────────────────

def page_live(auto: bool, every: int):
    st.markdown('<div class="sec">📡 Live Fleet Monitor</div>', unsafe_allow_html=True)

    try:
        machines = api.list_machines()
    except api.APIError:
        st.warning("Cannot reach the backend. Start it and (optionally) the simulator.")
        return

    if not machines:
        st.info("No machines yet. Run `python -m scripts.init_db` to seed some.")
        return

    # ── Fleet overview row: one live tile per machine ─────────────────────────
    st.markdown("#### Fleet overview")
    cols = st.columns(len(machines))
    for col, m in zip(cols, machines):
        with col:
            latest = api.get_latest(m["id"])
            pred = latest.get("prediction")
            if pred:
                st.markdown(
                    tile(
                        f'{m["machine_id"]} · {m["name"]}',
                        f'{pred["health_score"]}<span class="s"> /100</span>',
                        f'{status_badge(pred["status"])} · RUL {pred["rul_minutes"]}m',
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    tile(f'{m["machine_id"]} · {m["name"]}', "—", "no data yet"),
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Per-machine detail: pick one machine to drill into ────────────────────
    labels = {f'{m["machine_id"]} — {m["name"]}': m["id"] for m in machines}
    pick = st.selectbox("Inspect machine", list(labels.keys()))
    pk = labels[pick]

    readings = api.get_readings(pk, limit=60)
    preds = api.get_predictions(pk, limit=60)

    if not readings:
        st.info("No readings for this machine yet. Start the simulator: "
                "`python -m simulator.simulator`")
    else:
        # Oldest → newest for time-series plotting.
        rdf = pd.DataFrame(readings).iloc[::-1].reset_index(drop=True)
        rdf["timestamp"] = pd.to_datetime(rdf["timestamp"])

        latest = api.get_latest(pk)
        pred = latest.get("prediction") or {}

        # Top metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", pred.get("status", "—"))
        c2.metric("Failure probability", f'{pred.get("failure_probability",0):.0%}')
        c3.metric("Health score", f'{pred.get("health_score","—")}/100')
        c4.metric("Remaining useful life", f'{pred.get("rul_minutes","—")} min',
                  help=pred.get("rul_band", ""))

        # Sensor trend charts (Streamlit's built-in line_chart is perfect here)
        st.markdown("##### Sensor trends")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("Tool wear [min]")
            st.line_chart(rdf.set_index("timestamp")[["tool_wear"]], height=200)
            st.caption("Torque [Nm]")
            st.line_chart(rdf.set_index("timestamp")[["torque"]], height=200)
        with cc2:
            st.caption("Temperatures [K]")
            st.line_chart(rdf.set_index("timestamp")[["air_temp", "process_temp"]],
                          height=200)
            st.caption("Rotational speed [rpm]")
            st.line_chart(rdf.set_index("timestamp")[["rotational_speed"]], height=200)

        # Prediction trends
        if preds:
            pdf = pd.DataFrame(preds).iloc[::-1].reset_index(drop=True)
            pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
            st.markdown("##### Health & failure-probability history")
            ch1, ch2 = st.columns(2)
            with ch1:
                st.caption("Health score (0–100)")
                st.line_chart(pdf.set_index("timestamp")[["health_score"]], height=200)
            with ch2:
                st.caption("Failure probability")
                st.line_chart(pdf.set_index("timestamp")[["failure_probability"]],
                              height=200)

        # Recent alerts for this machine
        alerts = api.get_alerts(pk, limit=6)
        if alerts:
            st.markdown("##### Recent alerts")
            for a in alerts:
                icon = {"Critical": "🔴", "Warning": "🟡", "Info": "🔵"}.get(a["severity"], "⚪")
                st.write(f'{icon} **{a["title"]}** — {a["message"]}')

    # Auto-refresh: simple, dependency-free. When enabled, wait then rerun.
    if auto:
        time.sleep(every)
        st.rerun()


# ─── Page: Predict (what-if) ──────────────────────────────────────────────────

def page_predict():
    st.markdown('<div class="sec">🔮 Manual Prediction (what-if)</div>',
                unsafe_allow_html=True)
    st.caption("Enter sensor values and get an instant model verdict. "
               "This does not store anything in the database.")

    c1, c2 = st.columns(2)
    with c1:
        mtype = st.selectbox("Machine type", ["L", "M", "H"])
        air = st.number_input("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
        proc = st.number_input("Process temperature [K]", 305.0, 315.0, 310.0, 0.1)
    with c2:
        rpm = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500, 10)
        torque = st.number_input("Torque [Nm]", 0.0, 80.0, 40.0, 0.5)
        wear = st.number_input("Tool wear [min]", 0, 260, 100, 1)

    if st.button("Run prediction", type="primary"):
        try:
            r = api.predict({
                "type": mtype, "air_temp": air, "process_temp": proc,
                "rotational_speed": rpm, "torque": torque, "tool_wear": wear,
            })
        except api.APIError as e:
            st.error(f"Backend error: {e}")
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction", "FAILURE" if r["prediction"] else "OK")
        m2.metric("Probability", f'{r["probability"]:.1%}')
        m3.metric("Health", f'{r["health_score"]}/100')
        m4.metric("RUL", f'{r["rul_minutes"]} min')
        st.markdown(status_badge(r["status"]), unsafe_allow_html=True)
        st.info(f'**RUL band: {r["rul_band"]}** — {r["rul_explanation"]}')


# ─── Page: Alerts ─────────────────────────────────────────────────────────────

def page_alerts(auto: bool, every: int):
    st.markdown('<div class="sec">🚨 Alerts</div>', unsafe_allow_html=True)
    only_new = st.checkbox("Show only unacknowledged", value=True)

    try:
        alerts = api.get_alerts(limit=100, unacknowledged_only=only_new)
    except api.APIError:
        st.warning("Backend offline.")
        return

    if not alerts:
        st.success("No alerts 🎉")
    else:
        # Build a machine_id lookup so alerts show a friendly code.
        m_lookup = {m["id"]: m["machine_id"] for m in api.list_machines()}
        for a in alerts:
            icon = {"Critical": "🔴", "Warning": "🟡", "Info": "🔵"}.get(a["severity"], "⚪")
            cols = st.columns([6, 1])
            with cols[0]:
                st.write(f'{icon} **[{a["severity"]}] {a["title"]}** '
                         f'· {m_lookup.get(a["machine_id"], a["machine_id"])}')
                st.caption(a["message"])
            with cols[1]:
                if not a["acknowledged"]:
                    if st.button("Ack", key=f'ack-{a["id"]}'):
                        api.acknowledge_alert(a["id"])
                        st.rerun()
                else:
                    st.caption("✓ acked")

    if auto:
        time.sleep(every)
        st.rerun()


# ─── Page: Analytics (offline, dataset + model metrics) ───────────────────────

def page_analytics():
    st.markdown('<div class="sec">📊 Model & Data Analytics</div>',
                unsafe_allow_html=True)
    try:
        meta = _load_meta()
        df = _load_dataset()
    except Exception:
        st.warning("Model not trained yet. Run `python -m ml.train` first.")
        return

    from frontend.visualization import (
        failure_distribution_chart, correlation_heatmap,
        model_comparison_chart, roc_curve_chart, confusion_matrix_chart,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", meta["best_model"])
    best = meta["results"][meta["best_model"]]
    c2.metric("ROC-AUC", f'{best["roc_auc"]:.3f}')
    c3.metric("Recall", f'{best["recall"]:.3f}')

    st.plotly_chart(model_comparison_chart(meta["results"]), use_container_width=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.plotly_chart(roc_curve_chart(meta["roc_data"]), use_container_width=True)
    with cc2:
        st.plotly_chart(
            confusion_matrix_chart(best["confusion_matrix"], meta["best_model"]),
            use_container_width=True)

    st.markdown("##### Dataset")
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(failure_distribution_chart(df), use_container_width=True)
    with d2:
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)


# ─── Page: Explainability (SHAP, offline) ─────────────────────────────────────

def page_explainability():
    st.markdown('<div class="sec">🔍 Explainability (SHAP)</div>',
                unsafe_allow_html=True)
    try:
        predictor = _load_predictor()
        df = _load_dataset()
    except Exception:
        st.warning("Model not trained yet. Run `python -m ml.train` first.")
        return

    from ml.preprocess import preprocess_data
    from ml.config import FEATURE_COLUMNS
    from frontend.visualization import shap_summary_plot

    st.caption("SHAP shows how each sensor pushes the model toward or away from "
               "predicting a failure. Computed on a sample of the dataset.")
    sample = df.sample(min(300, len(df)), random_state=42)
    X, _, _, _ = preprocess_data(sample, fit=False,
                                 scaler=predictor.scaler, encoder=predictor.encoder)
    with st.spinner("Computing SHAP values …"):
        fig = shap_summary_plot(predictor.model, X, FEATURE_COLUMNS)
    st.pyplot(fig)


# ─── Page: Reports ────────────────────────────────────────────────────────────

def page_reports():
    st.markdown('<div class="sec">📄 PDF Maintenance Reports</div>',
                unsafe_allow_html=True)
    try:
        machines = api.list_machines()
    except api.APIError:
        st.warning("Backend offline.")
        return

    labels = {f'{m["machine_id"]} — {m["name"]}': m["id"] for m in machines}
    pick = st.selectbox("Machine", list(labels.keys()))
    pk = labels[pick]

    st.caption("Generates a one-page PDF with the machine's latest status, "
               "readings, prediction, RUL, recommendations and recent alerts.")
    if st.button("Generate report", type="primary"):
        try:
            pdf = api.get_report_bytes(pk)
            st.download_button("⬇️ Download PDF", data=pdf,
                               file_name=f"report_{pick.split(' ')[0]}.pdf",
                               mime="application/pdf")
            st.success("Report ready.")
        except Exception as e:
            st.error(f"Could not generate report: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    page, auto, every = sidebar()
    if page == "Live Monitor":
        page_live(auto, every)
    elif page == "Predict":
        page_predict()
    elif page == "Alerts":
        page_alerts(auto, every)
    elif page == "Analytics":
        page_analytics()
    elif page == "Explainability":
        page_explainability()
    elif page == "Reports":
        page_reports()


if __name__ == "__main__":
    main()
