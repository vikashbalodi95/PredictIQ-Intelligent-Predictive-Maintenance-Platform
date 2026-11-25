"""
utils/alerting.py
-----------------
The alerting engine.

Given a sensor reading and the ML result for it, this module decides whether
anything is worth notifying an operator about, and returns a list of alert
dicts. The backend then stores those in the `alerts` table.

Design: alerts are pure functions of (reading + prediction). We keep the rules
here — separate from the ML model and the database — so the logic is easy to
read, test, and tweak without touching anything else.

An alert dict has the shape:
    {"severity": "Critical"|"Warning"|"Info", "title": str, "message": str}
"""

from ml.config import THRESHOLDS, STATUS_THRESHOLDS


def evaluate_alerts(features: dict, result: dict) -> list[dict]:
    """
    Produce alerts for one reading.

    Parameters
    ----------
    features : the ML input dict (bracketed AI4I keys).
    result   : the dict returned by CNCPredictor.predict().

    Returns
    -------
    A list of alert dicts (possibly empty if everything looks fine).
    """
    alerts: list[dict] = []
    prob = result["probability"]

    # ── 1) Model-probability alerts ───────────────────────────────────────────
    if prob >= STATUS_THRESHOLDS["warning"]:        # >= 0.60 → Critical
        alerts.append({
            "severity": "Critical",
            "title": "High failure probability",
            "message": (
                f"Model predicts {prob:.0%} chance of failure. "
                f"Estimated remaining life: {result['rul']['rul_minutes']} min. "
                "Halt and inspect the machine."
            ),
        })
    elif prob >= STATUS_THRESHOLDS["healthy"]:      # >= 0.30 → Warning
        alerts.append({
            "severity": "Warning",
            "title": "Elevated failure risk",
            "message": (
                f"Failure probability is {prob:.0%}. "
                "Schedule inspection at the next maintenance window."
            ),
        })

    # ── 2) Individual sensor-threshold alerts ─────────────────────────────────
    tool_wear = features.get("Tool wear [min]", 0)
    if tool_wear > THRESHOLDS["tool_wear"]:
        alerts.append({
            "severity": "Critical",
            "title": "Tool wear over limit",
            "message": f"Tool wear {tool_wear:.0f} min exceeds "
                       f"{THRESHOLDS['tool_wear']} min — replace the cutting tool.",
        })

    if features.get("Torque [Nm]", 0) > THRESHOLDS["torque"]:
        alerts.append({
            "severity": "Warning",
            "title": "Excessive torque",
            "message": f"Torque {features['Torque [Nm]']:.1f} Nm exceeds "
                       f"{THRESHOLDS['torque']} Nm — check spindle and drive.",
        })

    if features.get("Process temperature [K]", 0) > THRESHOLDS["process_temp"]:
        alerts.append({
            "severity": "Warning",
            "title": "High process temperature",
            "message": f"Process temp {features['Process temperature [K]']:.1f} K exceeds "
                       f"{THRESHOLDS['process_temp']} K — inspect cooling system.",
        })

    if features.get("Rotational speed [rpm]", 0) > THRESHOLDS["rotational_speed"]:
        alerts.append({
            "severity": "Info",
            "title": "Over-speed condition",
            "message": f"Spindle speed {features['Rotational speed [rpm]']:.0f} rpm exceeds "
                       f"{THRESHOLDS['rotational_speed']} rpm — verify tool RPM rating.",
        })

    return alerts
