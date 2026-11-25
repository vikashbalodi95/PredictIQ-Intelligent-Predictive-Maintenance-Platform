"""
recommender.py
--------------
Rule-based maintenance recommendation engine.
Combines sensor threshold violations with overall failure probability
to generate actionable, prioritised maintenance advice.
"""

from ml.config import THRESHOLDS

# ─── Rule Definitions ─────────────────────────────────────────────────────────
# Each rule: condition (callable), severity, icon, title, recommendation, details

_RULES = [
    {
        "condition":      lambda d: d.get("Tool wear [min]", 0) > THRESHOLDS["tool_wear"],
        "severity":       "Critical",
        "icon":           "🔧",
        "title":          "Critical Tool Wear",
        "recommendation": "Replace cutting tool immediately to prevent catastrophic failure.",
        "details": [
            "Tool wear exceeds the 200-minute threshold — tool failure is imminent.",
            "Schedule a tool change at the next available stop.",
            "Inspect the tool holder and collet for runout or damage.",
            "Review tool-life management parameters in the CNC controller.",
            "Consider upgrading to a higher-grade carbide insert for this material.",
        ],
    },
    {
        "condition":      lambda d: 150 < d.get("Tool wear [min]", 0) <= THRESHOLDS["tool_wear"],
        "severity":       "Warning",
        "icon":           "⏱️",
        "title":          "Elevated Tool Wear",
        "recommendation": "Plan tool replacement within the next maintenance cycle.",
        "details": [
            "Tool wear is approaching the critical threshold.",
            "Monitor surface finish quality — degradation signals tool failure.",
            "Prepare a replacement tool to minimise changeover downtime.",
        ],
    },
    {
        "condition":      lambda d: d.get("Torque [Nm]", 0) > THRESHOLDS["torque"],
        "severity":       "Warning",
        "icon":           "⚙️",
        "title":          "Excessive Torque",
        "recommendation": "Inspect spindle bearings and check drive system integrity.",
        "details": [
            "Torque exceeds 60 Nm — spindle overstrain is a risk.",
            "Check spindle bearing preload and lubrication levels.",
            "Inspect drive belt or coupling for wear or slippage.",
            "Verify cutting parameters: reduce feed rate or depth of cut.",
            "Check for tool runout or imbalance contributing to excess load.",
        ],
    },
    {
        "condition":      lambda d: d.get("Process temperature [K]", 0) > THRESHOLDS["process_temp"],
        "severity":       "Warning",
        "icon":           "🌡️",
        "title":          "High Process Temperature",
        "recommendation": "Inspect cooling system and verify cutting fluid supply.",
        "details": [
            "Process temperature above 313 K — heat dissipation is insufficient.",
            "Check coolant nozzle position, flow rate, and pressure.",
            "Measure coolant concentration — degraded fluid reduces cooling.",
            "Inspect the heat exchanger and coolant reservoir for blockages.",
            "Verify that cutting speed is within the recommended SFM range.",
        ],
    },
    {
        "condition":      lambda d: d.get("Rotational speed [rpm]", 0) > THRESHOLDS["rotational_speed"],
        "severity":       "Warning",
        "icon":           "🔄",
        "title":          "Over-Speed Condition",
        "recommendation": "Reduce spindle RPM to the manufacturer-recommended range.",
        "details": [
            "Rotational speed exceeds 2,500 RPM — bearing fatigue risk increases.",
            "Verify spindle speed against the cutting tool's maximum RPM rating.",
            "Check for vibration or chatter — reduce speed if machining is unstable.",
            "Inspect spindle drive inverter for fault codes.",
            "Review part material machinability data and adjust surface speed.",
        ],
    },
    {
        "condition": lambda d: (
            (d.get("Process temperature [K]", 310) - d.get("Air temperature [K]", 300))
            < THRESHOLDS["temp_diff_min"]
            and d.get("Rotational speed [rpm]", 1500) < 1400
        ),
        "severity":       "Info",
        "icon":           "❄️",
        "title":          "Low Thermal Differential",
        "recommendation": "Check heat dissipation — possible cooling system inefficiency.",
        "details": [
            "Process-to-air temperature difference is below the 8.6 K baseline.",
            "Combined with low RPM, this pattern matches heat dissipation failure mode.",
            "Inspect coolant pump operation and flow rate.",
            "Monitor this parameter trend over the next 10 production cycles.",
        ],
    },
]

_SEVERITY_ORDER = {"Critical": 0, "Warning": 1, "Info": 2, "Healthy": 3}

_HEALTHY_CARD = {
    "severity":       "Healthy",
    "icon":           "✅",
    "title":          "Machine Operating Normally",
    "recommendation": "No immediate action required.",
    "details": [
        "All sensor readings are within normal operating ranges.",
        "Continue with the scheduled preventive maintenance programme.",
        "Log current readings as baseline for trend analysis.",
        "Next full inspection: refer to your PM calendar.",
    ],
}


# ─── Public API ───────────────────────────────────────────────────────────────

def get_recommendations(input_data: dict, failure_probability: float) -> list[dict]:
    """
    Generate prioritised maintenance recommendations.

    Parameters
    ----------
    input_data          : Raw sensor dict (same keys as FEATURE_COLUMNS).
    failure_probability : Model output probability for class=1 (failure).

    Returns
    -------
    List of recommendation dicts, sorted by severity (Critical → Info → Healthy).
    """
    recs = []

    # High-probability alert card (always shown when prob is elevated)
    if failure_probability >= 0.60:
        recs.append({
            "severity":       "Critical",
            "icon":           "🚨",
            "title":          f"High Failure Probability: {failure_probability:.1%}",
            "recommendation": "Halt production and perform full machine inspection.",
            "details": [
                "Model confidence in imminent failure exceeds 60%.",
                "Alert the maintenance team immediately.",
                "Run a full diagnostic cycle before resuming production.",
                "Document all anomalous sensor readings.",
                "Do not operate until cleared by a qualified maintenance engineer.",
            ],
        })
    elif failure_probability >= 0.30:
        recs.append({
            "severity":       "Warning",
            "icon":           "⚠️",
            "title":          f"Elevated Failure Risk: {failure_probability:.1%}",
            "recommendation": "Schedule inspection at the next available maintenance window.",
            "details": [
                "Model indicates non-trivial failure probability.",
                "Increase monitoring frequency for all sensor channels.",
                "Prepare maintenance resources and spare parts in advance.",
                "Review recent production quality metrics for early signs of degradation.",
            ],
        })

    # Sensor-threshold rule cards
    for rule in _RULES:
        if rule["condition"](input_data):
            recs.append({
                "severity":       rule["severity"],
                "icon":           rule["icon"],
                "title":          rule["title"],
                "recommendation": rule["recommendation"],
                "details":        rule["details"],
            })

    # Healthy card if nothing else triggered
    if not recs:
        recs.append(_HEALTHY_CARD)

    # Sort Critical → Warning → Info → Healthy
    recs.sort(key=lambda r: _SEVERITY_ORDER.get(r["severity"], 99))

    return recs
