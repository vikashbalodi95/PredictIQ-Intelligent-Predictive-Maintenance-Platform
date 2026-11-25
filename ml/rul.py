"""
ml/rul.py
---------
A SIMPLE Remaining-Useful-Life (RUL) estimator.

What is RUL?
============
RUL answers: "Roughly how much longer can this machine keep running before
the tool/machine is likely to fail?" We express it in MINUTES of remaining
tool life.

Important honesty note (great to say in an interview):
------------------------------------------------------
This is NOT a trained deep-learning RUL model. It is a transparent,
rule-based *heuristic*. We deliberately keep it simple and explainable:

  1. Tool wear has a known end-of-life limit (TOOL_WEAR_LIMIT minutes).
     Remaining wear budget = limit - current tool wear.

  2. We shrink that budget when the model thinks failure is likely
     (high failure probability) and when sensors are stressed
     (low health score). A stressed machine "uses up" its life faster.

The result is a single number a maintenance planner can act on, plus a
human-readable band ("OK", "Plan soon", "Urgent").
"""

from ml.config import TOOL_WEAR_LIMIT


def estimate_rul(
    tool_wear: float,
    failure_probability: float,
    health_score: float,
) -> dict:
    """
    Estimate remaining useful life in minutes.

    Parameters
    ----------
    tool_wear           : Current tool wear in minutes (sensor reading).
    failure_probability : Model probability of failure, 0..1.
    health_score        : Composite health score, 0..100.

    Returns
    -------
    dict with:
        rul_minutes : int   – estimated minutes of tool life left
        band        : str   – "OK" | "Plan soon" | "Urgent"
        explanation : str   – one-line plain-English reason
    """
    # 1) Raw remaining tool-wear budget (never negative).
    remaining_budget = max(0.0, TOOL_WEAR_LIMIT - float(tool_wear))

    # 2) A "stress factor" between ~0.15 and 1.0.
    #    - High failure probability => smaller factor => less life left.
    #    - Low health score        => smaller factor => less life left.
    #    We average the two signals and clamp so RUL never hits exactly zero
    #    just from the heuristic (a real failure event sets it to ~0 anyway).
    prob_factor   = 1.0 - float(failure_probability)        # 1 = safe, 0 = certain fail
    health_factor = max(0.0, float(health_score)) / 100.0   # 1 = perfect health
    stress_factor = max(0.15, (prob_factor + health_factor) / 2.0)

    rul_minutes = int(remaining_budget * stress_factor)

    # 3) Translate the number into an action band for the dashboard.
    if failure_probability >= 0.60 or rul_minutes <= 20:
        band = "Urgent"
        explanation = "Failure likely very soon — schedule maintenance now."
    elif failure_probability >= 0.30 or rul_minutes <= 60:
        band = "Plan soon"
        explanation = "Degradation detected — plan maintenance within this shift."
    else:
        band = "OK"
        explanation = "Healthy — continue normal operation and monitoring."

    return {
        "rul_minutes": rul_minutes,
        "band": band,
        "explanation": explanation,
    }
