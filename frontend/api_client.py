"""
frontend/api_client.py
----------------------
A thin wrapper around the backend's REST API.

Why have this layer?
  The Streamlit dashboard should NOT know about URLs, JSON shapes, or HTTP
  status codes. It just wants Python functions like `list_machines()` or
  `get_latest(pk)`. Putting all the `requests` calls here keeps the UI code
  clean and means there's one place to change if the API moves or changes.

This is the same idea as a "service layer" or "SDK" in larger apps.
"""

import requests

from core.settings import settings

BASE = settings.API_BASE_URL
TIMEOUT = 10  # seconds — fail fast if the backend is down


class APIError(Exception):
    """Raised when the backend can't be reached or returns an error."""


def _get(path: str, **params):
    try:
        r = requests.get(f"{BASE}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise APIError(str(exc)) from exc


def _post(path: str, json: dict):
    try:
        r = requests.post(f"{BASE}{path}", json=json, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise APIError(str(exc)) from exc


# ─── Convenience functions the dashboard calls ────────────────────────────────

def health() -> bool:
    """True if the backend is reachable."""
    try:
        return requests.get(f"{BASE}/health", timeout=3).ok
    except requests.RequestException:
        return False


def list_machines() -> list[dict]:
    return _get("/machines")


def get_latest(pk: int) -> dict:
    """Newest reading + prediction for the live tiles."""
    return _get(f"/machines/{pk}/latest")


def get_readings(pk: int, limit: int = 100) -> list[dict]:
    return _get(f"/machines/{pk}/readings", limit=limit)


def get_predictions(pk: int, limit: int = 100) -> list[dict]:
    return _get(f"/machines/{pk}/predictions", limit=limit)


def get_alerts(pk: int | None = None, limit: int = 50,
               unacknowledged_only: bool = False) -> list[dict]:
    if pk is None:
        return _get("/alerts", limit=limit, unacknowledged_only=unacknowledged_only)
    return _get(f"/machines/{pk}/alerts", limit=limit)


def acknowledge_alert(alert_id: int) -> dict:
    return _post(f"/alerts/{alert_id}/ack", json={})


def predict(payload: dict) -> dict:
    """Stateless what-if prediction."""
    return _post("/predict", json=payload)


def report_url(pk: int) -> str:
    """Direct link the dashboard offers for downloading the PDF report."""
    return f"{BASE}/machines/{pk}/report"


def get_report_bytes(pk: int) -> bytes:
    """Fetch the PDF report as raw bytes (for Streamlit's download_button)."""
    r = requests.get(report_url(pk), timeout=30)
    r.raise_for_status()
    return r.content
