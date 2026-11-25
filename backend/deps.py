"""
backend/deps.py
---------------
Shared "dependencies" the API routes pull in.

FastAPI's dependency-injection system lets a route declare what it needs
(a database session, the ML predictor) as function arguments, and FastAPI
supplies them. This keeps routes clean and makes testing easy (you can swap
a dependency for a fake one).

We load the heavy CNCPredictor ONCE and reuse it for every request, instead of
re-reading the model file on each call.
"""

from database.connection import get_db          # re-exported for convenience
from ml.predictor import CNCPredictor
from utils.logger import get_logger

log = get_logger("backend.deps")

# Module-level cache for the single predictor instance.
_predictor: CNCPredictor | None = None


def get_predictor() -> CNCPredictor:
    """
    Return the shared CNCPredictor, loading it on first use (lazy singleton).

    If the model hasn't been trained yet, this raises FileNotFoundError; the
    route layer turns that into a friendly HTTP error.
    """
    global _predictor
    if _predictor is None:
        log.info("Loading ML model into memory …")
        _predictor = CNCPredictor()
        log.info("Model loaded: %s", _predictor.best_model_name)
    return _predictor


# Re-export get_db so routes can simply `from backend.deps import get_db`.
__all__ = ["get_db", "get_predictor"]
