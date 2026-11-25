"""
backend/main.py
---------------
The FastAPI application entry point — this is the backend "server".

Run it with:
    uvicorn backend.main:app --reload

Then open http://localhost:8000/docs for the auto-generated, interactive API
documentation (one of FastAPI's best features — every endpoint is testable
from the browser).

What this file does:
  * creates the database tables on startup (lifespan),
  * enables CORS so the Streamlit frontend can call the API,
  * registers every router,
  * exposes a couple of top-level health/info endpoints.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import alerts, machines, predictions, readings, reports
from core.settings import settings
from database.connection import init_db
from utils.logger import get_logger

log = get_logger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code that runs ONCE on startup (before 'yield') and ONCE on shutdown (after).
    We use it to make sure the database tables exist before serving requests.
    """
    log.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    init_db()
    log.info("Database ready at %s", settings.DATABASE_URL.split("@")[-1])
    yield
    log.info("Shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="REST API for streaming CNC sensor data, ML failure prediction, "
                "RUL estimation, alerting, and PDF maintenance reports.",
    lifespan=lifespan,
)

# CORS = Cross-Origin Resource Sharing. The browser blocks a page served from
# one origin (the Streamlit app) from calling another origin (this API) unless
# the API explicitly allows it. For a local/demo project we allow everything.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Plug each router into the app. Order doesn't matter functionally.
app.include_router(machines.router)
app.include_router(readings.router)
app.include_router(predictions.router)
app.include_router(alerts.router)
app.include_router(reports.router)


@app.get("/", tags=["health"])
def root():
    """A friendly landing payload so you can confirm the API is alive."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "ok",
    }


@app.get("/health", tags=["health"])
def health():
    """Lightweight health check for Docker / uptime monitors."""
    return {"status": "healthy"}
