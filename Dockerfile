# ---------------------------------------------------------------------------
# Dockerfile — one image, three roles
# ---------------------------------------------------------------------------
# The same image runs the backend, the frontend, or the simulator. Which one
# it actually runs is decided by the `command:` in docker-compose.yml, so we
# only have to build this once.
# ---------------------------------------------------------------------------

# Start from a small, official Python image.
FROM python:3.12-slim

# --- System packages -------------------------------------------------------
# build-essential + libpq-dev let psycopg2 / numpy compile cleanly if needed.
# curl is handy for container health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Don't write .pyc files, and flush stdout immediately (nicer container logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Everything lives under /app inside the container.
WORKDIR /app

# --- Python dependencies ---------------------------------------------------
# Copy requirements first so Docker can cache the (slow) pip install layer
# and skip it on rebuilds when only source code changed.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Application code ------------------------------------------------------
COPY . .

# Make the project importable as packages (core, ml, backend, ...).
ENV PYTHONPATH=/app

# Backend (8000) and Streamlit (8501) — documentation only; compose maps ports.
EXPOSE 8000 8501

# Default command runs the API. docker-compose overrides this per service.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
