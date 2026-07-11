# Guardini — Forecasting (FastAPI + React, container unico)
# Stage 1: build del frontend
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ .
RUN npm run build

# Stage 2: runtime Python
FROM python:3.11-slim

WORKDIR /app

# libgomp1: richiesta a runtime da numba/statsforecast per il multi-threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/ ./engine/
COPY backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist

# I run vengono persistiti fuori dal container (volume in docker-compose.yml)
ENV GUARDINI_RUNS_DIR=/data/runs
ENV GUARDINI_N_JOBS=2

EXPOSE 8501

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8501"]
