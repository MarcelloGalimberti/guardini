# Guardini Forecasting — Webapp FastAPI + React

Architettura: `engine/` (motore statsforecast segmentato, condiviso con backtest e v9
Streamlit) → `backend/` (FastAPI: run persistiti, validazione upload, job forecast con
progress, export) → `frontend/` (React + Vite + TS + Tailwind: wizard a 4 step con
dashboard risultati, treemap cliccabile, confronto run).

## Sviluppo locale

```bash
# Terminale 1 — backend (dalla root del repo)
conda activate neuralprophet          # o qualsiasi env con i requirements
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# Terminale 2 — frontend con hot reload
cd frontend
npm install
npm run dev                           # http://localhost:5173 (proxy /api → 8000)
```

In alternativa senza npm: `frontend/dist/` è già buildato — basta il backend su
http://localhost:8000, che serve anche la UI.

## Variabili d'ambiente (backend)

| Variabile | Default | Note |
|---|---|---|
| `GUARDINI_RUNS_DIR` | `./runs_data` | dove vengono persistiti i run |
| `GUARDINI_N_JOBS` | `1` | parallelismo statsforecast (nel container: 2) |

## Deploy (Hetzner, pattern invariato)

Container unico: build multi-stage (node → python), uvicorn su 8501 interno,
mappato su `127.0.0.1:8504`, esposto dall'nginx di sistema sulla porta 8507
(`nginx.conf` invariato). I run persistono nel volume `guardini_runs`.

```bash
ssh root@65.21.182.192
cd /opt/guardini && git pull && docker compose up -d --build
```

## Test

Backend testato end-to-end con i dati reali `dati/dati_20260707`: upload+validazione
(incluso file errato → messaggio chiaro), prepare (1254 codici), classify, forecast
(~16 s con n_jobs=4), results, dettaglio serie, 3 export (galileo/dettaglio/new),
serving SPA con deep link. Il file J-Galileo è byte-compatibile col tracciato v8.
