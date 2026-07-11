# =============================================================================
# main.py — API FastAPI per il forecasting Guardini.
# Sviluppo:  uvicorn backend.main:app --reload --port 8000   (dalla root repo)
# Produzione: stesso processo serve anche il frontend buildato (frontend/dist).
# =============================================================================
import os
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import build_galileo, CLASSI_WARNING
from . import services, store
from .validation import SCHEMI, valida_file

app = FastAPI(title="Guardini Forecasting API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Modelli richiesta
# ---------------------------------------------------------------------------

class PreparaReq(BaseModel):
    classi_valutazione: list[str] = Field(default=["PF interno+confez."])
    usa_era_diventa: bool = True


class ClassificaReq(BaseModel):
    recent_months: int = 6
    min_nonzero: int = 12
    adi_limit: float = 1.40
    cv_limit: float = 0.80


class ForecastReq(BaseModel):
    orizzonte: int = Field(default=4, ge=2, le=12)
    cap_abilitato: bool = True
    cap_factor: float = Field(default=1.5, ge=1.0, le=5.0)


def _check_run(run_id: str):
    if not store.exists(run_id):
        raise HTTPException(404, "Run non trovato")


# ---------------------------------------------------------------------------
# Run e upload
# ---------------------------------------------------------------------------

@app.post("/api/runs")
def crea_run():
    return {"run_id": store.new_run()}


@app.get("/api/runs")
def elenco_run():
    return [{k: m.get(k) for k in ("run_id", "creato", "stato", "kpi", "sintesi_dati")}
            for m in store.list_runs()]


@app.get("/api/runs/{run_id}")
def stato_run(run_id: str):
    _check_run(run_id)
    return store.load_meta(run_id)


@app.post("/api/runs/{run_id}/files/{tipo}")
async def carica_file(run_id: str, tipo: str, file: UploadFile):
    _check_run(run_id)
    if tipo not in SCHEMI:
        raise HTTPException(400, f"Tipo file sconosciuto: {tipo}")
    contenuto = await file.read()
    esito = valida_file(tipo, contenuto, file.filename or tipo)
    if esito["ok"]:
        store.save_input(run_id, tipo, contenuto)
    meta = store.load_meta(run_id)
    meta["validazioni"][tipo] = esito
    store.save_meta(run_id, meta)
    return esito


# ---------------------------------------------------------------------------
# Step 1→2: preparazione dati
# ---------------------------------------------------------------------------

@app.post("/api/runs/{run_id}/prepare")
def prepara(run_id: str, req: PreparaReq):
    _check_run(run_id)
    meta = store.load_meta(run_id)
    obbligatori = ["venduto", "era_diventa", "promo"]
    mancanti = [t for t in obbligatori
                if not meta["validazioni"].get(t, {}).get("ok")]
    if mancanti:
        labels = ", ".join(SCHEMI[t]["label"] for t in mancanti)
        raise HTTPException(400, f"File mancanti o non validi: {labels}")
    try:
        return services.prepara_dati(run_id, req.classi_valutazione, req.usa_era_diventa)
    except Exception as e:
        raise HTTPException(500, f"Errore nella preparazione dati: {e}")


# ---------------------------------------------------------------------------
# Step 2: classificazione
# ---------------------------------------------------------------------------

@app.post("/api/runs/{run_id}/classify")
def classifica(run_id: str, req: ClassificaReq):
    _check_run(run_id)
    if not store.has_df(run_id, "pivot"):
        raise HTTPException(400, "Prima completare la preparazione dati (step 1)")
    return services.classifica(run_id, req.model_dump())


# ---------------------------------------------------------------------------
# Step 3: forecast (job in background + polling stato)
# ---------------------------------------------------------------------------

@app.post("/api/runs/{run_id}/forecast")
def forecast(run_id: str, req: ForecastReq):
    _check_run(run_id)
    if not store.has_df(run_id, "classi"):
        raise HTTPException(400, "Prima completare la classificazione (step 2)")
    meta = store.load_meta(run_id)
    if meta["stato"] == "forecast_in_corso":
        raise HTTPException(409, "Forecast già in corso per questo run")
    services.avvia_forecast(run_id, req.orizzonte,
                            req.cap_factor if req.cap_abilitato else None)
    return {"stato": "forecast_in_corso"}


# ---------------------------------------------------------------------------
# Step 4: risultati, dettaglio serie, confronto
# ---------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/results")
def risultati(run_id: str):
    _check_run(run_id)
    if not store.has_df(run_id, "forecast"):
        raise HTTPException(400, "Forecast non ancora disponibile")
    return services.risultati(run_id)


@app.get("/api/runs/{run_id}/series/{codice}")
def serie(run_id: str, codice: str):
    _check_run(run_id)
    out = services.serie_dettaglio(run_id, codice)
    if out is None:
        raise HTTPException(404, f"Codice {codice} non trovato")
    return out


@app.get("/api/runs/{run_id}/confronto/{altro_id}")
def confronto(run_id: str, altro_id: str):
    _check_run(run_id)
    _check_run(altro_id)
    if not (store.has_df(run_id, "forecast") and store.has_df(altro_id, "forecast")):
        raise HTTPException(400, "Entrambi i run devono avere un forecast completato")
    return services.confronto(run_id, altro_id)


# ---------------------------------------------------------------------------
# Export Excel
# ---------------------------------------------------------------------------

def _xlsx_response(df: pd.DataFrame, filename: str, index=False):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name="Sheet1", index=index)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/runs/{run_id}/export/galileo")
def export_galileo(run_id: str):
    _check_run(run_id)
    fc = store.load_df(run_id, "forecast")
    return _xlsx_response(build_galileo(fc), "df_galileo_per_forecasting.xlsx")


@app.get("/api/runs/{run_id}/export/dettaglio")
def export_dettaglio(run_id: str):
    _check_run(run_id)
    fc = store.load_df(run_id, "forecast")
    classi = store.load_df(run_id, "classi")
    metodo = store.load_df(run_id, "metodo")["metodo"]
    anag = store.load_df(run_id, "anagrafica")["descrizione"]
    out = fc.copy()
    out.columns = [f"{c:%Y-%m}" for c in out.columns]
    out.insert(0, "Descrizione", [anag.get(c, "") for c in fc.index])
    out.insert(1, "Classe", classi["Classe"])
    out.insert(2, "Metodo", metodo)
    return _xlsx_response(out.reset_index(), "forecast_dettaglio.xlsx")


@app.get("/api/runs/{run_id}/export/new")
def export_new(run_id: str):
    _check_run(run_id)
    classi = store.load_df(run_id, "classi")
    anag = store.load_df(run_id, "anagrafica")["descrizione"]
    ids = classi.index[classi["Classe"].isin(CLASSI_WARNING)]
    df = pd.DataFrame({
        "Codice Articolo": ids,
        "Descrizione": [anag.get(c, "") for c in ids],
        "Mesi di storico": classi.loc[ids, "Mesi_nonzero"].values,
    })
    return _xlsx_response(df, "codici_new_da_verificare.xlsx")


# ---------------------------------------------------------------------------
# Frontend statico (produzione: frontend/dist copiato accanto al backend)
# ---------------------------------------------------------------------------
_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidato = os.path.join(_dist, full_path)
        if full_path and os.path.isfile(candidato):
            return FileResponse(candidato)
        return FileResponse(os.path.join(_dist, "index.html"))
