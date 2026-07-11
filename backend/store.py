# =============================================================================
# store.py — Persistenza dei run su filesystem (nessun DB).
# runs_data/<run_id>/
#   metadata.json     stato, parametri, timestamp, validazioni, progress
#   input_<tipo>.xlsx file caricati (per riaprire/riprodurre il run)
#   pivot.parquet     matrice codice × mese
#   classi.parquet    classificazione SB
#   forecast.parquet  forecast per codice × mese
#   anagrafica.parquet
# =============================================================================
import json
import os
import threading
import uuid
from datetime import datetime

import pandas as pd

BASE_DIR = os.environ.get(
    "GUARDINI_RUNS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs_data"),
)

_lock = threading.Lock()


def _run_dir(run_id: str) -> str:
    return os.path.join(BASE_DIR, run_id)


def new_run() -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    os.makedirs(_run_dir(run_id), exist_ok=True)
    save_meta(run_id, {
        "run_id": run_id,
        "creato": datetime.now().isoformat(timespec="seconds"),
        "stato": "creato",          # creato -> dati_ok -> classificato -> forecast_in_corso -> completato | errore
        "progress": None,
        "validazioni": {},
        "parametri": {},
        "kpi": {},
    })
    return run_id


def save_meta(run_id: str, meta: dict) -> None:
    with _lock:
        path = os.path.join(_run_dir(run_id), "metadata.json")
        with open(path + ".tmp", "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
        os.replace(path + ".tmp", path)


def load_meta(run_id: str) -> dict:
    with open(os.path.join(_run_dir(run_id), "metadata.json")) as f:
        return json.load(f)


def update_meta(run_id: str, **kwargs) -> dict:
    meta = load_meta(run_id)
    meta.update(kwargs)
    save_meta(run_id, meta)
    return meta


def save_input(run_id: str, tipo: str, contenuto: bytes) -> None:
    with open(os.path.join(_run_dir(run_id), f"input_{tipo}.xlsx"), "wb") as f:
        f.write(contenuto)


def input_path(run_id: str, tipo: str) -> str:
    return os.path.join(_run_dir(run_id), f"input_{tipo}.xlsx")


def has_input(run_id: str, tipo: str) -> bool:
    return os.path.exists(input_path(run_id, tipo))


def save_df(run_id: str, nome: str, df: pd.DataFrame) -> None:
    df.to_parquet(os.path.join(_run_dir(run_id), f"{nome}.parquet"))


def load_df(run_id: str, nome: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_run_dir(run_id), f"{nome}.parquet"))


def has_df(run_id: str, nome: str) -> bool:
    return os.path.exists(os.path.join(_run_dir(run_id), f"{nome}.parquet"))


def list_runs() -> list:
    if not os.path.isdir(BASE_DIR):
        return []
    out = []
    for d in sorted(os.listdir(BASE_DIR), reverse=True):
        meta_path = os.path.join(BASE_DIR, d, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    out.append(json.load(f))
            except Exception:
                continue
    return out


def exists(run_id: str) -> bool:
    return os.path.isdir(_run_dir(run_id))
