# =============================================================================
# services.py — Logica applicativa: preparazione dati, classificazione,
# job di forecast con progress, risultati e confronto run.
# =============================================================================
import os
import threading
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from engine import classify, forecast_segmentato, CHAMPIONS, CLASSI_WARNING
from engine.pipeline import load_pivot
from . import store

# Linguaggio a due livelli: etichette business per il cliente
LABEL_BUSINESS = {
    "Smooth": "Regolare",
    "Erratic": "Irregolare",
    "Intermittent": "Sporadica",
    "Lumpy": "Sporadica",
    "Insufficient Data": "Storico limitato",
    "New": "Nuovo articolo",
}

# Accuratezza attesa dal backtest (REPORT_FASE2), mostrata in UI
WAPE_BACKTEST = {
    "Smooth": 28, "Erratic": 50, "Intermittent": 41,
    "Lumpy": 86, "Insufficient Data": 67, "New": 90,
}


def prepara_dati(run_id: str, classi_valutazione: list, usa_era_diventa: bool) -> dict:
    """Costruisce il pivot dai file caricati e lo salva nel run."""
    pivot, anagrafica = load_pivot(
        store.input_path(run_id, "venduto"),
        store.input_path(run_id, "era_diventa"),
        store.input_path(run_id, "promo"),
        store.input_path(run_id, "esclusioni") if store.has_input(run_id, "esclusioni") else None,
        classe_val_1=classi_valutazione,
        usa_era_diventa=usa_era_diventa,
    )
    store.save_df(run_id, "pivot", pivot)
    store.save_df(run_id, "anagrafica", anagrafica.to_frame("descrizione"))
    sintesi = {
        "n_codici": int(pivot.shape[0]),
        "n_mesi": int(pivot.shape[1]),
        "periodo": f"{pivot.columns[0]:%m/%Y} → {pivot.columns[-1]:%m/%Y}",
        "volume_totale": int(pivot.values.sum()),
    }
    store.update_meta(run_id, stato="dati_ok", sintesi_dati=sintesi,
                      parametri={"classi_valutazione": classi_valutazione,
                                 "usa_era_diventa": usa_era_diventa})
    return sintesi


def classifica(run_id: str, params: dict) -> dict:
    pivot = store.load_df(run_id, "pivot")
    classi = classify(pivot,
                      recent_months=params.get("recent_months", 6),
                      min_nonzero=params.get("min_nonzero", 12),
                      adi_limit=params.get("adi_limit", 1.40),
                      cv_limit=params.get("cv_limit", 0.80))
    store.save_df(run_id, "classi", classi)

    tot = pivot.sum(axis=1)
    riepilogo = []
    for classe in classi["Classe"].unique():
        ids = classi.index[classi["Classe"] == classe]
        riepilogo.append({
            "classe": classe,
            "label": LABEL_BUSINESS.get(classe, classe),
            "modello": CHAMPIONS.get(classe, "-"),
            "n_codici": int(len(ids)),
            "volume": int(tot[ids].sum()),
            "pct_volume": round(float(tot[ids].sum() / max(tot.sum(), 1) * 100), 1),
            "wape_backtest": WAPE_BACKTEST.get(classe),
            "warning": classe in CLASSI_WARNING,
        })
    riepilogo.sort(key=lambda r: -r["volume"])

    # punti per la matrice ADI/CV (solo classificabili)
    m = classi.reset_index().rename(columns={"index": "codice", "Codice Articolo": "codice"})
    m = m[m["ADI"].notna() & m["CV_demand"].notna()]
    punti = [{"codice": r["codice"], "adi": round(float(r["ADI"]), 2),
              "cv": round(float(r["CV_demand"]), 1), "classe": r["Classe"],
              "label": LABEL_BUSINESS.get(r["Classe"], r["Classe"])}
             for r in m.to_dict("records")]

    meta = store.load_meta(run_id)
    meta["parametri"].update(params)
    store.save_meta(run_id, meta)
    store.update_meta(run_id, stato="classificato")
    return {"riepilogo": riepilogo, "matrice": punti,
            "soglie": {"adi": params.get("adi_limit", 1.40),
                       "cv": params.get("cv_limit", 0.80) * 100}}


def _forecast_job(run_id: str, orizzonte: int, cap_factor):
    try:
        pivot = store.load_df(run_id, "pivot")
        classi = store.load_df(run_id, "classi")

        def progress(msg):
            store.update_meta(run_id, progress=msg)

        n_jobs = int(os.environ.get("GUARDINI_N_JOBS", "1"))
        fc, metodo = forecast_segmentato(pivot, classi, orizzonte,
                                         cap_factor=cap_factor, progress_cb=progress,
                                         n_jobs=n_jobs)
        store.save_df(run_id, "forecast", fc)
        store.save_df(run_id, "metodo", metodo.to_frame("metodo"))

        mesi_out = list(fc.columns[1:])  # il primo mese viene scartato nell'output J-Galileo
        n_new = int(classi["Classe"].isin(CLASSI_WARNING).sum())
        kpi = {
            "volume_mese": int(fc[mesi_out].sum().mean()) if mesi_out else 0,
            "n_codici_forecast": int((fc[mesi_out].sum(axis=1) > 0).sum()) if mesi_out else 0,
            "pct_volume_regolare": round(float(
                pivot.loc[classi["Classe"] == "Smooth"].sum().sum() /
                max(pivot.values.sum(), 1) * 100), 1),
            "n_new_da_verificare": n_new,
            "mesi_output": [f"{c:%Y-%m}" for c in mesi_out],
        }
        store.update_meta(run_id, stato="completato", progress=None, kpi=kpi,
                          completato=datetime.now().isoformat(timespec="seconds"),
                          parametri={**store.load_meta(run_id)["parametri"],
                                     "orizzonte": orizzonte, "cap_factor": cap_factor})
    except Exception:
        store.update_meta(run_id, stato="errore", progress=None,
                          errore=traceback.format_exc(limit=3))


def avvia_forecast(run_id: str, orizzonte: int, cap_factor) -> None:
    store.update_meta(run_id, stato="forecast_in_corso", progress="Avvio…")
    t = threading.Thread(target=_forecast_job, args=(run_id, orizzonte, cap_factor), daemon=True)
    t.start()


def risultati(run_id: str) -> dict:
    pivot = store.load_df(run_id, "pivot")
    classi = store.load_df(run_id, "classi")
    fc = store.load_df(run_id, "forecast")
    metodo = store.load_df(run_id, "metodo")["metodo"]
    anag = store.load_df(run_id, "anagrafica")["descrizione"]
    meta = store.load_meta(run_id)

    mesi_out = list(fc.columns[1:])
    storico_tot = pivot.sum(axis=1)

    righe = []
    for cod in fc.index:
        cl = classi.loc[cod, "Classe"]
        f_mesi = {f"{c:%Y-%m}": int(fc.loc[cod, c]) for c in mesi_out}
        righe.append({
            "codice": cod,
            "descrizione": str(anag.get(cod, "")),
            "classe": cl,
            "label": LABEL_BUSINESS.get(cl, cl),
            "metodo": str(metodo.get(cod, "")),
            "warning": cl in CLASSI_WARNING,
            "storico_totale": int(storico_tot[cod]),
            **f_mesi,
            "totale": int(sum(f_mesi.values())),
        })

    treemap = sorted(
        [r for r in righe if r["totale"] > 0],
        key=lambda r: -r["totale"])[:60]
    treemap = [{"codice": r["codice"], "descrizione": r["descrizione"],
                "totale": r["totale"], "label": r["label"], "warning": r["warning"]}
               for r in treemap]

    return {"kpi": meta.get("kpi", {}), "mesi": [f"{c:%Y-%m}" for c in mesi_out],
            "righe": righe, "treemap": treemap}


def serie_dettaglio(run_id: str, codice: str) -> dict:
    pivot = store.load_df(run_id, "pivot")
    fc = store.load_df(run_id, "forecast")
    classi = store.load_df(run_id, "classi")
    metodo = store.load_df(run_id, "metodo")["metodo"]
    anag = store.load_df(run_id, "anagrafica")["descrizione"]
    if codice not in pivot.index:
        return None
    cl = classi.loc[codice, "Classe"]
    return {
        "codice": codice,
        "descrizione": str(anag.get(codice, "")),
        "classe": cl,
        "label": LABEL_BUSINESS.get(cl, cl),
        "metodo": str(metodo.get(codice, "")),
        "warning": cl in CLASSI_WARNING,
        "adi": None if pd.isna(classi.loc[codice, "ADI"]) else round(float(classi.loc[codice, "ADI"]), 2),
        "cv": None if pd.isna(classi.loc[codice, "CV_demand"]) else round(float(classi.loc[codice, "CV_demand"]), 1),
        "storico": [{"mese": f"{c:%Y-%m}", "qty": int(pivot.loc[codice, c])} for c in pivot.columns],
        "forecast": [{"mese": f"{c:%Y-%m}", "qty": int(fc.loc[codice, c])} for c in fc.columns],
    }


def confronto(run_id: str, altro_id: str) -> dict:
    """Confronto sintetico tra due run completati."""
    out = {"run": run_id, "confronto_con": altro_id, "classi": [], "codici": {}}
    fc_a = store.load_df(run_id, "forecast")
    fc_b = store.load_df(altro_id, "forecast")
    cl_a = store.load_df(run_id, "classi")["Classe"]
    mesi_a, mesi_b = fc_a.columns[1:], fc_b.columns[1:]
    tot_a, tot_b = fc_a[mesi_a].sum(axis=1), fc_b[mesi_b].sum(axis=1)

    comuni = tot_a.index.intersection(tot_b.index)
    solo_a = tot_a.index.difference(tot_b.index)
    solo_b = tot_b.index.difference(tot_a.index)
    out["codici"] = {"comuni": int(len(comuni)), "nuovi": int(len(solo_a)),
                     "usciti": int(len(solo_b))}
    out["volume"] = {"attuale": int(tot_a.sum()), "precedente": int(tot_b.sum()),
                     "delta_pct": round(float((tot_a.sum() - tot_b.sum()) /
                                              max(tot_b.sum(), 1) * 100), 1)}
    for classe in cl_a.unique():
        ids = [i for i in comuni if cl_a.get(i) == classe]
        if not ids:
            continue
        a, b = tot_a[ids].sum(), tot_b[ids].sum()
        out["classi"].append({
            "classe": classe, "label": LABEL_BUSINESS.get(classe, classe),
            "attuale": int(a), "precedente": int(b),
            "delta_pct": round(float((a - b) / max(b, 1) * 100), 1)})
    delta = (tot_a[comuni] - tot_b[comuni]).abs().sort_values(ascending=False)
    out["top_variazioni"] = [
        {"codice": c, "attuale": int(tot_a[c]), "precedente": int(tot_b[c]),
         "delta": int(tot_a[c] - tot_b[c])}
        for c in delta.head(10).index]
    return out
