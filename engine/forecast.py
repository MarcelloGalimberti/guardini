# =============================================================================
# forecast.py — Motore segmentato v9 (champion per classe dal backtest Fase 2).
#
# | Classe SB          | Modello     | WAPE cum. backtest (v8 sulla stessa classe) |
# |--------------------|-------------|---------------------------------------------|
# | Smooth             | AutoETS     | 28,2%  (38,2%)                              |
# | Erratic            | ADIDA       | 49,9%  (69,7%)                              |
# | Intermittent       | CrostonSBA  | 41,1%  (42,3%)                              |
# | Insufficient Data  | AutoARIMA   | 67,3%  (112,9%)                             |
# | Lumpy              | SBA piatto  | 86,1%  (86,1%)                              |
# | New                | SBA piatto  | 90,2%  — INAFFIDABILE, richiede warning     |
# =============================================================================
import numpy as np
import pandas as pd

SEASON = 12

CHAMPIONS = {
    "Smooth": "AutoETS",
    "Erratic": "ADIDA",
    "Intermittent": "CrostonSBA",
    "Insufficient Data": "AutoARIMA",
    "Lumpy": "SBA_piatto",
    "New": "SBA_piatto",
}

CLASSI_WARNING = ["New"]  # forecast inaffidabile: da segnalare all'utente


def _to_long_trimmed(train_pivot):
    """Long format (unique_id, ds, y), senza gli zeri precedenti la prima vendita."""
    rows = []
    cols = np.array(train_pivot.columns)
    for uid, r in zip(train_pivot.index, train_pivot.values):
        nz = np.nonzero(r)[0]
        if len(nz) == 0:
            continue
        rows.append(pd.DataFrame({"unique_id": uid, "ds": cols[nz[0]:][: len(r) - nz[0]],
                                  "y": r[nz[0]:]}))
    return pd.concat(rows, ignore_index=True)


def _sba_flat(classi, ids):
    """Rate SBA come in v8: Mean_Demand / ADI × (1 - alpha/2), alpha=0.1."""
    adi = classi.loc[ids, "ADI"].replace(0, 1)
    return ((classi.loc[ids, "Mean_Demand"] / adi) * 0.95).fillna(0)


def forecast_segmentato(pivot, classi, horizon, cap_factor=1.5, progress_cb=None,
                        n_jobs=1):
    """Forecast per tutte le serie del pivot, instradate per classe SB.

    Ritorna (fc, metodo):
      fc      DataFrame index=Codice Articolo, colonne=mesi futuri (Timestamp), valori int
      metodo  Series index=Codice Articolo, nome del modello usato
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoARIMA, CrostonSBA, ADIDA, SeasonalNaive

    future_cols = [pivot.columns[-1] + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    fc = pd.DataFrame(0.0, index=pivot.index, columns=future_cols)
    metodo = pd.Series("", index=pivot.index, name="Metodo")

    modelli_sf = {
        "AutoETS": lambda: AutoETS(season_length=SEASON),
        "ADIDA": lambda: ADIDA(),
        "CrostonSBA": lambda: CrostonSBA(),
        "AutoARIMA": lambda: AutoARIMA(season_length=SEASON),
    }

    for classe, nome_modello in CHAMPIONS.items():
        ids = classi.index[classi["Classe"] == classe]
        ids = [i for i in ids if i in pivot.index]
        if not ids:
            continue
        if progress_cb:
            progress_cb(f"{classe}: {len(ids)} codici → {nome_modello}")

        if nome_modello == "SBA_piatto":
            flat = _sba_flat(classi, ids)
            for col in future_cols:
                fc.loc[ids, col] = flat
        else:
            df_long = _to_long_trimmed(pivot.loc[ids])
            # n_jobs=1 di default: il multiprocessing di statsforecast crasha
            # dentro Streamlit (BrokenProcessPool). Il backend FastAPI può
            # passare n_jobs>1 (variabile d'ambiente GUARDINI_N_JOBS).
            sf = StatsForecast(models=[modelli_sf[nome_modello]()], freq="MS", n_jobs=n_jobs,
                               fallback_model=SeasonalNaive(season_length=SEASON))
            pred = sf.forecast(df=df_long, h=horizon)
            pred = pred.reset_index() if "unique_id" not in pred.columns else pred
            col_pred = [c for c in pred.columns if c not in ("unique_id", "ds")][0]
            wide = pred.pivot(index="unique_id", columns="ds", values=col_pred)
            wide.columns = [pd.Timestamp(c) for c in wide.columns]
            wide = wide.reindex(index=ids, columns=future_cols).fillna(0)
            fc.loc[ids, :] = wide.values

        metodo.loc[ids] = nome_modello

    # Non-negatività + guardrail anti-picco (cap = fattore × max storico mensile)
    fc = fc.clip(lower=0)
    if cap_factor is not None:
        caps = pivot.max(axis=1) * cap_factor
        fc = fc.clip(upper=caps, axis=0)

    return fc.round(0).astype(int), metodo
