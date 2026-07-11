# =============================================================================
# pipeline.py — Replica standalone della pipeline dati di guardini_j_galileo_v8
# (senza Streamlit). Produce la matrice codice × mese e la classificazione
# Syntetos-Boylan con gli stessi default della v8.
# =============================================================================
import numpy as np
import pandas as pd

# Default v8 (valori proposti nella UI)
DEFAULTS = dict(
    CLASSE_VAL_1="PF interno+confez.",
    RECENT_MONTHS=6,
    MIN_NONZERO=12,
    ADI_LIMIT=1.40,
    CV_LIMIT=0.80,
    CV_MAX=120,
    DATA_MIN="2023-01-01",
)

PREFISSI_ESCLUSI = ("SK", "-", "*", ".", ".SCAR FERROSO", "_", "ST")
DESCR_ESCLUSE = ("SET", "EXPO")


def costruisci_mappa_transitiva(df_ed, max_iter=25):
    """Identica alla v8: risolve le catene era-diventa fino al codice finale."""
    mappa_diretta = dict(
        zip(df_ed["Era"].astype(str).str.strip(), df_ed["Diventa"].astype(str).str.strip())
    )
    mappa_finale = {}
    for codice_iniziale in mappa_diretta:
        corrente = codice_iniziale
        visitati = [corrente]
        for _ in range(max_iter):
            successivo = mappa_diretta.get(corrente)
            if successivo is None or successivo == corrente:
                break
            if successivo in visitati:
                break
            corrente = successivo
            visitati.append(corrente)
        mappa_finale[codice_iniziale] = corrente
    return mappa_finale


def load_pivot(path_venduto, path_era_diventa, path_promo, path_escludere=None,
               classe_val_1=DEFAULTS["CLASSE_VAL_1"], data_min=DEFAULTS["DATA_MIN"],
               usa_era_diventa=True):
    """Replica i passi dati della v8. Ritorna (pivot, anagrafica).

    pivot: DataFrame index=Codice Articolo, colonne=pd.Timestamp (primo del mese),
           valori=quantità mensile (>=0, fill 0).
    classe_val_1: stringa singola o lista di classi di valutazione da includere.
    usa_era_diventa: True = sostituzione transitiva era-diventa (default);
                     False = raggruppamento per primi 5 digit del codice.
    """
    df = pd.read_excel(path_venduto, engine="openpyxl")
    df = df[["CDCFST", "DSCFST", "CDARST", "DSARST", "QTFTST", "DTBOST", "DCA1ST", "DCA2ST", "DC02ST"]]
    for col in ["CDCFST", "DSCFST", "CDARST", "DSARST", "DCA1ST", "DCA2ST", "DC02ST"]:
        df[col] = df[col].astype(object).fillna("").astype(str)
    df["QTFTST"] = pd.to_numeric(df["QTFTST"], errors="coerce").fillna(0).astype(np.float64)
    s = df["DTBOST"].astype(str).str.zfill(8)
    df["DTBOST"] = pd.to_datetime(s.str[:4] + "-" + s.str[4:6] + "-" + s.str[6:8], errors="coerce")

    promo = pd.read_excel(path_promo, engine="openpyxl")
    promo["CDCFST"] = promo["CDCFST"].astype(object).fillna("").astype(str)
    df = df[~df["CDCFST"].isin(promo["CDCFST"])]

    if path_escludere is not None:
        esc = pd.read_excel(path_escludere, engine="openpyxl")
        esc["CDARMA"] = esc["CDARMA"].astype(object).fillna("").astype(str)
        df = df[~df["CDARST"].isin(set(esc["CDARMA"]))]

    classi_val = [classe_val_1] if isinstance(classe_val_1, str) else list(classe_val_1)
    df = df[df["DCA1ST"].isin(classi_val)]

    if usa_era_diventa:
        ed = pd.read_excel(path_era_diventa, engine="openpyxl")
        mappa = costruisci_mappa_transitiva(ed)
        df["CDARST"] = df["CDARST"].map(lambda c: mappa.get(c, c))
    else:
        df["CDARST"] = df["CDARST"].str[:5]

    # Obsoleti: elimina i codici per cui TUTTE le righe hanno descrizione '***'
    df["Obsoleto"] = df["DSARST"].str.startswith("***")
    df["Eliminare"] = df.groupby("CDARST")["Obsoleto"].transform("all")
    df = df[~df["Eliminare"]]

    df["Descrizione univoca"] = df["DSARST"].map(
        lambda d: d[3:] if d.startswith("***") or d.startswith("---") else d
    )

    df = df[~df["CDARST"].str.startswith(PREFISSI_ESCLUSI)]
    df = df[~df["DSARST"].str.contains("|".join(DESCR_ESCLUSE))]
    df = df[df["DTBOST"] >= pd.to_datetime(data_min)]

    df["Mese"] = df["DTBOST"].dt.to_period("M")
    g = df.groupby(["CDARST", "Mese"]).agg(
        qty=("QTFTST", "sum"), descr=("Descrizione univoca", "first")
    ).reset_index()

    pivot = g.pivot_table(index="CDARST", columns="Mese", values="qty",
                          aggfunc="sum", fill_value=0).clip(lower=0)
    pivot.columns = [c.to_timestamp() for c in pivot.columns]
    pivot = pivot[sorted(pivot.columns)]
    pivot.index.name = "Codice Articolo"

    anagrafica = g[["CDARST", "descr"]].drop_duplicates("CDARST").set_index("CDARST")["descr"]
    return pivot, anagrafica


# ---------------------------------------------------------------------------
# Classificazione Syntetos-Boylan (replica fedele della v8)
# ---------------------------------------------------------------------------

def _first_last_nonzero(row):
    nz = np.where(row.values != 0)[0]
    if len(nz) == 0:
        return np.nan, np.nan
    return nz[0], nz[-1]


def classify(pivot, recent_months=DEFAULTS["RECENT_MONTHS"], min_nonzero=DEFAULTS["MIN_NONZERO"],
             adi_limit=DEFAULTS["ADI_LIMIT"], cv_limit=DEFAULTS["CV_LIMIT"]):
    """Ritorna DataFrame per codice: ADI, CV_demand, Mean_Demand, classe SB."""
    values = pivot
    date_cols = list(values.columns)
    current_ref = date_cols[-1]

    mesi_con_valore = (values != 0).sum(axis=1)
    attivo_recent = (values.iloc[:, -recent_months:] != 0).any(axis=1)

    fl = values.apply(_first_last_nonzero, axis=1)
    first_idx = pd.Series([r[0] for r in fl], index=values.index)
    last_idx = pd.Series([r[1] for r in fl], index=values.index)
    span = (last_idx - first_idx + 1).where(~first_idx.isna(), np.nan)
    first_date = first_idx.map(lambda i: date_cols[int(i)] if pd.notna(i) else pd.NaT)

    def calc_adi(s, c):
        if c <= 1:
            return np.nan
        if s <= 1:
            return 0
        return s / c

    adi = pd.Series([calc_adi(s, c) for s, c in zip(span.fillna(0), mesi_con_valore.fillna(0))],
                    index=values.index)

    def calc_cv(row):
        nzv = row[row != 0]
        if len(nzv) < 2:
            return np.nan
        return (nzv.std() / nzv.mean()) * 100

    cv = values.apply(calc_cv, axis=1)

    def calc_mean(row):
        nzv = row[row != 0]
        return nzv.mean() if len(nzv) else 0.0

    mean_d = values.apply(calc_mean, axis=1)

    def classe(a, c, act, m, f_date):
        if m == 1:
            if pd.notna(f_date) and f_date > (current_ref - pd.DateOffset(months=6)):
                return "New"
            return "Lumpy"
        if m < 6 and act:
            if pd.notna(f_date) and f_date > (current_ref - pd.DateOffset(months=6)):
                return "New"
        if m < min_nonzero:
            return "Insufficient Data"
        if pd.isna(a) or pd.isna(c):
            return "Insufficient Data"
        cr = c / 100.0
        if a < adi_limit:
            return "Smooth" if cr < cv_limit else "Erratic"
        return "Intermittent" if cr < cv_limit else "Lumpy"

    classi = [classe(a, c, act, m, fd)
              for a, c, act, m, fd in zip(adi, cv, attivo_recent, mesi_con_valore, first_date)]

    return pd.DataFrame({
        "ADI": adi, "CV_demand": cv, "Mean_Demand": mean_d,
        "Mesi_nonzero": mesi_con_valore, "Attivo_recente": attivo_recent,
        "Classe": classi,
    }, index=values.index)
