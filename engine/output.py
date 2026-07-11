# =============================================================================
# output.py — Costruzione del file per J-Galileo (struttura IDENTICA alla v8):
# colonne: Cliente/Fornitore | Articolo | Commessa | Sotto commessa |
#          Proprietà | Magazzino | <mesi forecast>
# Come in v8: si scarta il primo mese di forecast (mese corrente, già avviato)
# e si eliminano le righe con forecast tutto a zero.
# =============================================================================
import pandas as pd


def build_galileo(fc, drop_first_month=True):
    """fc: DataFrame index=Codice Articolo, colonne=mesi (Timestamp), valori int."""
    df = fc.copy()
    date_cols = list(df.columns)
    if drop_first_month and len(date_cols) > 1:
        df = df.drop(columns=date_cols[0])
        date_cols = date_cols[1:]

    df = df[(df[date_cols] != 0).any(axis=1)]

    out = df.reset_index().rename(columns={df.index.name or "index": "Articolo"})
    out.insert(0, "Cliente/Fornitore", "")
    out.insert(2, "Commessa", "")
    out.insert(3, "Sotto commessa", "")
    out.insert(4, "Proprietà", "0")
    out.insert(5, "Magazzino", "100")
    return out
