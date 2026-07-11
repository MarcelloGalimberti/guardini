# =============================================================================
# validation.py — Validazione dei file Excel caricati, con messaggi in italiano.
# Ogni file viene letto e controllato PRIMA di avviare l'elaborazione, così
# l'utente riceve un errore chiaro invece di un traceback.
# =============================================================================
from io import BytesIO

import pandas as pd

SCHEMI = {
    "venduto": {
        "label": "Venduto J-Galileo",
        "colonne": ["CDCFST", "DSCFST", "CDARST", "DSARST", "QTFTST", "DTBOST", "DCA1ST"],
        "descrizione": "estrazione venduto (righe di fatturato per cliente/articolo/data)",
    },
    "era_diventa": {
        "label": "Era-Diventa",
        "colonne": ["Era", "Diventa"],
        "descrizione": "mappatura sostituzione codici articolo",
    },
    "promo": {
        "label": "Clienti promozioni",
        "colonne": ["CDCFST", "DSCFST"],
        "descrizione": "elenco clienti promo da escludere",
    },
    "esclusioni": {
        "label": "Articoli da escludere",
        "colonne": ["CDARMA"],
        "descrizione": "elenco codici articolo da escludere (opzionale)",
    },
}


def valida_file(tipo: str, contenuto: bytes, nome_file: str) -> dict:
    """Ritorna {ok, messaggi, righe, dettagli} senza sollevare eccezioni."""
    schema = SCHEMI[tipo]
    esito = {"tipo": tipo, "nome_file": nome_file, "ok": False, "righe": 0,
             "messaggi": [], "dettagli": {}}

    try:
        df = pd.read_excel(BytesIO(contenuto), engine="openpyxl")
    except Exception:
        esito["messaggi"].append(
            f"Il file '{nome_file}' non è un Excel leggibile. Esportare da J-Galileo "
            f"in formato .xlsx e ricaricare.")
        return esito

    mancanti = [c for c in schema["colonne"] if c not in df.columns]
    if mancanti:
        esito["messaggi"].append(
            f"Il file '{nome_file}' non sembra essere «{schema['label']}» "
            f"({schema['descrizione']}): mancano le colonne {', '.join(mancanti)}. "
            f"Colonne trovate: {', '.join(str(c) for c in df.columns[:8])}…")
        return esito

    if len(df) == 0:
        esito["messaggi"].append(f"Il file '{nome_file}' è vuoto.")
        return esito

    esito["ok"] = True
    esito["righe"] = int(len(df))

    if tipo == "venduto":
        s = df["DTBOST"].astype(str).str.zfill(8)
        date = pd.to_datetime(s.str[:4] + "-" + s.str[4:6] + "-" + s.str[6:8], errors="coerce")
        n_date_ko = int(date.isna().sum())
        esito["dettagli"] = {
            "codici_univoci": int(df["CDARST"].nunique()),
            "periodo": f"{date.min():%d/%m/%Y} → {date.max():%d/%m/%Y}" if date.notna().any() else "n/d",
            "classi_valutazione": sorted(df["DCA1ST"].dropna().astype(str).unique().tolist()),
        }
        if n_date_ko > 0:
            esito["messaggi"].append(
                f"Attenzione: {n_date_ko} righe hanno una data non interpretabile e "
                f"verranno ignorate.")
    return esito
