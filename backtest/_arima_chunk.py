# Script di servizio: AutoARIMA per un singolo cutoff, spezzato in chunk di serie
# (aggira il timeout della sandbox). Uso:
#   python _arima_chunk.py <cutoff YYYY-MM> <chunk_id> <n_chunks>   # fit chunk
#   python _arima_chunk.py <cutoff YYYY-MM> assemble <n_chunks>    # valuta e salva part
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load_pivot, classify  # noqa: E402
from metrics import mase_scale, evaluate  # noqa: E402
import models_sf  # noqa: E402

DD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dati", "dati_20260707")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
H = 3

cutoff = pd.Timestamp(sys.argv[1] + "-01")
mode = sys.argv[2]
n_chunks = int(sys.argv[3])

pivot, _ = load_pivot(
    os.path.join(DD, "VENDUTO_LUG 23_GIU 26.xlsx"),
    os.path.join(DD, "db_era_diventa_v2.xlsx"),
    os.path.join(DD, "db_clienti_promo.xlsx"),
    os.path.join(DD, "ARTICOLI DA ESCLUDERE.xlsx"),
)
train = pivot[[c for c in pivot.columns if c <= cutoff]]
train = train[(train != 0).any(axis=1)]
tmp_dir = os.path.join(OUT, "parts_tmp")
os.makedirs(tmp_dir, exist_ok=True)

if mode == "assemble":
    chunks = [pd.read_csv(os.path.join(tmp_dir, f"arima_{cutoff.date()}_{i}.csv"),
                          index_col=0, parse_dates=True).T
              for i in range(n_chunks)]
    fc = pd.concat(chunks)
    fc.columns = [pd.Timestamp(c) for c in fc.columns]
    test_cols = [cutoff + pd.DateOffset(months=i) for i in range(1, H + 1)]
    fc = fc.reindex(index=train.index, columns=test_cols).fillna(0)
    actual = pivot.loc[train.index, test_cols]
    classi = classify(train)
    scale = mase_scale(train)
    res = evaluate(fc, actual, classi, scale, cutoff, "AutoARIMA")
    part = os.path.join(OUT, "parts", f"AutoARIMA__{cutoff.date()}.csv")
    res.to_csv(part + ".tmp", index=False)
    os.replace(part + ".tmp", part)
    print("part salvato:", part)
else:
    chunk_id = int(mode)
    ids = [uid for k, uid in enumerate(train.index) if k % n_chunks == chunk_id]
    sub = train.loc[ids]
    out = models_sf.run_statsforecast(sub, H, which=["AutoARIMA"])["AutoARIMA"]
    out.T.to_csv(os.path.join(tmp_dir, f"arima_{cutoff.date()}_{chunk_id}.csv"))
    print(f"chunk {chunk_id}/{n_chunks} ok: {len(ids)} serie")
