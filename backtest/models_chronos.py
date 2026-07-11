# =============================================================================
# models_chronos.py — Modello fondazionale Chronos-Bolt (Amazon), zero-shot.
# Gira in locale (richiede torch): pip install chronos-forecasting
# Il modello resta locale: nessun dato esce dalla macchina (a differenza di
# TimeGPT, che è un'API esterna).
# =============================================================================
import numpy as np
import pandas as pd

_MODEL = "amazon/chronos-bolt-small"
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from chronos import BaseChronosPipeline
        _pipeline = BaseChronosPipeline.from_pretrained(
            _MODEL, device_map="cpu", torch_dtype=torch.float32
        )
    return _pipeline


def run_chronos(train_pivot, horizon, batch_size=256):
    """Zero-shot forecast per tutte le serie. Leading zeros rimossi per serie.
    Usa la mediana predetta; clip a 0."""
    import torch
    pipe = _get_pipeline()

    future_cols = [train_pivot.columns[-1] + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    ids, contexts = [], []
    vals = train_pivot.values
    for uid, r in zip(train_pivot.index, vals):
        nz = np.nonzero(r)[0]
        if len(nz) == 0:
            continue
        ids.append(uid)
        contexts.append(torch.tensor(r[nz[0]:], dtype=torch.float32))

    preds = []
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        # primo argomento posizionale: si chiama 'context' nelle versioni vecchie
        # di chronos-forecasting e 'inputs' in quelle recenti
        q, _ = pipe.predict_quantiles(batch, prediction_length=horizon,
                                      quantile_levels=[0.5])
        preds.append(q[:, :, 0].cpu().numpy())  # mediana
    yhat = np.vstack(preds)

    out = pd.DataFrame(yhat, index=ids, columns=future_cols)
    out = out.reindex(index=train_pivot.index).fillna(0)
    return out.clip(lower=0).round(0)
