# =============================================================================
# metrics.py — Metriche concordate: WAPE cumulato sull'orizzonte (primaria),
# bias cumulato, WAPE mensile (controllo timing), MASE mediano (controllo).
# =============================================================================
import numpy as np
import pandas as pd


def mase_scale(train_pivot):
    """Scala MASE per serie: MAE in-sample del naive stagionale (lag 12),
    fallback naive lag 1 se lo storico è corto."""
    vals = train_pivot.values
    n = vals.shape[1]
    scales = []
    for r in vals:
        if n > 12:
            errs = np.abs(r[12:] - r[:-12])
        else:
            errs = np.abs(np.diff(r))
        s = errs.mean() if len(errs) else np.nan
        scales.append(s if s and s > 0 else np.nan)
    return pd.Series(scales, index=train_pivot.index)


def evaluate(forecast, actual, classi, scale, cutoff, modello):
    """forecast/actual: DataFrame index=ID, colonne=mesi test (allineate).
    Ritorna df per-serie con F_cum, A_cum, err assoluti mensili, MASE."""
    cols = list(forecast.columns)
    ids = forecast.index
    A = actual.reindex(index=ids, columns=cols).fillna(0)
    F = forecast.fillna(0)

    f_cum = F.sum(axis=1)
    a_cum = A.sum(axis=1)
    abs_monthly = (F - A).abs().sum(axis=1)
    mae_monthly = (F - A).abs().mean(axis=1)
    mase = mae_monthly / scale.reindex(ids)

    return pd.DataFrame({
        "cutoff": str(cutoff.date()),
        "modello": modello,
        "ID": ids,
        "Classe": classi.reindex(ids)["Classe"].values,
        "F_cum": f_cum.values,
        "A_cum": a_cum.values,
        "abs_err_cum": (f_cum - a_cum).abs().values,
        "err_cum": (f_cum - a_cum).values,
        "abs_err_monthly_sum": abs_monthly.values,
        "MASE": mase.values,
    })


def summarize(df, by=("modello",)):
    """Aggrega le metriche pooled su serie e cutoff."""
    g = df.groupby(list(by))
    out = pd.DataFrame({
        "WAPE_cum_%": g.apply(lambda x: 100 * x["abs_err_cum"].sum() / max(x["A_cum"].sum(), 1e-9), include_groups=False),
        "Bias_%": g.apply(lambda x: 100 * x["err_cum"].sum() / max(x["A_cum"].sum(), 1e-9), include_groups=False),
        "WAPE_mensile_%": g.apply(lambda x: 100 * x["abs_err_monthly_sum"].sum() / max(x["A_cum"].sum(), 1e-9), include_groups=False),
        "MASE_mediano": g.apply(lambda x: x["MASE"].median(), include_groups=False),
        "n_serie_cutoff": g.size(),
        "Volume_actual": g["A_cum"].sum(),
    }).reset_index()
    return out.round(2)
