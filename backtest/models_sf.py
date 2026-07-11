# =============================================================================
# models_sf.py — Modelli Fase 2: statsforecast (Nixtla) + LightGBM globale.
# Le serie vengono passate senza gli zeri iniziali (prima della prima vendita):
# gli zeri di pre-vita distorcerebbero livello e stagionalità dei modelli.
# =============================================================================
import numpy as np
import pandas as pd

SEASON = 12


def _to_long_trimmed(train_pivot):
    """Long format (unique_id, ds, y) con leading zeros rimossi per serie."""
    rows = []
    cols = np.array(train_pivot.columns)
    vals = train_pivot.values
    for uid, r in zip(train_pivot.index, vals):
        nz = np.nonzero(r)[0]
        if len(nz) == 0:
            continue
        start = nz[0]
        rows.append(pd.DataFrame({"unique_id": uid, "ds": cols[start:], "y": r[start:]}))
    return pd.concat(rows, ignore_index=True)


def _wide(fc_long, col, train_pivot, future_cols):
    out = fc_long.pivot(index="unique_id", columns="ds", values=col)
    out.columns = [pd.Timestamp(c) for c in out.columns]
    out = out.reindex(index=train_pivot.index, columns=future_cols).fillna(0)
    return out.clip(lower=0).round(0)


def run_statsforecast(train_pivot, horizon, which=None):
    """Ritorna dict {nome_modello: DataFrame wide dei forecast}.
    which: lista di nomi per limitare i modelli (default: tutti)."""
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoETS, AutoARIMA, AutoTheta, CrostonSBA, TSB, IMAPA, ADIDA, SeasonalNaive,
    )

    df_long = _to_long_trimmed(train_pivot)
    future_cols = [train_pivot.columns[-1] + pd.DateOffset(months=i) for i in range(1, horizon + 1)]

    catalogo = {
        "AutoETS": lambda: AutoETS(season_length=SEASON),
        "AutoARIMA": lambda: AutoARIMA(season_length=SEASON),
        "AutoTheta": lambda: AutoTheta(season_length=SEASON),
        "CrostonSBA": lambda: CrostonSBA(),
        "TSB": lambda: TSB(alpha_d=0.2, alpha_p=0.2),
        "IMAPA": lambda: IMAPA(),
        "ADIDA": lambda: ADIDA(),
    }
    nomi = which or list(catalogo)
    models = [catalogo[n]() for n in nomi]
    sf = StatsForecast(models=models, freq="MS", n_jobs=4,
                       fallback_model=SeasonalNaive(season_length=SEASON))
    fc = sf.forecast(df=df_long, h=horizon)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc

    rename = {
        "AutoETS": "AutoETS", "AutoARIMA": "AutoARIMA", "AutoTheta": "AutoTheta",
        "CrostonSBA": "CrostonSBA", "TSB": "TSB", "IMAPA": "IMAPA", "ADIDA": "ADIDA",
    }
    out = {}
    for col, nome in rename.items():
        if col in fc.columns:
            out[nome] = _wide(fc[["unique_id", "ds", col]], col, train_pivot, future_cols)

    return out


def _run_ml(train_pivot, horizon, models):
    """Modelli ML globali via mlforecast: lag, rolling mean, mese."""
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean

    df_long = _to_long_trimmed(train_pivot)
    future_cols = [train_pivot.columns[-1] + pd.DateOffset(months=i) for i in range(1, horizon + 1)]

    fcst = MLForecast(
        models=models,
        freq="MS",
        lags=[1, 2, 3, 6, 12],
        lag_transforms={1: [RollingMean(window_size=3), RollingMean(window_size=6)],
                        12: [RollingMean(window_size=3)]},
        date_features=["month"],
    )
    fcst.fit(df_long, static_features=[])
    fc = fcst.predict(h=horizon)
    out = {}
    for nome in models:
        out[nome] = _wide(fc.rename(columns={nome: "yhat"})[["unique_id", "ds", "yhat"]],
                          "yhat", train_pivot, future_cols)
    return out


def run_lgbm(train_pivot, horizon):
    import lightgbm as lgb
    m = {"LightGBM_globale": lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=0, verbosity=-1)}
    return _run_ml(train_pivot, horizon, m)["LightGBM_globale"]


def run_xgb(train_pivot, horizon):
    import xgboost as xgb
    m = {"XGBoost_globale": xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        min_child_weight=20, subsample=0.9, colsample_bytree=0.9,
        random_state=0, verbosity=0)}
    return _run_ml(train_pivot, horizon, m)["XGBoost_globale"]
