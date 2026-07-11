# =============================================================================
# models.py — Modelli baseline + riproduzione v8 (NeuralProphet + SBA)
# Ogni modello: f(train_pivot, classi, horizon) -> DataFrame index=Codice,
# colonne = i mesi futuri (Timestamp), valori = forecast.
# =============================================================================
import numpy as np
import pandas as pd

CLASSI_NP = ["Smooth", "Erratic"]


def _future_months(train_pivot, horizon):
    last = train_pivot.columns[-1]
    return [last + pd.DateOffset(months=i) for i in range(1, horizon + 1)]


def _empty(train_pivot, horizon):
    return pd.DataFrame(0.0, index=train_pivot.index, columns=_future_months(train_pivot, horizon))


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def seasonal_naive(train_pivot, classi, horizon):
    """Forecast mese m = venduto di m-12; fallback media ultimi 3 mesi."""
    out = _empty(train_pivot, horizon)
    ma3 = train_pivot.iloc[:, -3:].mean(axis=1)
    for col in out.columns:
        ref = col - pd.DateOffset(months=12)
        if ref in train_pivot.columns:
            out[col] = train_pivot[ref]
        else:
            out[col] = ma3
    return out


def moving_average_3(train_pivot, classi, horizon):
    out = _empty(train_pivot, horizon)
    ma3 = train_pivot.iloc[:, -3:].mean(axis=1)
    for col in out.columns:
        out[col] = ma3
    return out


# ---------------------------------------------------------------------------
# SBA come implementato in v8 (rate statico Mean/ADI * 0.95)
# ---------------------------------------------------------------------------

def sba_v8_flat(classi, ids):
    adi = classi.loc[ids, "ADI"].replace(0, 1)
    rate = (classi.loc[ids, "Mean_Demand"] / adi) * (1 - 0.1 / 2)
    return rate.fillna(0).round(0)


# ---------------------------------------------------------------------------
# NeuralProphet con configurazione "Iron" v8 + guardrail anti-picco
# ---------------------------------------------------------------------------

def np_v8_forecast(train_pivot, ids_np, horizon, cap_factor=1.5):
    """Fit globale NeuralProphet (parametri Iron v8) sulle serie ids_np.
    Ritorna DataFrame index=ID, colonne=mesi futuri."""
    from neuralprophet import NeuralProphet, set_random_seed, set_log_level
    set_log_level("ERROR")
    set_random_seed(0)

    df_series = train_pivot.loc[ids_np].reset_index()
    df_long = df_series.melt(id_vars="Codice Articolo", var_name="ds", value_name="y")
    df_long.rename(columns={"Codice Articolo": "ID"}, inplace=True)
    df_long["ds"] = pd.to_datetime(df_long["ds"])
    df_long = df_long.sort_values(["ID", "ds"]).reset_index(drop=True)

    m = NeuralProphet(
        trend_global_local="local",
        season_global_local="local",
        seasonality_mode="multiplicative",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        n_lags=12, n_forecasts=horizon,
        ar_layers=[],
        learning_rate=0.005,
        n_changepoints=3,
        trend_reg=0.2,
        seasonality_reg=0.3,
        epochs=60,
        loss_func="MSE",
        normalize="standardize",
    )
    m = m.add_seasonality(name="m12", period=12, fourier_order=6)
    m.fit(df_long, freq="MS", progress=None)

    df_future = m.make_future_dataframe(df_long, n_historic_predictions=False, periods=horizon)
    fc = m.predict(df_future)

    # clip a 0 (Iron clipping v8)
    yhat_cols = [c for c in fc.columns if c.startswith("yhat")]
    for c in yhat_cols:
        fc[c] = fc[c].clip(lower=0)

    # guardrail anti-picco: cap = fattore × max storico mensile del codice
    caps = train_pivot.loc[ids_np].max(axis=1) * cap_factor
    cap_map = caps.to_dict()
    cap_series = fc["ID"].map(lambda i: cap_map.get(i, np.inf))
    for c in yhat_cols:
        fc[c] = np.minimum(fc[c], cap_series)

    # per le righe future (y NaN) prendi il primo yhat non nullo (= latest forecast)
    fut = fc[fc["y"].isna()].copy()

    def latest(row):
        for i in range(1, horizon + 1):
            v = row.get(f"yhat{i}")
            if pd.notna(v):
                return v
        return 0.0

    fut["forecast"] = fut.apply(latest, axis=1)
    out = fut.pivot(index="ID", columns="ds", values="forecast").fillna(0)
    out.columns = [pd.Timestamp(c) for c in out.columns]
    return out


def v8_composite(train_pivot, classi, horizon, cap_factor=1.5):
    """Riproduzione v8: NeuralProphet per Smooth/Erratic, SBA piatto per il resto."""
    out = _empty(train_pivot, horizon)
    ids_np = classi.index[classi["Classe"].isin(CLASSI_NP)].tolist()
    ids_stat = classi.index[~classi["Classe"].isin(CLASSI_NP)].tolist()

    if ids_stat:
        flat = sba_v8_flat(classi, ids_stat)
        for col in out.columns:
            out.loc[ids_stat, col] = flat

    if ids_np:
        fc_np = np_v8_forecast(train_pivot, ids_np, horizon, cap_factor=cap_factor)
        common_cols = [c for c in out.columns if c in fc_np.columns]
        out.loc[fc_np.index, common_cols] = fc_np[common_cols].values

    return out.round(0)


MODELLI_BASE = {
    "SeasonalNaive": seasonal_naive,
    "MediaMobile3": moving_average_3,
}
