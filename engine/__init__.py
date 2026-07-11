# Engine di forecasting Guardini v9 — statsforecast segmentato per classe SB.
# Architettura scelta in Fase 3 (vedi backtest/REPORT_FASE2.md):
# Smooth -> AutoETS | Erratic -> ADIDA | Intermittent -> CrostonSBA
# Insufficient Data -> AutoARIMA | Lumpy, New -> rate SBA piatto (New con warning)
from .pipeline import load_pivot, classify, costruisci_mappa_transitiva, DEFAULTS  # noqa: F401
from .forecast import forecast_segmentato, CHAMPIONS, CLASSI_WARNING  # noqa: F401
from .output import build_galileo  # noqa: F401
