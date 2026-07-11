# Backtest forecasting Guardini — Fase 1

Harness di rolling-origin cross-validation (piano: `piano_miglioramento_forecasting.md`).

- 6 cutoff mensili: set 2025 → feb 2026, orizzonte 3 mesi
- Metrica primaria: **WAPE cumulato sull'orizzonte** (uso: approvvigionamento MP a lungo lead time)
- Bias separato (segno: + = sovrastima), WAPE mensile e MASE come controlli
- La pipeline dati replica esattamente la v8 (promo, era-diventa transitivo, esclusioni, classificazione SB con default ADI 1.40 / CV 0.80)

## File

| File | Contenuto |
|---|---|
| `pipeline.py` | Pipeline dati + classificazione SB (replica v8, senza Streamlit) |
| `models.py` | Baseline (SeasonalNaive, MediaMobile3, SBA_v8) + composito v8 (NeuralProphet Iron + guardrail 1.5×) |
| `metrics.py` | WAPE cumulato, bias, WAPE mensile, MASE |
| `run_backtest.py` | Orchestrazione; i run successivi si accumulano in `results/backtest_detail.csv` |

## Run baseline (già eseguito, risultati in `results/`)

```bash
python backtest/run_backtest.py --data-dir "dati/dati_20260707" --models base
```

## Run v8 (NeuralProphet) — DA ESEGUIRE IN LOCALE

Richiede torch/neuralprophet, quindi va lanciato nel conda env locale:

```bash
conda activate neuralprophet
cd /Users/marcello_galimberti/Developer/forecasting_np/Guardini
python backtest/run_backtest.py --data-dir "dati/dati_20260707" --models v8
```

Tempo stimato: ~5-20 min (6 fit NeuralProphet su ~230 serie Smooth/Erratic ciascuno).
I risultati si aggiungono a quelli baseline già presenti; il riepilogo aggiornato viene
riscritto in `results/backtest_summary.xlsx`.

## Run Chronos-Bolt (modello fondazionale) — DA ESEGUIRE IN LOCALE

Zero-shot, modello scaricato da HuggingFace ed eseguito in locale (nessun dato esce):

```bash
conda activate neuralprophet
pip install chronos-forecasting
cd /Users/marcello_galimberti/Developer/forecasting_np/Guardini
python backtest/run_backtest.py --data-dir "dati/dati_20260707" --models chronos
```

Tempo stimato: ~5-10 min su CPU (chronos-bolt-small, ~1.200 serie × 6 cutoff).

## Risultati punto zero (baseline, pooled 6 cutoff, ~1.200 serie/cutoff)

| Modello | WAPE cum. | Bias | WAPE mensile | MASE med. |
|---|---|---|---|---|
| MediaMobile3 | 63,5% | −10,3% | 82,5% | 0,35 |
| SBA_v8 (tutte le serie) | 63,1% | +10,2% | 90,0% | 0,72 |
| SeasonalNaive | 70,6% | +6,8% | 104,2% | 0,43 |

Lettura: anche i metodi banali sbagliano ~63% della quantità cumulata a 3 mesi — il
portafoglio è difficile (64% dei codici ha <12 mesi di vendite). Il confronto che conta
è v8 vs questi numeri, e poi Fase 2 vs v8. Dettaglio per classe SB e per cutoff nei
fogli di `results/backtest_summary.xlsx`.
