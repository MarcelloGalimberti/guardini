# Report Fase 2 — Torneo modelli

Data: 2026-07-11. Backtest rolling-origin: 6 cutoff (set 2025 → feb 2026), orizzonte 3 mesi,
~1.200 serie per cutoff. Metrica primaria: WAPE cumulato sull'orizzonte. Bias: + = sovrastima.

## Classifica generale

| Modello | WAPE cum. | Bias | WAPE mensile |
|---|---|---|---|
| **v9 segmentato** (champion per classe) | **47,2%** | −5,9% | 73,6% |
| AutoARIMA | 54,2% | −0,5% | 81,3% |
| Chronos-Bolt (fondazionale, zero-shot) | 56,6% | **−27,4%** | 72,9% |
| ADIDA | 57,2% | +1,7% | 82,7% |
| AutoETS | 57,9% | +6,9% | 85,8% |
| AutoTheta | 58,1% | −7,8% | 79,9% |
| IMAPA | 58,1% | +1,7% | 83,1% |
| LightGBM globale | 59,8% | +3,8% | 86,6% |
| TSB | 62,8% | +5,6% | 87,2% |
| SBA v8 (piatto, tutte le serie) | 63,1% | +10,2% | 90,0% |
| MediaMobile3 | 63,5% | −10,3% | 82,5% |
| **v8 attuale (NP+SBA)** ¹ | **66,5%** | **+12,8%** | 94,4% |
| SeasonalNaive | 70,6% | +6,8% | 104,2% |
| CrostonSBA (su tutte) ² | 152,1% | +109,0% | 181,1% |
| XGBoost globale ³ | 171,5% | +123,9% | 196,8% |

¹ Dal run locale (i dettagli per serie sono andati persi per un problema tecnico di run
concorrenti, ora risolto; i totali e il per-classe erano stati salvati). Rilanciabile con
`--models v8` per ripristinare il dettaglio.
² CrostonSBA applicato indiscriminatamente sovrastima le serie morte/spente: va usato solo
sulla classe Intermittent, dove è champion (41,1%).
³ Testato su richiesta con le stesse feature di LightGBM e in 3 configurazioni (depth 3,
depth 6, lossguide 31 foglie): sempre fuori scala (114-171% WAPE, bias +73/+124%). Sul
forecasting ricorsivo di serie sparse LightGBM gestisce molto meglio gli zeri; XGBoost
compone sovrastime a cascata. Non vale ulteriore tuning: anche il migliore dei GBM
(LightGBM, 59,8%) resta dietro ai metodi statistici.

## Modelli fondazionali

- **Chronos-Bolt (Amazon)**, zero-shot, eseguito in locale (run del 2026-07-11): 2° posto
  assoluto (56,6%) e miglior WAPE mensile/MASE — notevole per un modello che non ha mai
  visto questi dati. Ma: bias **−27,4%** (sottostima forte, rischio stock-out), pessimo su
  Intermittent (55,4%) e New (410,9%). Sugli Smooth pareggia AutoETS (27,9% vs 28,2%) con
  bias però peggiore (−11,5% vs −3,6%): sostituirlo nel segmentato non paga
  (47,1% vs 47,2%, bias peggiore) e costerebbe la dipendenza torch. **Conclusione: valida
  la classifica, non entra nel champion.** Da riconsiderare se in futuro servisse un
  modello unico senza classificazione.
- **TimeGPT (Nixtla)**: API esterna — il venduto Guardini verrebbe inviato a un servizio
  terzo. Da NON testare senza autorizzazione esplicita del cliente.

## Champion per classe (→ composizione del "v9 segmentato")

| Classe | % volume | Champion | WAPE cum. | v8 sulla stessa classe |
|---|---|---|---|---|
| Smooth | 38,0% | **AutoETS** | 28,2% | 38,2% (NeuralProphet) |
| Insufficient Data | 28,6% | **AutoARIMA** | 67,3% | 112,9% (SBA piatto) |
| Intermittent | 16,4% | **CrostonSBA** | 41,1% | 42,3% (SBA piatto) |
| Erratic | 11,5% | **ADIDA** (≈IMAPA) | 49,9% | 69,7% (NeuralProphet) |
| Lumpy | 4,5% | SBA v8 piatto | 86,1% | 86,1% (invariato) |
| New | 1,0% | SBA v8 piatto (min peggio) | 90,2% | 90,2% — **tutti i modelli falliscono** |

## Conclusioni

1. **v9 segmentato: WAPE 47,2% vs 66,5% della v8** → −19 punti (−29% relativo), con bias
   quasi neutro (−5,9% vs +12,8%). Il miglioramento viene da tre fronti: AutoETS batte
   NeuralProphet anche sugli Smooth (il suo terreno di casa), AutoARIMA dimezza l'errore
   sugli Insufficient Data (29% del volume, il buco più grande della v8), ADIDA sistema
   gli Erratic.
2. **NeuralProphet perde in tutte le classi** → la v9 può eliminare torch/neuralprophet
   dalle dipendenze (container più piccolo, fit da minuti a secondi).
3. **Classe New irrisolta** (1% del volume): nessun metodo statistico funziona su codici
   con <6 mesi di storico. Opzioni per la Fase 3: media degli ultimi 3 mesi con fattore di
   ramp-up, analogia con codici simili (stessa famiglia 5-digit), o gestione manuale.
4. Il bias del segmentato è −5,9% (leggera sottostima): per l'approvvigionamento MP si può
   valutare un fattore correttivo o la scelta del secondo classificato su classi in
   sottostima (es. AutoETS sugli Intermittent, bias −17,6% → CrostonSBA −13,1% già scelto).

## Appendice — Classe New: esperimento e decisione (Fase 3)

Testate 13 regole (medie/mediane dei mesi non-zero, ultimo valore, MA3 smorzata,
scala del rate SBA da ×0,5 a ×2). Evidenza: i New sono **bimodali** — la maggior parte fa
1-2 ordini iniziali e poi quasi nulla nei 3 mesi successivi (ogni regola più generosa
esplode: media non-zero → WAPE 1391%, bias +1379%), una minoranza fa ramp-up (da cui il
bias negativo del rate basso). Con soli 1-5 mesi di storico le due popolazioni sono
indistinguibili.

| Regola | WAPE cum. | Bias |
|---|---|---|
| Rate SBA ×0,75 | 83,4% (minimo) | −66,6% |
| **Rate SBA ×1,0 (scelta)** | 90,2% | −55,4% |
| Rate SBA ×2,0 | 131,0% | −10,7% |

**Decisione**: rate SBA ×1,0. Il minimo WAPE (×0,75) peggiorerebbe la sottostima proprio
sui prodotti in lancio — il caso in cui lo stock-out è più costoso commercialmente. Impatto
sul totale trascurabile (New = 1% del volume). **Requisito v9**: i codici New devono essere
marcati con un warning esplicito in UI e nel file di output, perché su questa classe
nessun modello è affidabile: per i lanci importanti il dato va verificato dal cliente.

## Riproducibilità

```bash
# baseline + statsforecast + LightGBM (veloci)
python backtest/run_backtest.py --data-dir "dati/dati_20260707" --models base
python backtest/run_backtest.py --data-dir "dati/dati_20260707" --models sf
python backtest/run_backtest.py --data-dir "dati/dati_20260707" --models lgbm
# v8 (richiede conda env neuralprophet)
python backtest/run_backtest.py --data-dir "dati/dati_20260707" --models v8
```

Dettaglio per serie: `results/parts/*.csv` (un file per modello×cutoff, robusto a run
concorrenti) aggregati in `results/backtest_detail.csv` e `results/backtest_summary.xlsx`.
