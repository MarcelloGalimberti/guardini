# Piano di miglioramento forecasting Guardini

Data: 2026-07-11 — rev. 1 (decisioni Marcello recepite)

## 1. Fotografia dello stato attuale

### Dati (dati_20260707, dopo i filtri della pipeline v8)

| Voce | Valore |
|---|---|
| Storico disponibile | 35 mesi (lug 2023 → mag 2026) |
| Codici articolo dopo filtri (classe "PF interno+confez.") | 1.254 |
| Mediana mesi con vendite ≠ 0 per codice | 6 |
| Codici con < 12 mesi di vendite | 801 (64%) |
| Codici con ≥ 24 mesi di vendite | 209 (17%) |
| Classificazione SB (ADI 1.40 / CV 0.80) | Smooth 134, Erratic 132, Intermittent 484, Lumpy 218, non classificabili ~286 |
| Quota volume gestita da NeuralProphet (Smooth+Erratic) | ~59% |
| Concentrazione | top 50 codici = 46% del volume |

Il portafoglio è quindi dominato da serie **corte e intermittenti**: solo ~270 codici hanno una domanda regolare, ma valgono più di metà del volume.

### Architettura v8

- Smooth/Erratic → NeuralProphet globale (trend/stagionalità locali, n_lags=12, m12 Fourier order 6, 60 epoche) + guardrail anti-picco (cap 1.5× max storico).
- Tutte le altre classi → forecast piatto "SBA": `Mean Demand / ADI × 0.95`.
- Clienti promo esclusi a monte; era-diventa transitivo; esclusioni articoli.
- Output: orizzonte 3-4 mesi per J-Galileo (906 articoli nell'ultimo run).

## 2. Criticità individuate (in ordine di impatto)

1. **Manca un backtest vero.** Il MAE/NMAE "ultimi 12 mesi" è calcolato in-sample: il modello viene addestrato su tutto lo storico e poi valutato sugli stessi dati. Le metriche sono ottimistiche e — soprattutto — non permettono di confrontare alternative. Qualsiasi miglioramento va misurato con validazione out-of-sample, altrimenti si naviga a vista.
2. **NeuralProphet è sovradimensionato per serie di 35 punti mensili.** Una rete con 12 lag e stagionalità Fourier di ordine 6 su meno di 3 cicli annui completi tende a overfittare (il picco di luglio corretto in v8 ne è il sintomo; il guardrail anti-picco è un cerotto, non una cura). L'evidenza delle M-competitions è consistente: su serie mensili corte i metodi statistici (ETS, ARIMA, Theta) e i gradient boosting globali battono le reti tipo Prophet. NeuralProphet è inoltre un progetto poco mantenuto.
3. **L'"SBA" attuale non è SBA.** È un rate medio statico (`Mean/ADI × 0.95`), senza smoothing esponenziale di taglie e intervalli: reagisce male ai cambi di livello, ignora il trend e per i codici "New" sottostima il ramp-up. Copre ~700 codici e ~40% del volume.
4. **Metrica di errore non pesata per volume.** L'NMAE medio per codice tratta allo stesso modo un articolo da 50 pz/anno e uno da 50.000. La metrica di riferimento dovrebbe essere il WAPE (errore assoluto totale / venduto totale), eventualmente per classe.
5. **Parametri Optuna congelati** (feb 2026, top-30 articoli, weighted MAE 8548) e comunque globali: un solo set di iperparametri per tutte le serie Smooth/Erratic.
6. **Promozioni eliminate, non modellate.** Il venduto dei 39 clienti promo viene tolto dal dataset. Da verificare col cliente se il forecast debba coprire solo la domanda base (allora va bene) o la domanda totale (allora servono regressori promo).

## 3. Alternative valutate

| Opzione | Cosa offre | Giudizio |
|---|---|---|
| **statsforecast (Nixtla)** | AutoETS, AutoARIMA, AutoTheta + famiglia intermittente vera (CrostonSBA, TSB, IMAPA, ADIDA), fit di migliaia di serie in secondi | **Candidato principale.** Adatta a serie corte, robusta, leggera (niente torch), ben mantenuta |
| **mlforecast + LightGBM** | Modello globale con feature (lag, mese, rolling stats): impara pattern cross-serie, prezioso quando i singoli storici sono corti | **Challenger** per Smooth/Erratic, e per i "New" (impara da codici simili) |
| **TSB** (Teunter-Syntetos-Babai) | Variante Croston che aggiorna la probabilità di domanda a ogni periodo: gestisce l'obsolescenza | Da usare per Intermittent/Lumpy con rischio dismissione |
| Modelli fondazionali (TimeGPT, Chronos) | Zero-shot, nessun training | Esperimento opzionale in coda; dipendenza esterna/API |
| Restare su NeuralProphet ottimizzato | Continuità col codice | Sconsigliato come motore unico: i problemi sono strutturali, non di tuning |

Nota: nessuna libreria "vince a priori". L'architettura probabile è **segmentata**: statistico/boosting per Smooth-Erratic, Croston-family vera per Intermittent-Lumpy, regole dedicate per New. Ma la scelta la fa il backtest, non l'opinione.

## 4. Piano operativo

### Fase 1 — Harness di backtest (fondamenta) ~1 giornata
Script standalone (fuori da Streamlit) che:
- replica esattamente la pipeline dati v8 (promo, era-diventa, esclusioni, pivot mensile);
- esegue rolling-origin cross-validation: ~6 cutoff mensili (da nov 2025 ad apr 2026), orizzonte 3-4 mesi;
- calcola le metriche concordate: **WAPE cumulato sull'orizzonte** (metrica primaria — il forecast serve ad approvvigionare materia prima a lungo lead time, quindi conta la quantità totale sui 3-4 mesi, non il timing mensile) e **bias** (l'errore è asimmetrico: sottostima = stock-out, sovrastima = magazzino), per classe SB e aggregate pesate per volume; MASE come controllo secondario;
- misura le baseline: Seasonal Naive, media mobile 3m, e la v8 attuale (NP + SBA) riprodotta offline.

Deliverable: tabella "quanto sbaglia davvero la v8" — il punto zero.

### Fase 2 — Torneo modelli ~1-2 giornate
Sullo stesso harness:
- statsforecast: AutoETS, AutoARIMA, AutoTheta, CrostonSBA, TSB, IMAPA;
- LightGBM globale (mlforecast);
- ensemble semplici (media dei 2-3 migliori);
- selezione champion per classe SB (o per serie, se paga).

Deliverable: classifica modelli per classe + stima del miglioramento vs v8 (WAPE).

### Fase 3 — Decisione insieme ✅ (2026-07-11)
Architettura **segmentata confermata** (WAPE cum. 47,2% vs 66,5% v8, dettagli in
`backtest/REPORT_FASE2.md`): AutoETS per Smooth, ADIDA per Erratic, CrostonSBA per
Intermittent, AutoARIMA per Insufficient Data, rate SBA piatto per Lumpy e New.
NeuralProphet/torch eliminati dalle dipendenze. Classe New: rate SBA ×1,0 con **warning
obbligatorio** in UI e output (nessun modello affidabile su codici <6 mesi; verificare i
lanci importanti col cliente). Testati anche XGBoost (non competitivo) e Chronos-Bolt
(2° assoluto ma bias −27%, non entra nel champion).

### Fase 4 — Integrazione app (v9) ✅ (2026-07-11, opzione A: prima Streamlit, poi React)
Consegnati: `engine/` (pipeline + forecast segmentato + output J-Galileo, riusabile dal
futuro backend FastAPI), `guardini_j_galileo_v9.py` (stessa UI e stesso file output della
v8, motore nuovo, warning New in evidenza, nota di accuratezza out-of-sample per classe),
`requirements.txt` e `Dockerfile` alleggeriti (rimossi torch/neuralprophet/scipy, aggiunto
statsforecast). Verifiche: struttura file J-Galileo identica alla v8 (stesse colonne,
stessi mesi), fit completo in ~15 s, app avviata senza errori. Il frontend
FastAPI+React (sezione 6) diventa lo step successivo dopo il deploy.

### Fase 5 — Validazione e deploy
Run parallelo v8 vs v9 sui dati correnti, verifica del file per J-Galileo, deploy su Hetzner (pattern esistente).

## 5. Decisioni prese (2026-07-11)

1. **Clienti promo**: esclusione confermata come scelta di business — il forecast copre solo la domanda base. Nessun regressore promo da aggiungere.
2. **Metrica**: errore sull'intero orizzonte di forecast (uso: approvvigionamento materia prima a lungo lead time). Metrica primaria = WAPE cumulato sull'orizzonte; bias tracciato separatamente per gestire l'asimmetria stock-out/magazzino.
3. **Codici critici**: non esistono; 10118FATBBAES era solo un esempio del malfunzionamento v6 che ha originato la v8. Il backtest valuta l'intero portafoglio.
4. **Runtime**: vincolo server CPX22 confermato; statsforecast/LightGBM più leggeri di NeuralProphet, nessun problema atteso.

## 6. Stack tecnologico v9 (proposta)

Obiettivo: sostituire Streamlit con backend Python + frontend React per un'esperienza utente migliore (niente rerun completi, stato gestito correttamente, UI professionale per il cliente).

### Architettura

| Layer | Tecnologia | Note |
|---|---|---|
| Motore forecasting | Package Python puro (`guardini_engine/`) | Nasce già nelle Fasi 1-2: pipeline dati + modelli + backtest, zero dipendenze UI. Riusabile da CLI, API e test |
| Backend API | **FastAPI** + Pydantic | Endpoint: upload file J-Galileo, validazione, run forecast, download Excel output, metriche. Con statsforecast il fit è questione di secondi → API sincrona o job leggeri in background, niente code esterne |
| Frontend | **React + Vite + TypeScript** | UI kit (es. Mantine o shadcn/ui), Plotly.js per i grafici (continuità visiva con l'attuale), TanStack Query per lo stato server |
| Deploy | Docker multi-stage: build React → statici serviti da FastAPI | **Un solo container**, stesso pattern Hetzner attuale (nginx di sistema → porta interna). Nessun cambiamento all'infrastruttura |

### Sequenza consigliata (disaccoppiata)

Il motore (Fasi 1-3) è il prerequisito comune e non dipende dalla scelta UI. Due strade per la Fase 4, da decidere dopo la Fase 3:

- **Opzione A — due step**: v9 Streamlit col motore nuovo (rilascio rapido, accuratezza subito in produzione), poi v10 FastAPI+React con calma. Rischio basso, valore anticipato.
- **Opzione B — tutto insieme**: v9 direttamente FastAPI+React. Rilascio unico ma più lontano nel tempo (+3-5 giornate rispetto all'opzione A per API, frontend e collaudo).

In entrambi i casi il file di output per J-Galileo resta identico.

## Riferimenti

- [statsforecast — CrostonSBA (Nixtla)](https://nixtlaverse.nixtla.io/statsforecast/docs/models/crostonsba.html)
- [Kourentzes — Intermittent demand forecasts with neural networks](https://kourentzes.com/forecasting/2013/04/19/intermittent-demand-forecasts-with-neural-networks/)
- [Combining Probabilistic Forecasts of Intermittent Demand (arXiv)](https://arxiv.org/pdf/2304.03092)
- [intermittent-forecast (PyPI): Croston, SBA, TSB, ADIDA](https://pypi.org/project/intermittent-forecast/1.0.0/)
