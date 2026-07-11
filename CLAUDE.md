# Guardini — Forecasting domanda materiali

App di forecasting per Guardini. Carica dati di venduto/promozioni/era-diventa
da file Excel esportati da J-Galileo, classifica i codici per tipo di domanda
(Syntetos-Boylan) e produce previsioni caricabili nel gestionale del cliente
(file "df_galileo_per_forecasting.xlsx", tracciato fisso).

## Versione corrente e direzione (aggiornato 2026-07-11)

- **In produzione (Docker su Hetzner): `guardini_j_galileo_v8.py`**
  (NeuralProphet + SBA piatto).
- **`guardini_j_galileo_v9.py`** (Streamlit + motore statsforecast segmentato):
  funzionante, testata in locale da Marcello, **NON verrà deployata**. Decisione
  del 2026-07-11: si passa direttamente a **backend FastAPI + frontend React**
  costruiti attorno al package `engine/`. La v9 resta come riferimento
  funzionale del flusso e del motore.
- **`engine/`** è il cuore riusabile (e sarà il core del backend FastAPI):
  `pipeline.py` (caricamento/filtri/pivot/classificazione SB — copia
  autoritativa; `backtest/pipeline.py` è la copia congelata per riproducibilità),
  `forecast.py` (motore segmentato), `output.py` (tracciato J-Galileo).
- Le versioni `guardini_j_galileo.py`, `_v2`...`_v6` sono storiche, non deployate.
  `modelli/optimal_params_np.json` serve solo alla v8.
- **Webapp FastAPI + React: COSTRUITA (2026-07-11), pronta per test UI e deploy.**
  `backend/` (API: run persistiti in parquet+json, validazione upload con
  messaggi in italiano, job forecast in thread con progress, export, confronto
  run) + `frontend/` (React+Vite+TS+Tailwind: wizard 4 step, dashboard con KPI,
  treemap cliccabile, tabella con badge business, warning New, storico run).
  Vedi `README_WEBAPP.md` per sviluppo locale e deploy. Dockerfile (multi-stage
  node→python), docker-compose (volume `guardini_runs`) e requirements.txt
  puntano già alla webapp; `nginx.conf` invariato. Backend testato end-to-end
  coi dati reali; la UI browser va provata da Marcello prima del deploy.

## Motore di forecasting v9+ (decisioni chiave, sessione 2026-07-11)

Architettura scelta con backtest rolling-origin (6 cutoff set 2025→feb 2026,
orizzonte 3 mesi, metrica primaria WAPE cumulato sull'orizzonte — l'output serve
per approvvigionare MP a lungo lead time; bias tracciato separatamente).
Risultato: **WAPE cumulato 47,2% vs 66,5% della v8**, bias −5,9% vs +12,8%.
Dettagli in `backtest/REPORT_FASE2.md`; processo in
`piano_miglioramento_forecasting.md`.

| Classe SB | Modello | Note |
|---|---|---|
| Smooth | AutoETS | batte NeuralProphet anche qui (28% vs 38%) |
| Erratic | ADIDA | |
| Intermittent | CrostonSBA | |
| Insufficient Data | AutoARIMA | era il buco più grande della v8 (113%→67%) |
| Lumpy | rate SBA piatto | |
| New | rate SBA piatto + **WARNING obbligatorio** | nessun modello affidabile su codici <6 mesi: l'utente deve verificare i lanci col cliente |

Altre decisioni: esclusione clienti promo confermata (scelta di business,
forecast = domanda base); NeuralProphet/torch eliminati dalle dipendenze;
XGBoost testato e scartato (non competitivo su serie sparse); Chronos-Bolt
testato (2° assoluto ma bias −27%, non entra); TimeGPT NON testato (API
esterna: i dati del cliente uscirebbero — serve autorizzazione esplicita).

## Trappole note del motore

- **statsforecast con `n_jobs>1` crasha dentro Streamlit** (BrokenProcessPool,
  spawn su macOS / fork nel server): `engine/forecast.py` usa `n_jobs=1`.
  Nel backend FastAPI si può rivalutare il parallelismo (fuori da Streamlit).
- **chronos-forecasting**: `predict_quantiles` ha rinominato il primo argomento
  (`context`→`inputs` nelle versioni recenti): passarlo posizionale.
- Il file J-Galileo scarta il **primo mese di forecast** (mese corrente già
  avviato) e le righe tutte a zero: comportamento ereditato dalla v8, il
  tracciato colonne è fisso (Cliente/Fornitore, Articolo, Commessa,
  Sotto commessa, Proprietà='0', Magazzino='100', mesi).

## Backtest (`backtest/`)

Harness rolling-origin riusabile per futuri confronti modelli
(`run_backtest.py --data-dir dati/<cartella> --models base|sf|lgbm|xgb|v8|chronos`).
I risultati si salvano in `results/parts/` (un CSV per modello×cutoff, robusto a
run concorrenti — MAI tornare al file unico: una race li ha già corrotti una
volta) e vengono aggregati in `backtest_detail.csv` / `backtest_summary.xlsx`.
I gruppi `v8` e `chronos` richiedono il conda env locale `neuralprophet`
(torch non installabile nella sandbox Cowork).

## Webapp FastAPI + React (costruita 2026-07-11)

Design da `ux_review_per_redesign.md` (critica UX approvata + mappa schermate).
Implementato: wizard a 4 step (Dati → Classificazione → Forecast → Risultati),
KPI header, **treemap volumi cliccabile** con drill-down per codice, linguaggio
a due livelli (business di default: Regolare/Irregolare/Sporadica/Storico
limitato/Nuovo articolo — mapping in `backend/services.py` e
`frontend/src/theme.ts`; tecnica on-demand), persistenza run con confronto,
validazione upload in italiano, warning New in KPI/badge/pannello. Stack
effettivo: FastAPI+Pydantic, React+Vite+TS, Tailwind (componenti custom sullo
stile del mockup approvato, niente shadcn), TanStack Query, Plotly.js
(`plotly.js-dist-min` con d.ts custom in `frontend/src/plotly.d.ts`).

Note tecniche webapp:
- run persistiti su filesystem (`GUARDINI_RUNS_DIR`, default `runs_data/`):
  metadata.json + parquet, nessun DB;
- `GUARDINI_N_JOBS` controlla il parallelismo statsforecast (default 1;
  2 nel container — qui NON c'è il vincolo Streamlit);
- il forecast gira in un thread daemon, il frontend fa polling su
  `GET /api/runs/{id}` (campo `progress`);
- `frontend/dist/` è gitignorato ma una build funzionante è presente in
  locale; in Docker la build avviene nello stage node.

## Deployment: Docker su Hetzner

Segue lo stesso pattern usato per altre app (analisi-clienti,
scavolini-codifica, llg-monitor, istruzioni-montaggio): un container Docker
espone l'app su una porta interna localhost, e l'**nginx di sistema** (già
presente sul server, non nginx-in-Docker) la espone verso l'esterno.

| Voce | Valore |
|------|--------|
| Server | Hetzner CPX22, IP `65.21.182.192` (hostname `streamlit-server`) |
| Path sul server | `/opt/guardini` |
| Repo GitHub | `github.com/MarcelloGalimberti/guardini` (branch `main`) |
| Container | `guardini`, porta interna Docker `127.0.0.1:8504` → `8501` (Streamlit) |
| Porta pubblica nginx | **8507** (http://65.21.182.192:8507), nessuna auth |
| Altri container sullo stesso server | 8501 analisi-clienti, 8502 scavolini-codifica, 8503 llg-monitor, 8506 istruzioni-montaggio (pubblico 8505 con Basic Auth) |

File di deploy nel repo: `Dockerfile`, `requirements.txt`, `docker-compose.yml`,
`nginx.conf`, `.dockerignore`. `nginx.conf` va copiato a mano in
`/etc/nginx/sites-available/guardini` e linkato in `sites-enabled` (fatto una
tantum, non si aggiorna da solo col `git pull`).

### Aggiornare l'app dopo modifiche al codice

```bash
ssh root@65.21.182.192
cd /opt/guardini
git pull
docker compose up -d --build
```

### Log e stato

```bash
docker compose -f /opt/guardini/docker-compose.yml ps
docker logs -f guardini
docker inspect guardini --format 'RestartCount={{.RestartCount}} ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}}'
```

## Bug risolti / decisioni non ovvie

- **`pandas`/`pyarrow` pinnati** (`pandas==2.2.3`, `pyarrow==18.1.0` in
  `requirements.txt`). Senza pin, pip installa l'ultima `pyarrow` disponibile
  (era arrivata alla 25.x), che **segfault** (SIGSEGV, exit code 139) dentro
  `pandas_compat.convert_column` quando Streamlit converte un DataFrame reale
  in Arrow per `st.dataframe()`. Sintomo: l'app "si riavvia" e torna alla
  schermata di upload appena si conferma il caricamento file. Le versioni
  pinnate sono quelle dell'ambiente conda locale `neuralprophet`, verificato
  funzionante. Se in futuro si sblocca il pin, testare con dati reali (un
  fit sintetico a singola serie NON riproduce il bug).
- **Torch CPU-only nel Dockerfile** (`pip install torch==2.5.1 --index-url
  https://download.pytorch.org/whl/cpu` prima di `pip install -r
  requirements.txt`): il server non ha GPU, installare il wheel di default
  trascinerebbe ~2GB di librerie CUDA inutili.
- **`shm_size: 1gb`** e **`PYTHONFAULTHANDLER=1`** in `docker-compose.yml`:
  aggiunti durante il debug del segfault sopra (il secondo stampa nei log la
  riga Python esatta in caso di crash nativo). Innocui, lasciati per
  diagnosi future.
- **`plotly` non `plotly-express`**: la vecchia dipendenza `plotly-express`
  (pacchetto standalone deprecato) causava `Importing plotly failed.
  Interactive plots will not work.` su Streamlit Cloud. Il messaggio
  "Importing plotly failed" che compare comunque nei log del container
  Docker viene da un import interno di neuralprophet (probabilmente
  `plotly_resampler`, non installato) ed è **cosmetico/innocuo** — non è lo
  stesso problema e non impedisce a `plotly.express`/`plotly.graph_objects`
  di funzionare nell'app.
- **Requirements curati vs. pip-freeze**: `requirements.txt` (root, usato dal
  Dockerfile) contiene solo le dipendenze realmente importate da
  `guardini_j_galileo_v8.py` (streamlit, neuralprophet, scipy, plotly,
  openpyxl, xlsxwriter, + pin pandas/pyarrow). Le cartelle `requirements txt
  per v6/` e `requirements txt per v8/` sono riferimenti storici, non usate
  dal deploy.
- **Rimossi dal repo** (superati dal deploy Docker): `.devcontainer/` (config
  per GitHub Codespaces) e `guardini_fcst.py` (prototipo pre-v1). Restano
  nella storia Git se servissero.

## Cosa NON è (ancora) versionato su GitHub

**Da committare (sessione 2026-07-11)**: `engine/`, `backend/`, `frontend/`
(senza `node_modules/` e `dist/`, già in .gitignore), `backtest/` (senza
`results/parts/` se si vuole tenere leggero il repo), `guardini_j_galileo_v9.py`,
`piano_miglioramento_forecasting.md`, `ux_review_per_redesign.md`,
`README_WEBAPP.md`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`,
`.gitignore`, questo CLAUDE.md.

Rimasti solo locali, esclusi dal commit Docker per non allargare lo scope:
`Guardini_Cowork/`, `documentazione/`, `analysis_param.py`, `optimize_np.py`,
`note_v8.md`, `runtime.txt`, i PDF, `requirements txt per v6/` (il contenuto
di `requirements txt per v8/` invece è stato aggiunto). `.gitignore` esclude
inoltre `lightning_logs/` e `dati/` (log di training e dati locali, pesanti e
rigenerabili).
