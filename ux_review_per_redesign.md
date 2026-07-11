# Design Critique — App Forecasting Guardini (v8/v9 Streamlit)

Data: 2026-07-11. Scopo: base di lavoro per il redesign FastAPI + React.
Contesto: strumento usato da Marcello (operatore esperto) e mostrato/consegnato al
cliente Guardini (pianificazione/acquisti, utente business non tecnico).

## Impressione generale

L'app è funzionalmente completa ma è organizzata come un notebook di analisi, non come
un prodotto: un'unica pagina infinita in cui configurazione, diagnostica e risultato
finale hanno tutti lo stesso peso visivo. Il valore per il cliente (il file J-Galileo e
la fiducia nei numeri) è sepolto in fondo alla pagina, sotto strati di tabelle tecniche.
La più grande opportunità del redesign: trasformare un flusso da analista in un flusso a
step con un risultato-dashboard di cui il cliente si fida a colpo d'occhio.

## Usabilità

| Problema | Gravità | Raccomandazione per la versione React |
|---|---|---|
| Flusso lineare implicito: upload → filtri → parametri → conferma → forecast → export vivono in un'unica pagina scrollabile; l'utente non sa a che punto è né cosa manca | 🔴 | Wizard a 4 step espliciti: **1. Dati → 2. Classificazione → 3. Forecast → 4. Risultati/Export**, con stepper visibile e stato per step |
| Doppie conferme non standard ("OK - Conferma Parametri", "Procedi", "Reset") — workaround dei rerun Streamlit, non pattern UI riconoscibili | 🔴 | Con React spariscono: bottone primario unico per step, stato gestito dal backend |
| Upload di 3-4 file con tracciato preciso e zero validazione preventiva: file sbagliato → errore Python criptico (KeyError su colonna) | 🔴 | Dropzone per file con validazione immediata dello schema (colonne attese, righe lette, intervallo date) e messaggi d'errore in italiano; card verde/rossa per file |
| Nessuna persistenza: refresh del browser = si riparte da zero; nessuno storico dei run | 🔴 | Il backend salva ogni run (input, parametri, output): elenco run precedenti, riapertura, **confronto run attuale vs precedente** |
| Attesa fit (1-3 min) con spinner generico | 🟡 | Progress per segmento ("AutoARIMA: 494 codici — 60%") via polling/SSE; il backend lo sa già (`progress_cb`) |
| Parametri esperti (ADI, CV, mesi minimi...) sempre esposti nel flusso principale | 🟡 | Pannello "Impostazioni avanzate" collassato, default validati dal backtest; il 95% dei run non li tocca |
| Download sparsi lungo la pagina (forecast dettaglio, elenco New, file J-Galileo) | 🟡 | Pannello unico "Esportazioni" nello step Risultati, col file J-Galileo come azione primaria |
| Warning New: testuale, lontano dalla tabella dei numeri | 🟡 | Badge ⚠️ a livello di riga + contatore nel KPI header + pannello dedicato "Codici da verificare col cliente" |

## Gerarchia visiva

- **Cosa cattura l'occhio per primo**: il logo Guardini (400px) e il pivot 1.254×35 —
  entrambi sbagliati come punto focale. Il logo va in una topbar compatta; il pivot in
  una vista dati secondaria.
- **Ordine di lettura**: oggi è cronologico rispetto alla pipeline interna (l'ordine in
  cui il codice elabora), non rispetto alle domande dell'utente ("quanto ordino?",
  "di cosa mi fido?"). Il redesign deve invertire: prima il risultato, poi il perché.
- **Enfasi**: il numero più importante (volume totale previsto per mese, codici da
  verificare) non è mai messo in evidenza. → KPI header nello step Risultati: volume
  previsto per mese, n. codici, % volume per classe di affidabilità, n. New da verificare.
- **Il treemap volumi per codice (presente in v8, perso in v9) va reintrodotto**: è la
  vista di sintesi che il cliente capisce subito. Suggerimento: treemap cliccabile nello
  step Risultati — click sul rettangolo → drill-down storico+forecast del codice.

## Coerenza e linguaggio

| Elemento | Problema | Raccomandazione |
|---|---|---|
| Terminologia | ADI, CV, WAPE, "Insufficient Data", AutoETS esposti al cliente | Due livelli: di default etichette business ("Domanda regolare / irregolare / sporadica / nuova"), dettaglio tecnico on-demand (tooltip/pannello); mappa: Smooth→Regolare, Erratic→Irregolare, Intermittent/Lumpy→Sporadica, Insufficient Data→Storico limitato, New→Nuovo articolo |
| Colori | Arancione Streamlit arbitrario + color map matplotlib (green/orange/blue/red/purple/gray) | Design token dal brand Guardini (dal logo) + scala semantica coerente per le classi, riusata OVUNQUE (matrice, treemap, badge, KPI) |
| Numeri | Tabelle senza formattazione (float grezzi, nessun separatore migliaia) | Formattazione it-IT (1.254, non 1254), allineamento a destra, unità esplicite (pz) |
| Grafici | Stili plotly di default, titoli tecnici | Tema unico (font, colori, griglie), titoli in linguaggio business |

## Accessibilità

- **Contrasto**: i default Streamlit passano; nel redesign verificare la scala colori
  classi su sfondo bianco (il giallo/arancio per "Erratic" rischia < 4.5:1 sul testo).
- **Target**: pills e toggle Streamlit sono piccoli; in React usare componenti con
  target ≥ 40px (tabelle incluse: righe cliccabili, non solo icone).
- **Tabelle grandi**: virtualizzazione + ricerca + ordinamento (TanStack Table), mai
  1.254 righe renderizzate nude.
- **Lingua**: tutta l'interfaccia in italiano coerente (oggi mescola italiano e inglese:
  "Forecastability", "Download Excel workbook").

## Cosa funziona bene (da conservare)

- La matrice ADI/CV con soglie tratteggiate: ottimo strumento per spiegare al cliente
  *perché* certi codici sono difficili — va tenuta, con le etichette business.
- Il riepilogo per classe con % volume: è la base perfetta per il KPI header.
- Il warning New con elenco scaricabile (v9): giusto concetto, da promuovere a
  cittadino di prima classe della UI.
- La trasparenza sul metodo per codice (colonna "Metodo" v9): differenziante
  professionale, da tenere come badge.
- Il file J-Galileo identico al tracciato del gestionale: non toccarlo.

## Raccomandazioni prioritarie per il redesign React

1. **Wizard a 4 step + dashboard risultati.** Step 1 Dati (dropzone con validazione),
   Step 2 Classificazione (matrice + riepilogo classi, avanzate collassate), Step 3
   Forecast (orizzonte, guardrail, progress per segmento), Step 4 Risultati (KPI header,
   treemap cliccabile, tabella interattiva con badge metodo/warning, pannello export).
   Risolve da solo i primi tre problemi critici.
2. **Persistenza dei run con confronto.** Ogni run salvato dal backend; vista "questo
   run vs precedente" (volumi per classe, codici entrati/usciti, delta forecast).
   Trasforma lo strumento da calcolatrice a sistema di lavoro — e per il cliente è la
   prova di controllo del processo.
3. **Linguaggio a due livelli.** Business di default, tecnica on-demand. È la
   differenza tra un tool che il cliente usa con supervisione e uno che adotta.
4. **Design system minimo ma rigoroso.** Palette dal brand Guardini, scala semantica
   unica per le classi di domanda, formattazione numeri it-IT, tema grafici unico.
   Stack suggerito: Tailwind + shadcn/ui + TanStack Table/Query + Plotly.js.

## Struttura proposta (mappa schermate)

```
Topbar: logo compatto | nome run | stato | [Run precedenti]
├── Step 1 — Dati        dropzone × 4 (venduto, era-diventa, promo, esclusioni)
│                        card di validazione per file, CTA "Continua"
├── Step 2 — Classi      KPI: codici attivi, % volume per classe (etichette business)
│                        matrice ADI/CV | [Avanzate ▸ soglie]
├── Step 3 — Forecast    orizzonte, guardrail | progress per segmento
└── Step 4 — Risultati   KPI header: volume/mese, codici, ⚠️ New da verificare
                         treemap volumi (drill-down per codice)
                         tabella forecast (ricerca, badge metodo, ⚠️ per riga)
                         [Esporta J-Galileo] [Dettaglio Excel] [Elenco New]
```
