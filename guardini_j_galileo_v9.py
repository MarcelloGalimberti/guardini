# =====================================================================
# Forecasting Materiali Guardini - v9
# Motore: statsforecast SEGMENTATO per classe Syntetos-Boylan
# (sostituisce NeuralProphet+SBA della v8; architettura scelta col
# backtest rolling-origin — vedi backtest/REPORT_FASE2.md).
#   Smooth -> AutoETS | Erratic -> ADIDA | Intermittent -> CrostonSBA
#   Insufficient Data -> AutoARIMA | Lumpy, New -> rate SBA piatto
# Novità v9:
#   - WAPE cum. backtest 47% vs 67% della v8, bias quasi neutro
#   - niente torch/neuralprophet: fit in ~15s invece di minuti
#   - WARNING obbligatorio sui codici "New" (<6 mesi di storico):
#     nessun modello è affidabile, verificare i lanci col cliente
#   - UI e file di output J-Galileo INVARIATI rispetto alla v8
# =====================================================================

import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings('ignore')

from engine import (  # noqa: E402
    classify, costruisci_mappa_transitiva, forecast_segmentato,
    build_galileo, CHAMPIONS, CLASSI_WARNING,
)

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def scarica_excel(df, filename, label="Download Excel workbook"):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()
    st.download_button(label=label, data=output.getvalue(), file_name=filename,
                       mime="application/vnd.ms-excel", key=f"dl_{filename}")


# ---------------------------------------------------------------------
# Impaginazione
# ---------------------------------------------------------------------
st.set_page_config(layout="wide")
col_1, col_2 = st.columns([2, 3])
with col_1:
    st.image('guardini.png', width=400)
st.divider()
with col_2:
    st.title('Forecasting Materiali v9 | :orange[statsforecast]')
    st.caption('Motore segmentato per classe di domanda (AutoETS · ADIDA · CrostonSBA · AutoARIMA · SBA)')

# ---------------------------------------------------------------------
# Caricamento dati J-Galileo (identico alla v8)
# ---------------------------------------------------------------------

@st.cache_data
def load_venduto(file_data):
    df = pd.read_excel(file_data, engine='openpyxl')
    df = df[['CDCFST', 'DSCFST', 'CDARST', 'DSARST', 'QTFTST', 'DTBOST', 'DCA1ST', 'DCA2ST', 'DC02ST']]
    for col in ['CDCFST', 'DSCFST', 'CDARST', 'DSARST', 'DCA1ST', 'DCA2ST', 'DC02ST']:
        df[col] = df[col].astype(object).fillna('').astype(str)
    df['QTFTST'] = pd.to_numeric(df['QTFTST'], errors='coerce').fillna(0).astype(np.float64)
    s = df['DTBOST'].astype(str).str.zfill(8)
    df['DTBOST'] = pd.to_datetime(s.str[:4] + '-' + s.str[4:6] + '-' + s.str[6:8],
                                  format='%Y-%m-%d', errors='coerce')
    return df


path_venduto = st.sidebar.file_uploader('Caricare il file venduto da J-Galileo (es. "VENDUTO_LUG_23_GIU_26.xlsx")')
if not path_venduto:
    st.warning('"VENDUTO" non caricato, aprire sidebar e caricare file')
    st.stop()
df_venduto = load_venduto(path_venduto)


@st.cache_data
def load_era_diventa(file_data):
    return pd.read_excel(file_data, engine='openpyxl')


path_era_diventa = st.sidebar.file_uploader('Caricare il file "ERA DIVENTA" (db_era_diventa.xlsx)')
if not path_era_diventa:
    st.warning('"ERA DIVENTA" non caricato, aprire sidebar e caricare file')
    st.stop()
df_era_diventa = load_era_diventa(path_era_diventa)


@st.cache_data
def load_promo(file_data):
    df = pd.read_excel(file_data, engine='openpyxl')
    for col in ['CDCFST', 'DSCFST']:
        df[col] = df[col].astype(object).fillna('').astype(str)
    return df


path_promo = st.sidebar.file_uploader('Caricare il file "Promozioni" (db_clienti_promo.xlsx)')
if not path_promo:
    st.warning('"Promozioni" non caricato, aprire sidebar e caricare file')
    st.stop()
df_promo = load_promo(path_promo)


@st.cache_data
def load_articoli_da_escludere(file_data):
    df = pd.read_excel(file_data, engine='openpyxl')
    df['CDARMA'] = df['CDARMA'].astype(object).fillna('').astype(str)
    return df


st.sidebar.divider()
st.sidebar.markdown("**Esclusioni Articoli (Opzionale)**")
path_escludere = st.sidebar.file_uploader(
    'Caricare il file "Articoli da Escludere" (es. "ARTICOLI DA ESCLUDERE.xlsx") — opzionale',
    key='articoli_da_escludere')
df_articoli_da_escludere = load_articoli_da_escludere(path_escludere) if path_escludere else None

# ---------------------------------------------------------------------
# Preprocessing (identico alla v8)
# ---------------------------------------------------------------------
df_venduto.rename(columns={
    'CDCFST': 'Codice Cliente', 'DSCFST': 'Descrizione Cliente',
    'CDARST': 'Codice Articolo', 'DSARST': 'Descrizione Articolo',
    'QTFTST': 'Quantità venduta', 'DTBOST': 'Data',
    'DCA1ST': 'Classe di valutazione 1', 'DCA2ST': 'Classe di valutazione 2',
    'DC02ST': 'Classe di valutazione 3'}, inplace=True)

with st.expander('Anteprima dati caricati'):
    col_a, col_b, col_c, col_d = st.columns([4, 1, 1, 1])
    with col_a:
        st.write('**Venduto:**', df_venduto.shape[0], 'righe |',
                 df_venduto['Codice Articolo'].nunique(), 'codici univoci')
        st.dataframe(df_venduto.head())
    with col_b:
        st.write('**ERA DIVENTA:**', df_era_diventa.shape[0], 'righe')
        st.dataframe(df_era_diventa.head())
    with col_c:
        st.write('**Promozioni:**', df_promo.shape[0], 'clienti')
        st.dataframe(df_promo.head())
    with col_d:
        if df_articoli_da_escludere is not None:
            st.write('**Da escludere:**', df_articoli_da_escludere.shape[0], 'codici')
            st.dataframe(df_articoli_da_escludere.head())
        else:
            st.write('**Articoli da Escludere:** non caricato')

# Clienti promo fuori
df_venduto = df_venduto[~df_venduto['Codice Cliente'].isin(df_promo['CDCFST'])]

# Esclusioni articoli
if df_articoli_da_escludere is not None:
    codici_da_escludere = set(df_articoli_da_escludere['CDARMA'].tolist())
    n_prima = df_venduto['Codice Articolo'].nunique()
    df_venduto = df_venduto[~df_venduto['Codice Articolo'].isin(codici_da_escludere)]
    st.info(f'🚫 Articoli da Escludere applicato: rimossi '
            f'**{n_prima - df_venduto["Codice Articolo"].nunique()} codici** su {n_prima}.')

# Filtro classe di valutazione 1
classi_disponibili = sorted(df_venduto['Classe di valutazione 1'].dropna().unique().tolist())
default_class = 'PF interno+confez.'
default_selection = [default_class] if default_class in classi_disponibili else None
classi_selezionate = st.pills("Filtro Classe di valutazione 1:", options=classi_disponibili,
                              default=default_selection, selection_mode="multi")
if not classi_selezionate:
    st.warning("Seleziona almeno una Classe di valutazione 1 per proseguire.")
    st.stop()
df_venduto = df_venduto[df_venduto['Classe di valutazione 1'].isin(classi_selezionate)]

# Era-diventa transitivo (v8) vs raggruppamento 5 digit
usa_era_diventa = st.toggle("Usa 'Era Diventa' (default)", value=True)

if usa_era_diventa:
    mappa = costruisci_mappa_transitiva(df_era_diventa)
    df_venduto['Codice Articolo'] = df_venduto['Codice Articolo'].map(lambda c: mappa.get(c, c))
    st.caption(f"🔁 Era-diventa: sostituzione transitiva su {len(mappa)} mappature.")
else:
    st.info("ℹ️ Modalità attiva: codici raggruppati per i primi 5 digit.")
    df_venduto['Codice Articolo'] = df_venduto['Codice Articolo'].str[:5]

# Obsoleti (tutte le righe del codice con descrizione '***')
df_venduto['Obsoleto'] = df_venduto['Descrizione Articolo'].str.startswith('***')
df_venduto['Eliminare'] = df_venduto.groupby('Codice Articolo')['Obsoleto'].transform('all')
df_venduto = df_venduto[~df_venduto['Eliminare']]

df_venduto['Descrizione Articolo univoca'] = df_venduto['Descrizione Articolo'].map(
    lambda d: d[3:] if d.startswith('***') or d.startswith('---') else d)

# Liste di esclusione (identiche alla v8)
lista_codici = ('SK', '-', '*', '.', '.SCAR FERROSO', '_', 'ST')
df_venduto = df_venduto[~df_venduto['Codice Articolo'].str.startswith(lista_codici)]
df_venduto = df_venduto[~df_venduto['Descrizione Articolo'].str.contains('SET|EXPO')]
df_venduto = df_venduto[df_venduto['Data'] >= pd.to_datetime('2023-01-01')]

st.info('Eliminati articoli riferiti a promozioni, codici SET e EXPO, codici SK, -, *, ., .SCAR FERROSO, _, ST')

# Aggregazione mensile e pivot
df_venduto['Mese'] = df_venduto['Data'].dt.to_period('M')
df_gruppi = df_venduto.groupby(['Codice Articolo', 'Mese']).agg(
    qty=('Quantità venduta', 'sum'),
    descr=('Descrizione Articolo univoca', 'first')).reset_index()

df_anagrafica = df_gruppi[['Codice Articolo', 'descr']].drop_duplicates('Codice Articolo')
df_anagrafica = df_anagrafica.set_index('Codice Articolo')['descr']

pivot = df_gruppi.pivot_table(index='Codice Articolo', columns='Mese', values='qty',
                              aggfunc='sum', fill_value=0).clip(lower=0)
pivot.columns = [c.to_timestamp() for c in pivot.columns]
pivot = pivot[sorted(pivot.columns)]

st.write('**Pivot venduto (codice × mese):**', pivot.shape[0], 'codici ×', pivot.shape[1], 'mesi',
         f"({pivot.columns[0]:%Y-%m} → {pivot.columns[-1]:%Y-%m})")
with st.expander('Mostra pivot venduto completo'):
    piv_show = pivot.copy()
    piv_show.columns = [f'{c:%Y-%m}' for c in piv_show.columns]
    st.dataframe(piv_show)

with st.expander('Grafico vendite mensili totali'):
    tot = pivot.sum(axis=0).reset_index()
    tot.columns = ['Mese', 'Quantità Totale']
    st.plotly_chart(px.bar(tot, x='Mese', y='Quantità Totale',
                           title='Quantità totale venduta per mese', height=500))

# ---------------------------------------------------------------------
# Parametri classificazione (identici alla v8)
# ---------------------------------------------------------------------
st.subheader('Parametri di Analisi per associazione classe di forecasting ai codici', divider='orange')
col1, col2 = st.columns(2)
with col1:
    st.write("**Parametri temporali:**")
    RECENT_MONTHS = st.number_input("Orizzonte per 'attivo ultimi X mesi'", 1, 24, 6, 1)
    MIN_NONZERO = st.number_input("Soglia minima mesi con valore != 0", 1, 60, 12, 1)
with col2:
    st.write("**Soglie Classificazione Syntetos-Boylan:**")
    ADI_LIMIT = st.number_input("Soglia ADI (standard SB = 1.32; default 1.40)", value=1.40, step=0.01)
    CV_LIMIT = st.number_input("Soglia CV", value=0.80, step=0.01)

if st.button("✅ OK - Conferma Parametri", type="primary"):
    st.session_state.params_confirmed = True
if 'params_confirmed' not in st.session_state or not st.session_state.params_confirmed:
    st.warning("⚠️ Clicca su 'OK - Conferma Parametri' per procedere con l'analisi")
    st.stop()

# ---------------------------------------------------------------------
# Classificazione (engine)
# ---------------------------------------------------------------------
classi = classify(pivot, recent_months=RECENT_MONTHS, min_nonzero=MIN_NONZERO,
                  adi_limit=ADI_LIMIT, cv_limit=CV_LIMIT)
df_classi = classi.reset_index()

st.subheader('Matrice ADI/CV (Syntetos-Boylan Classification)', divider='orange')
color_map = {"Smooth": "green", "Erratic": "orange", "Intermittent": "blue",
             "Lumpy": "red", "New": "purple", "Insufficient Data": "gray"}
fig_matrix = px.scatter(df_classi, x='ADI', y='CV_demand', color='Classe',
                        color_discrete_map=color_map,
                        title='Matrice ADI vs CV (Demand) - Classificazione Domanda',
                        labels={'ADI': 'Average Inter-Demand Interval (ADI)',
                                'CV_demand': 'Coefficient of Variation (Demand Size) %'},
                        hover_data=['Codice Articolo'], height=700)
fig_matrix.add_vline(x=ADI_LIMIT, line_width=2, line_dash="dash", line_color="red")
fig_matrix.add_hline(y=CV_LIMIT * 100, line_width=2, line_dash="dash", line_color="red")
fig_matrix.update_traces(marker=dict(sizemin=5))
st.plotly_chart(fig_matrix)

# Riepilogo per classe con modello assegnato
st.subheader("Riepilogo per Classe → Modello assegnato (backtest Fase 2)")
tot_qty = pivot.sum(axis=1)
riep = df_classi.groupby('Classe').agg(Numero_Codici=('Classe', 'count')).reset_index()
riep['Quantità Totale'] = riep['Classe'].map(
    lambda c: tot_qty[classi['Classe'] == c].sum())
riep['% Quantità'] = (riep['Quantità Totale'] / tot_qty.sum() * 100).round(1)
riep['Modello v9'] = riep['Classe'].map(CHAMPIONS)
st.dataframe(riep, use_container_width=True)

with st.expander('Dettaglio classificazione per codice (ADI, CV, classe)'):
    st.dataframe(df_classi)

# ---------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------
st.subheader('Forecasting segmentato (statsforecast)', divider='orange')

periodi = st.number_input("Orizzonte previsionale [mesi], default = 4", 1, 12, 4, 1)
col_cap1, col_cap2 = st.columns(2)
with col_cap1:
    abilita_cap = st.toggle("Limita i forecast al massimo storico (consigliato)", value=True)
with col_cap2:
    fattore_cap = st.number_input("Fattore tetto (× max storico mensile)", 1.0, 5.0, 1.5, 0.1)

col_b1, col_b2 = st.columns(2)
with col_b1:
    if st.button('✅ Procedi con il forecasting'):
        st.session_state.forecasting_done = True
with col_b2:
    if st.button('🔄 Reset forecasting'):
        st.session_state.forecasting_done = False
        st.rerun()
if not st.session_state.get('forecasting_done'):
    st.stop()


@st.cache_data
def run_forecast(_pivot, _classi, orizzonte, cap, data_hash):
    return forecast_segmentato(_pivot, _classi, orizzonte,
                               cap_factor=cap, progress_cb=None)


with st.spinner('Fit dei modelli in corso (~1-3 min, la parte lunga è AutoARIMA)...'):
    fc, metodo = run_forecast(pivot, classi, periodi,
                              fattore_cap if abilita_cap else None,
                              data_hash=(pivot.shape, str(pivot.columns[-1]), periodi))
st.success(f'Forecasting completato: {len(fc)} codici, orizzonte {periodi} mesi.')

# --- WARNING CODICI NEW (requisito Fase 3) ---
codici_new = classi.index[classi['Classe'].isin(CLASSI_WARNING)].tolist()
if codici_new:
    st.warning(
        f"⚠️ **ATTENZIONE — {len(codici_new)} codici 'New' (prima vendita < 6 mesi)**: "
        f"su questi articoli NESSUN modello di forecasting è affidabile (il backtest mostra "
        f"errori 90-400% per tutti i metodi testati). Il forecast emesso è un rate prudenziale. "
        f"**Per i lanci commercialmente importanti, verificare le quantità con il cliente prima "
        f"dell'approvvigionamento.**"
    )
    df_new = pd.DataFrame({
        'Codice Articolo': codici_new,
        'Descrizione': [df_anagrafica.get(c, '') for c in codici_new],
        'Mesi di storico': classi.loc[codici_new, 'Mesi_nonzero'].values,
    })
    with st.expander(f"Mostra i {len(codici_new)} codici New da verificare"):
        st.dataframe(df_new, use_container_width=True)
    scarica_excel(df_new, 'codici_new_da_verificare.xlsx',
                  label="Download elenco codici New da verificare")

# Tabella forecast con metodo
df_fc_show = fc.copy()
df_fc_show.columns = [f'{c:%Y-%m}' for c in df_fc_show.columns]
df_fc_show.insert(0, 'Metodo', metodo)
df_fc_show.insert(0, 'Descrizione', [df_anagrafica.get(c, '') for c in fc.index])
st.write('**Forecast per codice (tutti i mesi dell\'orizzonte):**')
st.dataframe(df_fc_show)
scarica_excel(df_fc_show.reset_index(), 'forecast_v9_dettaglio.xlsx',
              label="Download forecast dettagliato (con metodo)")

# ---------------------------------------------------------------------
# File per J-Galileo (struttura identica alla v8)
# ---------------------------------------------------------------------
st.subheader('Adattamento del forecasting a J-Galileo', divider='orange')
df_galileo = build_galileo(fc, drop_first_month=True)
st.write('**📦 File finale per J-Galileo (pronto da scaricare):**')
st.dataframe(df_galileo)
st.write('Numero codici articoli:', df_galileo['Articolo'].nunique())
st.caption(f"Nota: il primo mese di forecast ({fc.columns[0]:%Y-%m}) viene scartato "
           f"come in v8 (mese corrente già avviato). Righe con forecast nullo eliminate.")
scarica_excel(df_galileo, 'df_galileo_per_forecasting.xlsx')

# ---------------------------------------------------------------------
# Analisi di dettaglio per codice
# ---------------------------------------------------------------------
st.subheader('Analisi di dettaglio per Codice Articolo', divider='orange')
codice = st.selectbox('Seleziona codice articolo:', options=sorted(fc.index.tolist()))

descr = df_anagrafica.get(codice, 'N/D')
cl_art = classi.loc[codice, 'Classe']
st.markdown(f"**Codice:** `{codice}` | **Descrizione:** *{descr}* | "
            f"**Classe:** `{cl_art}` | **Modello:** `{metodo[codice]}`"
            + (" | ⚠️ **CODICE NEW — VERIFICARE**" if cl_art in CLASSI_WARNING else ""))

hist = pivot.loc[codice]
fut = fc.loc[codice]
df_plot = pd.concat([
    pd.DataFrame({'Data': hist.index, 'Quantità': hist.values, 'Tipo': 'Venduto'}),
    pd.DataFrame({'Data': fut.index, 'Quantità': fut.values, 'Tipo': 'Forecast'}),
])
fig_det = px.line(df_plot, x='Data', y='Quantità', color='Tipo', markers=True,
                  color_discrete_map={'Venduto': '#1f77b4', 'Forecast': '#ff7f0e'},
                  title=f'Storico e forecast — {codice}')
st.plotly_chart(fig_det, use_container_width=True)

# Nota metodologica sull'accuratezza (out-of-sample, non in-sample)
with st.expander("ℹ️ Accuratezza attesa per classe (backtest out-of-sample, 6 cutoff)"):
    st.markdown("""
    | Classe | Modello | WAPE cumulato 3 mesi | Confronto v8 |
    |---|---|---|---|
    | Smooth | AutoETS | 28% | 38% |
    | Erratic | ADIDA | 50% | 70% |
    | Intermittent | CrostonSBA | 41% | 42% |
    | Insufficient Data | AutoARIMA | 67% | 113% |
    | Lumpy | SBA piatto | 86% | 86% |
    | New | SBA piatto | 90% ⚠️ inaffidabile | 90% |

    Fonte: `backtest/REPORT_FASE2.md` — rolling-origin CV su 6 cutoff (set 2025-feb 2026),
    metrica primaria WAPE cumulato sull'orizzonte, coerente con l'uso per approvvigionamento
    materia prima. A differenza della v8, l'errore NON è calcolato in-sample.
    """)

st.stop()
