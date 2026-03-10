# neuralprophet conda

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from neuralprophet import set_log_level
from scipy.stats import linregress
import json
import os
import subprocess
import tempfile
import sys

# Funzioni
#======================================================================
def scarica_excel(df, filename):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1',index=False)
    writer.close()

    st.download_button(
        label="Download Excel workbook",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.ms-excel"
    )


####### Impaginazione

st.set_page_config(layout="wide")
url_immagine = url_immagine = 'guardini.png' # poi mettere in GitHub e linkare da lì
col_1, col_2 = st.columns([2, 3])
with col_1:
    st.image(url_immagine, width=400)
st.divider()
with col_2:
    st.title('Forecasting Materiali v5 | :orange[NeuralProphet]')

# Caricamento dati J-Galileo

@st.cache_data
def load_venduto(file_data):
    """Carica e processa estrazione J-Galileo con caching per migliorare le performance"""
    df = pd.read_excel(file_data, engine='openpyxl')
    df = df[['CDCFST', 'DSCFST', 'CDARST', 'DSARST', 'QTFTST', 'DTBOST', 'DCA1ST', 'DCA2ST', 'DC02ST']]
    
    # Converti tutto a tipi base numpy per evitare problemi con Arrow/Streamlit
    # Colonne stringa
    for col in ['CDCFST', 'DSCFST', 'CDARST', 'DSARST', 'DCA1ST', 'DCA2ST', 'DC02ST']:
        df[col] = df[col].astype(object).fillna('').astype(str)
    
    # Colonne numeriche
    df['QTFTST'] = pd.to_numeric(df['QTFTST'], errors='coerce').fillna(0).astype(np.float64)
    
    # Converti DTBOST da formato YYYYMMDD o YYYYMM a datetime
    # Prima converti in stringa con padding per gestire valori numerici
    df['DTBOST'] = df['DTBOST'].astype(str).str.zfill(8)
    # Estrai anno (primi 4), mese (5-6), giorno (ultimi 2)
    df['year'] = df['DTBOST'].str[:4]
    df['month'] = df['DTBOST'].str[4:6]
    df['day'] = df['DTBOST'].str[6:8]
    # Crea la data combinando anno-mese-giorno
    df['DTBOST'] = pd.to_datetime(df['year'] + '-' + df['month'] + '-' + df['day'], format='%Y-%m-%d', errors='coerce')
    # Rimuovi colonne temporanee
    df.drop(columns=['year', 'month', 'day'], inplace=True)
    
    return df


path_venduto = st.sidebar.file_uploader('Caricare il file venduto da J-Galileo (es. "GUA90DAT.CEFPC00F.xlsx")')
if not path_venduto:
    st.warning('"GUA90DAT.CEFPC00F.xlsx" non caricato, aprire sidebar e caricare file')
    st.stop()
df_venduto = load_venduto(path_venduto)

@st.cache_data
def load_era_diventa(file_data):
    """Carica e processa estrazione J-Galileo con caching per migliorare le performance"""
    df = pd.read_excel(file_data, engine='openpyxl')
    return df

path_era_diventa = st.sidebar.file_uploader('Caricare il file "ERA DIVENTA" db_era_diventa.xlsx)')
if not path_era_diventa:
    st.warning('"ERA DIVENTA" non caricato, aprire sidebar e caricare file')
    st.stop()
df_era_diventa = load_era_diventa(path_era_diventa)


@st.cache_data
def load_promo(file_data):
    """Carica e processa estrazione clienti promozioni"""
    df = pd.read_excel(file_data, engine='openpyxl')
    for col in ['CDCFST', 'DSCFST']:
        df[col] = df[col].astype(object).fillna('').astype(str)
    return df

path_promo = st.sidebar.file_uploader('Caricare il file "Promozioni" db_clienti_promo.xlsx)')
if not path_promo:
    st.warning('"Promozioni" non caricato, aprire sidebar e caricare file')
    st.stop()
df_promo = load_promo(path_promo)


@st.cache_data
def load_articoli_da_escludere(file_data):
    """Carica il file degli articoli da escludere dal forecasting (colonna CDARMA = CDARST nel venduto)."""
    df = pd.read_excel(file_data, engine='openpyxl')
    df['CDARMA'] = df['CDARMA'].astype(object).fillna('').astype(str)
    return df

st.sidebar.divider()
st.sidebar.markdown("**Esclusioni Articoli (Opzionale)**")
path_escludere = st.sidebar.file_uploader(
    'Caricare il file "Articoli da Escludere" (es. "ARTICOLI DA ESCLUDERE.xlsx") — opzionale',
    key='articoli_da_escludere'
)
if path_escludere:
    df_articoli_da_escludere = load_articoli_da_escludere(path_escludere)
else:
    df_articoli_da_escludere = None


# --- FUNZIONI DI SUPPORTO MODELLI ---
@st.cache_resource
def load_optimal_params():
    """Carica i parametri ottimali da JSON gestendo sia il formato nuovo che quello legacy."""
    json_path = 'modelli/optimal_params_np.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Se il formato è legacy (piatto), lo incapsula per coerenza
            if data and 'params' not in data:
                return {
                    'params': data,
                    'best_mae': 'N/A',
                    'last_update': 'Precedente (Formato Legacy)'
                }
            return data
        except Exception as e:
            st.sidebar.error(f"Errore caricamento parametri: {e}")
    return None

@st.cache_resource
def opt_model(df, periodi=12, use_optimized=False):
    set_random_seed(0)
    
    # Carica parametri (JSON completo)
    opt_data = load_optimal_params()
    
    if use_optimized and opt_data:
        # Estrai solo la sezione 'params' per NeuralProphet
        params = opt_data.get('params', {}).copy()
        params['n_forecasts'] = periodi
        m_np = NeuralProphet(**params)
        st.sidebar.success("🚀 Caricato Modello Ottimizzato")
    else:
        # Default "Iron"
        m_np = NeuralProphet(
            trend_global_local='local',
            season_global_local='local',
            seasonality_mode='additive', 
            n_lags=12, n_forecasts=periodi,
            ar_layers=[], 
            learning_rate=0.005,
            n_changepoints=3, 
            trend_reg=1.0,    
            seasonality_reg=0.5,
            epochs=60,
            loss_func='Huber', 
            normalize='standardize'
            )
            
    m_np = m_np.add_seasonality(name="m12", period=12, fourier_order=6)
    m_np.set_plotting_backend('plotly')
    metrica_np = m_np.fit(df, freq='MS')
    return m_np, metrica_np


# Rinomina le colonne

# Rinomina le colonne
df_venduto.rename(columns={
    'CDCFST': 'Codice Cliente',
    'DSCFST': 'Descrizione Cliente',
    'CDARST': 'Codice Articolo',
    'DSARST': 'Descrizione Articolo',
    'QTFTST': 'Quantità venduta',
    'DTBOST': 'Data',
    'DCA1ST': 'Classe di valutazione 1',
    'DCA2ST': 'Classe di valutazione 2',
    'DC02ST': 'Classe di valutazione 3'
}, inplace=True)


with st.expander('Anteprima dati caricati'):
    col_caricato, col_era_diventa, col_promo, col_escludere = st.columns([4, 1, 1, 1])
    with col_caricato:
        st.write('**Dati venduto caricati da J-Galileo:**', df_venduto.shape[0], '**righe**', 'e', df_venduto.shape[1], '**colonne**')
        st.dataframe(df_venduto.head())
        st.write('Codici Articolo univoci:', df_venduto['Codice Articolo'].nunique())
    with col_era_diventa:
        st.write('**Dati "ERA DIVENTA" caricati da J-Galileo:**', df_era_diventa.shape[0], '**righe**', 'e', df_era_diventa.shape[1], '**colonne**')
        st.dataframe(df_era_diventa.head())
    with col_promo:
        st.write('**Dati "Promozioni" caricati da J-Galileo:**', df_promo.shape[0], '**righe**', 'e', df_promo.shape[1], '**colonne**')
        st.dataframe(df_promo.head())
    with col_escludere:
        if df_articoli_da_escludere is not None:
            st.write('**Articoli da Escludere:**', df_articoli_da_escludere.shape[0], '**codici**')
            st.dataframe(df_articoli_da_escludere.head())
        else:
            st.write('**Articoli da Escludere:** non caricato')


# Eliminare clienti promozioni da df_venduto
# elimina righe di df_venduto['Codice Cliente'] che sono presenti in df_promo['CDCFST']
df_venduto = df_venduto[~df_venduto['Codice Cliente'].isin(df_promo['CDCFST'])]

# Esclusione articoli da file "ARTICOLI DA ESCLUDERE" (v4)
# I codici CDARMA del file corrispondono a CDARST (rinominato 'Codice Articolo') nel file venduto
if df_articoli_da_escludere is not None:
    codici_da_escludere = set(df_articoli_da_escludere['CDARMA'].tolist())
    n_codici_prima = df_venduto['Codice Articolo'].nunique()
    n_righe_prima = len(df_venduto)
    df_venduto = df_venduto[~df_venduto['Codice Articolo'].isin(codici_da_escludere)]
    n_codici_dopo = df_venduto['Codice Articolo'].nunique()
    n_righe_dopo = len(df_venduto)
    n_codici_esclusi = n_codici_prima - n_codici_dopo
    n_righe_escluse = n_righe_prima - n_righe_dopo
    st.info(
        f'🚫 **Articoli da Escludere** applicato: rimossi **{n_codici_esclusi} codici articolo** '
        f'({n_righe_escluse} righe) su {n_codici_prima} totali. '
        f'Codici rimanenti: **{n_codici_dopo}**.'
    )

# Filtro interattivo per 'Classe di valutazione 1'
classi_disponibili = sorted(df_venduto['Classe di valutazione 1'].dropna().unique().tolist())
default_class = 'PF interno+confez.'
default_selection = [default_class] if default_class in classi_disponibili else None

classi_selezionate = st.pills(
    "Filtro Classe di valutazione 1:",
    options=classi_disponibili,
    default=default_selection,
    selection_mode="multi"
)

if not classi_selezionate:
    st.warning("Seleziona almeno una Classe di valutazione 1 per proseguire l'elaborazione.")
    st.stop()

df_venduto = df_venduto[df_venduto['Classe di valutazione 1'].isin(classi_selezionate)]

# OPZIONE 1: sostituzione Codice Articolo da era-diventa (per continuità storica)

# Creazione dataframe com contunuità da era-diventa
df_con_sostituzione_era_diventa = df_venduto.copy()

# Se trova Codice Articolo di df_con_sostituzione_era_diventa che è presente in df_era_diventa nella colonna "Codice Articolo", sostituisce il Codice Articolo di df_con_sostituzione_era_diventa con il valore corrispondente nella colonna "Diventa" di df_era_diventa
df_con_sostituzione_era_diventa = df_con_sostituzione_era_diventa.merge(df_era_diventa, left_on='Codice Articolo', right_on='Era', how='left')
df_con_sostituzione_era_diventa['Codice Articolo'] = df_con_sostituzione_era_diventa.apply(lambda row: row['Diventa'] if pd.notna(row['Diventa']) else row['Codice Articolo'], axis=1)
df_con_sostituzione_era_diventa.drop(columns=['Diventa','Era'], inplace=True)

#st.write('df_venduto con sostituzione Codice Articolo da ERA DIVENTA (per continuità storica):', df_con_sostituzione_era_diventa.shape[0], '**righe**', 'e', df_con_sostituzione_era_diventa.shape[1], '**colonne**')

df_con_sostituzione_era_diventa['Obsoleto'] = np.where(df_con_sostituzione_era_diventa['Descrizione Articolo'].str.startswith('***'), True, False)
df_con_sostituzione_era_diventa['Eliminare']=df_con_sostituzione_era_diventa.groupby('Codice Articolo')['Obsoleto'].transform(lambda x: x.all())

# Funzione per aggiungere colonna "Descrizione Articolo univoca" con le seguenti regole:
# - se i primi tre digit di Descrizione Articolo sono "***" oppure "---" allora elimina i primi tre digit e mantieni il resto della stringa
# - altrimenti mantieni la Descrizione Articolo originale
def estrai_descrizione_univoca(descrizione):
    if descrizione.startswith('***') or descrizione.startswith('---'):
        return descrizione[3:]
    return descrizione

df_con_sostituzione_era_diventa['Descrizione Articolo univoca'] = df_con_sostituzione_era_diventa['Descrizione Articolo'].apply(estrai_descrizione_univoca)

n_obsoleti_era = df_con_sostituzione_era_diventa[df_con_sostituzione_era_diventa['Eliminare'] == True]['Codice Articolo'].nunique()


# OPZIONE 2: sostituzione Codice Articolo a 5 digit (per aggregazione a livello di famiglia)

df_con_sostituzione_5_digit = df_venduto.copy()
df_con_sostituzione_5_digit['Codice Articolo 5 digit'] = df_con_sostituzione_5_digit['Codice Articolo'].str[:5]


df_con_sostituzione_5_digit['Descrizione Articolo univoca'] = df_con_sostituzione_5_digit['Descrizione Articolo'].apply(estrai_descrizione_univoca)
df_con_sostituzione_5_digit['Obsoleto'] = np.where(df_con_sostituzione_5_digit['Descrizione Articolo'].str.startswith('***'), True, False)

# Marca come "Eliminare" le righe per cui TUTTE le righe di un Codice Articolo 5 digit hanno Obsoleto == True
# Raggruppa per Codice Articolo 5 digit e verifica se tutte le righe hanno Obsoleto == True
df_con_sostituzione_5_digit['Eliminare'] = df_con_sostituzione_5_digit.groupby('Codice Articolo 5 digit')['Obsoleto'].transform(lambda x: x.all())
df_con_sostituzione_5_digit['Codice Articolo Originale'] = df_con_sostituzione_5_digit['Codice Articolo']
df_con_sostituzione_5_digit['Codice Articolo'] = df_con_sostituzione_5_digit['Codice Articolo 5 digit']
df_con_sostituzione_5_digit.drop(columns=['Codice Articolo 5 digit'], inplace=True)

n_obsoleti_5d = df_con_sostituzione_5_digit[df_con_sostituzione_5_digit['Eliminare'] == True]['Codice Articolo'].nunique()

# Inserisci toggle per scelta dataframe da utilizzare per i passaggi successivi (filtraggio, pivot, analisi)
usa_era_diventa = st.toggle("Usa 'Era Diventa' (default)", value=True)

if usa_era_diventa:
    df_venduto = df_con_sostituzione_era_diventa.copy()
else:
    st.info("ℹ️ Modalità attiva: I codici articolo sono raggruppati per i primi 5 digit.")
    df_venduto = df_con_sostituzione_5_digit.copy()

df_venduto = df_venduto[df_venduto['Eliminare'] == False]


# Liste eliminazione
lista_codice_articolo_da_escludere = ['SK','-','*','.','.SCAR FERROSO','_','ST'] # aggiunto ST
#lista_descrizione_articolo_da_escludere = ['***']

# filtra df_venduto escludendo le righe Codice Articolo che iniziano con qualunque elemento di lista_codice_articolo_da_escludere 
# e filtra df_venduto escludendo le righe di Descrizione Articolo che iniziano con qualunque elemento di lista_descrizione_articolo_da_escludere

df_venduto_filtrato = df_venduto[~df_venduto['Codice Articolo'].str.startswith(tuple(lista_codice_articolo_da_escludere))]
#df_venduto_filtrato = df_venduto_filtrato[~df_venduto_filtrato['Descrizione Articolo'].str.startswith(tuple(lista_descrizione_articolo_da_escludere))]

lista_descrizione_articolo_da_escludere = ['SET','EXPO']
# elimina righe di Descrizione Articolo che contengono qualunque elemento di lista_descrizione_articolo_da_escludere
df_venduto_filtrato = df_venduto_filtrato[~df_venduto_filtrato['Descrizione Articolo'].str.contains('|'.join(lista_descrizione_articolo_da_escludere))] 

# Filtra per Data >= 01/01/2023
df_venduto_filtrato = df_venduto_filtrato[df_venduto_filtrato['Data'] >= pd.to_datetime('2023-01-01')]

df_venduto_filtrato.reset_index(drop=True, inplace=True)

# Seleziona colonnne rilevanti e raggruppa per codice e mese
colonne_rilevanti = ['Codice Articolo', 'Descrizione Articolo univoca', 'Quantità venduta', 'Data']
df_venduto_filtrato = df_venduto_filtrato[colonne_rilevanti].copy()

# raggruppa per codice articolo, descrizione articolo univoca e anno-mese (estratto da Data) sommando la quantità venduta
df_venduto_filtrato['Data'] = df_venduto_filtrato['Data'].dt.to_period('M')
df_venduto_filtrato = df_venduto_filtrato.groupby(['Codice Articolo', 'Data']).agg({
    'Quantità venduta': 'sum',
    'Descrizione Articolo univoca': 'first'
}).reset_index()

st.info('Eliminati articoli riferiti a promozioni, codici SET e EXPO, codici SK, -, *, ., .SCAR FERROSO, _, ST')

# ---------------------------------------------------------------------------
# Creazione Anagrafica Articoli (per descrizioni)
# ---------------------------------------------------------------------------
df_anagrafica = df_venduto_filtrato[['Codice Articolo', 'Descrizione Articolo univoca']].drop_duplicates(subset='Codice Articolo')
st.caption(f"Anagrafica Articoli creata: {len(df_anagrafica)} codici univoci")


# Crea pivot table per visualizzare venduto per mese e anno, con indici Codice Articolo e Descrizione Articolo
df_venduto_pivot = pd.pivot_table(
    df_venduto_filtrato,
    values='Quantità venduta',
    index=['Codice Articolo', ],#'Descrizione Articolo univoca'
    columns=df_venduto_filtrato['Data'],#.dt.to_period('M'),
    aggfunc='sum',
    fill_value=0
    ).clip(lower=0)  # Assicura che non ci siano valori negativi

# Converti le colonne Period in stringhe per evitare problemi di serializzazione
df_venduto_pivot.columns = [col.strftime('%Y-%m') if isinstance(col, pd.Period) else col for col in df_venduto_pivot.columns]

# Unisci gli indici in una singola colonna "Codice_Descrizione" e rimuovi i livelli di indice
df_venduto_pivot.reset_index(inplace=True)
#df_venduto_pivot['Codice_Descrizione'] = df_venduto_pivot['Codice Articolo'] + ' | ' + df_venduto_pivot['Descrizione Articolo univoca']
#df_venduto_pivot.drop(columns=['Codice Articolo', 'Descrizione Articolo univoca'], inplace=True)
df_venduto_pivot.set_index('Codice Articolo', inplace=True)

st.write('**Pivot table venduto per mese e anno:**', df_venduto_pivot.shape[0], '**righe**', 'e', df_venduto_pivot.shape[1], '**colonne**')

df_venduto_pivot

# Grafico per visualizzare la quantità venduta complessiva per mese (somma su tutte le colonne) con plotly
# Calcola il totale per ogni mese (somma su tutte le righe)
df_totale_mensile = df_venduto_pivot.sum(axis=0).reset_index()
df_totale_mensile.columns = ['Mese', 'Quantità Totale']

fig_vendite_mensili = px.bar(
    df_totale_mensile,
    x='Mese',
    y='Quantità Totale',
    title='Quantità totale venduta per mese',
    labels={'Mese': 'Mese', 'Quantità Totale': 'Quantità venduta'},
    height=500
)

with st.expander('Visualizza grafico vendite mensili (Codici e Descrizioni non filtrati)'):
    st.plotly_chart(fig_vendite_mensili)


# Analisi per associazione classe di forecasting ai materiali ========================================================


df_pivot_analisi = df_venduto_pivot.copy()

# Converti le colonne che sono stringhe in formato 'YYYY-MM' in datetime
rename_dict = {}
for col in df_pivot_analisi.columns:
    if col not in ['Codice Articolo', 'Codice_Descrizione']:
        try:
            # Se la colonna è una stringa in formato 'YYYY-MM', convertila in datetime
            if isinstance(col, str) and '-' in str(col) and len(str(col)) == 7:
                rename_dict[col] = pd.to_datetime(col + '-01')
        except:
            pass

if rename_dict:
    df_pivot_analisi.rename(columns=rename_dict, inplace=True)

# individua le colonne con data in df_pivot_analisi (ora sono datetime)
colonne_date = [col for col in df_pivot_analisi.columns if isinstance(col, pd.Timestamp)]


# in df_pivot_analisi voglio la colonna totale che è la somma di tutte le colonne tipo data
df_pivot_analisi['Totale'] = df_pivot_analisi[colonne_date].sum(axis=1)

# in df_pivot_analisi voglio la colonna CV che è la deviazione standard delle colonne con data divisa per la media
df_pivot_analisi['CV'] = df_pivot_analisi[colonne_date].std(axis=1) / df_pivot_analisi[colonne_date].mean(axis=1) * 100

# ordina df_pivot_analisi per colonna Totale decrescente
df_pivot_analisi = df_pivot_analisi.sort_values(by='Totale', ascending=False)

# Resetta l'indice per renderlo disponibile come colonna nell'hover
df_pivot_analisi = df_pivot_analisi.reset_index()


###### Completamento analisi 

st.subheader('Parametri di Analisi per associazione classe di forecasting ai codici', divider='orange')

# Crea due colonne per organizzare meglio i parametri
col1, col2 = st.columns(2)

with col1:
    st.write("**Parametri temporali:**")
    RECENT_MONTHS = st.number_input(
        "Orizzonte per 'attivo ultimi X mesi'",
        min_value=1,
        max_value=24,
        value=6,
        step=1,
        help="Numero di mesi recenti da considerare per l'analisi dell'attività"
    )
    
    MIN_NONZERO = st.number_input(
        "Soglia minima mesi con valore != 0",
        min_value=1,
        max_value=60,
        value=12,
        step=1,
        help="Numero minimo di mesi con valori non nulli per considerare una serie valida"
    )

with col2:
    st.write("**Parametri di qualità:**")
    MAX_ZERO_SEQ_ALLOWED = st.number_input(
        "Soglia massima zeri consecutivi tollerabile",
        min_value=1,
        max_value=24,
        value=6,
        step=1,
        help="Numero massimo di mesi consecutivi con valore zero considerato accettabile"
    )
    
    # Range per CV
    st.write("**Range CV (Coefficient of Variation):**")
    cv_col1, cv_col2 = st.columns(2)
    with cv_col1:
        CV_MIN = st.number_input(
            "CV Minimo (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            help="Valore minimo per il Coefficient of Variation"
        )
    with cv_col2:
        CV_MAX = st.number_input(
            "CV Massimo (%)",
            min_value=50,
            max_value=700,
            value=120,
            step=10,
            help="Valore massimo per il Coefficient of Variation"
        )

    st.write("**Soglie Classificazione Syntetos-Boylan:**")
    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        ADI_LIMIT = st.number_input("Soglia ADI", value=4.00, step=0.01)
    with sb_col2:
        CV_LIMIT = st.number_input("Soglia CV", value=0.80, step=0.01)



# Pulsanti per confermare i parametri e reset
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("✅ OK - Conferma Parametri", type="primary"):
        st.session_state.params_confirmed = True
        st.session_state.params = {
            "RECENT_MONTHS": RECENT_MONTHS,
            "MIN_NONZERO": MIN_NONZERO,
            "MAX_ZERO_SEQ_ALLOWED": MAX_ZERO_SEQ_ALLOWED,
            "CV_MIN": CV_MIN,
            "CV_MAX": CV_MAX,
            "ADI_LIMIT": ADI_LIMIT,
            "CV_LIMIT": CV_LIMIT
        }
        st.success("✅ Parametri confermati! L'analisi può procedere.")

with col_btn2:
    if st.button("🔄 Reset Parametri"):
        # Rimuove la conferma dei parametri
        if 'params_confirmed' in st.session_state:
            del st.session_state.params_confirmed
        if 'params' in st.session_state:
            del st.session_state.params
        st.info("🔄 Parametri ripristinati ai valori di default. Ricarica la pagina per vedere i valori predefiniti.")
        st.rerun()

# Mostra un riepilogo dei parametri selezionati
msg = """
**Parametri selezionati:**
- Mesi recenti da analizzare: {}
- Minimo mesi non nulli richiesti: {}
- Massimo zeri consecutivi tollerati: {}
- Range CV accettabile: {}% - {}%
- Soglia ADI: {}
- Soglia CV (Matrix): {}
""".format(RECENT_MONTHS, MIN_NONZERO, MAX_ZERO_SEQ_ALLOWED, CV_MIN, CV_MAX, ADI_LIMIT, CV_LIMIT)
st.info(msg)

# Controlla se i parametri sono stati confermati
if 'params_confirmed' not in st.session_state or not st.session_state.params_confirmed:
    st.warning("⚠️ Clicca su 'OK - Conferma Parametri' per procedere con l'analisi")
    st.stop()

# 1) Individua le colonne data (ora sono di tipo Timestamp)
# =======================================================
non_date_cols = {"Codice Articolo", "Totale", "CV", "Codice_Descrizione"}
# Identifica le colonne datetime/Timestamp
date_cols = [c for c in df_pivot_analisi.columns if (isinstance(c, pd.Timestamp) and c not in non_date_cols)]
date_cols = sorted(date_cols)  # ordine cronologico garantito

if len(date_cols) == 0:
    st.error("❌ Nessuna colonna data trovata nel dataframe. Verifica la struttura dei dati.")
    st.write("Colonne presenti:", df_pivot_analisi.columns.tolist())
    st.write("Tipi di dato:", df_pivot_analisi.dtypes)
    st.stop()


values = df_pivot_analisi[date_cols].copy()

# 2) Funzioni metriche
# =======================================================
def max_seq_zeros_row(row_vals: pd.Series) -> int:
    """Massima sequenza consecutiva di zeri in una riga."""
    max_len = curr_len = 0
    for v in row_vals:
        if v == 0:
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 0
    return max_len

def first_last_nonzero_idx(row_vals: pd.Series):
    """Indice del primo/ultimo mese con valore != 0. Restituisce (first_idx, last_idx) o (np.nan, np.nan)."""
    nz = np.where(row_vals.values != 0)[0]
    if len(nz) == 0:
        return np.nan, np.nan
    return nz[0], nz[-1]

# 3) Metriche descrittive
# =======================================================
mesi_con_valore = (values != 0).sum(axis=1)
attivo_recent   = (values.iloc[:, -RECENT_MONTHS:] != 0).any(axis=1)
max_zero_seq    = values.apply(max_seq_zeros_row, axis=1)
prop_zeri       = (values == 0).mean(axis=1)

results = values.apply(first_last_nonzero_idx, axis=1)
first_idx = pd.Series([r[0] for r in results], index=values.index)
last_idx  = pd.Series([r[1] for r in results], index=values.index)

def idx_to_date(idx_series: pd.Series, date_cols_sorted: list):
    out = []
    for i in idx_series:
        if pd.isna(i):
            out.append(np.nan)
        else:
            out.append(date_cols_sorted[int(i)])
    return pd.Series(out, index=idx_series.index, dtype="datetime64[ns]")

first_date   = idx_to_date(first_idx, date_cols)
last_date    = idx_to_date(last_idx,  date_cols)
span_nonzero = (last_idx - first_idx + 1).where(~first_idx.isna(), np.nan)



# 4) Etichetta descrittiva e flag "Forecastabile (rule)"
#    NB: usa la colonna 'CV_demand' calcolata solo sui valori != 0
# =======================================================

# Calcolo ADI = (Last Index - First Index) / (Count - 1)
# Se Count <= 1, ADI non definito (o infinito), mettiamo un valore alto
# Nota: La formula classica usa Total Periods / Non-Zero Demand.
# Qui usiamo lo "span di vita attivo" per non penalizzare i nuovi prodotti.
# span_nonzero e mesi_con_valore sono Series con stesso indice.

span_nonzero_filled = span_nonzero.fillna(0)
mesi_con_valore_filled = mesi_con_valore.fillna(0)

def calculate_adi(span, count):
    if count <= 1:
        return np.nan # o un valore molto alto
    if span <= 1: 
       return 0 # Caso limite
    return span / count

adi_values = [calculate_adi(s, c) for s, c in zip(span_nonzero_filled, mesi_con_valore_filled)]
adi_series = pd.Series(adi_values, index=values.index)

# Calcolo CV (Demand Sizes)
# CV = std(non_zeros) / mean(non_zeros)
def calculate_cv_demand(row_vals):
    non_zeros = row_vals[row_vals != 0]
    if len(non_zeros) < 2:
        return np.nan
    return (non_zeros.std() / non_zeros.mean()) * 100

cv_demand_series = values.apply(calculate_cv_demand, axis=1)

# Calcolo Mean Demand (Average of Non-Zero values)
def calculate_mean_demand(row_vals):
    non_zeros = row_vals[row_vals != 0]
    if len(non_zeros) == 0:
        return 0.0
    return non_zeros.mean()

mean_demand_series = values.apply(calculate_mean_demand, axis=1)

# Calcolo Trend (Slope)
# Usa linregress sui valori non-zero? O su tutti i valori nell'intervallo attivo?
# Standard approach: Trend on non-zero demand sizes.
def calculate_trend_slope(row_vals):
    non_zeros = row_vals[row_vals != 0]
    if len(non_zeros) < 3: # Need at least 3 points for trend
        return 0.0
    # X = range(len), Y = values
    slope, _, _, p_val, _ = linregress(range(len(non_zeros)), non_zeros)
    return slope if p_val < 0.05 else 0.0 # Return slope only if significant

trend_series = values.apply(calculate_trend_slope, axis=1)

# Calcolo Stagionalità (Autocorrelation lag 12)
# Richiede serie lunga e piena (con zeri).
def check_seasonality(row_vals):
    # Fill NaN with 0 if any (shouldn't be in `values`)
    ts = row_vals.fillna(0)
    if len(ts) < 24: # Need at least 2 full cycles
        return False
    # Simple autocorrelation at lag 12
    # We can also use ACF check
    acf_12 = ts.autocorr(lag=12)
    return acf_12 > 0.4 # Threshold typically 0.2-0.5

seasonal_series = values.apply(check_seasonality, axis=1)


def classify_matrix(adi, cv, active, months_count, first_sale_date, current_date_limit):
    # Gestione "Single Mode" (1 solo valore)
    if months_count == 1:
        # Se il primo (e unico) vendita è recente
        if pd.notna(first_sale_date) and first_sale_date > (current_date_limit - pd.DateOffset(months=6)):
             return "New"
        else:
             return "Lumpy" # Vecchio e solo una vendita = Sporadico estremo / Obsoleto
             
    # Se ha troppi pochi dati, è "New" o "Insufficient Data"
    if months_count < 6 and active: # Meno di 6 mesi di storico ma attivo
        # Check se è "New" basato su data prima vendita
        if pd.notna(first_sale_date) and first_sale_date > (current_date_limit - pd.DateOffset(months=6)):
             return "New"
        
    if months_count < MIN_NONZERO: # Troppo pochi dati non-zero (e non è New)
        return "Insufficient Data"
    
    # Classificazione ADI/CV
    if pd.isna(adi) or pd.isna(cv):
        return "Insufficient Data"
        
    cv_ratio = cv / 100.0 # CV è in percentuale nel dataframe (0-100+)
    
    if adi < ADI_LIMIT: # Frequent
        if cv_ratio < CV_LIMIT:
            return "Smooth"
        else:
            return "Erratic"
    else: # Intermittent
        if cv_ratio < CV_LIMIT:
            return "Intermittent"
        else:
            return "Lumpy"

# Data reference per "New" item identification
current_ref_date = date_cols[-1] # Ultima data disponibile nel dataset

forecastability_class = [
    classify_matrix(adi, cv, act, m, f_date, current_ref_date)
    for adi, cv, act, m, f_date in zip(adi_series, cv_demand_series, attivo_recent, mesi_con_valore, first_date)
]

# Forecastabile se è Smooth o Erratic (con warning) e Attivo
forecastable_rule = (
    (attivo_recent) &
    (pd.Series(forecastability_class).isin(["Smooth", "Erratic"])) &
    (cv_demand_series <= CV_MAX) # Filtro extra utente
)

# 5) Scrittura colonne nel dataframe
# =======================================================
df_pivot_analisi = df_pivot_analisi.copy()
df_pivot_analisi["Mesi con valore ≠ 0"]             = mesi_con_valore
df_pivot_analisi[f"Attivo ultimi {RECENT_MONTHS}m"]   = attivo_recent
df_pivot_analisi["Max zeri consecutivi"]            = max_zero_seq
df_pivot_analisi["Proporzione zeri"]                = prop_zeri
df_pivot_analisi["Primo mese != 0"]                 = first_date
df_pivot_analisi["Ultimo mese != 0"]                = last_date
df_pivot_analisi["Span mesi non-zero"]              = span_nonzero
df_pivot_analisi["ADI"]                             = adi_series
df_pivot_analisi["CV (Demand)"] = cv_demand_series.round(1)
df_pivot_analisi["Mean Demand"] = mean_demand_series.round(2)
df_pivot_analisi["Trend Slope"] = trend_series.round(4)
df_pivot_analisi["Seasonality"]                     = seasonal_series
df_pivot_analisi["Forecastability (descr.)"]        = forecastability_class
df_pivot_analisi["Forecastabile (rule)"]            = forecastable_rule




# 6) Colonna "Motivo esclusione principale"
# =======================================================

def motivo_esclusione(class_descr, cv, adi):
    """Restituisce la causa principale per cui la serie non è forecastabile."""
    if class_descr == "Insufficient Data":
        return "Dati Insufficienti (< soglia)"
    if class_descr == "New":
        return "Articolo Nuovo (< 6 mesi)"
    if class_descr == "Lumpy":
        return f"Lumpy (ADI={adi:.2f}, CV={cv:.2f})"
    if class_descr == "Intermittent":
        return f"Intermittent (ADI={adi:.2f})"
    if cv > CV_MAX:
         return f"CV (Demand) > Max ({cv:.2f} > {CV_MAX})"
    return ""

# Applica la funzione riga per riga
motivo_esclusione_principale = [
    motivo_esclusione(c_descr, cv, adi)
    for c_descr, cv, adi in zip(
        df_pivot_analisi["Forecastability (descr.)"],
        df_pivot_analisi["CV (Demand)"],
        df_pivot_analisi["ADI"],
    )
]

# Aggiungi la colonna al dataframe
df_pivot_analisi["Motivo esclusione principale"] = motivo_esclusione_principale

st.write('df_pivot_analisi con parametri di forecastability:')
st.dataframe(df_pivot_analisi)

# Inserisci pulsante per scaricare df_pivot_analisi in Excel
btn_scarica_pivot = st.button("Scarica df_pivot_analisi in Excel")
if btn_scarica_pivot:
    df_pivot_analisi.to_excel("df_pivot_analisi.xlsx")
    st.success("df_pivot_analisi scaricato con successo!")


# con plotly crea un grafico scatter Matrix ADI / CV
st.subheader('Matrice ADI/CV (Syntetos-Boylan Classification)', divider='orange')

# Define consistent color map
color_map = {
    "Smooth": "green",
    "Erratic": "orange",
    "Intermittent": "blue",
    "Lumpy": "red",
    "New": "purple",
    "Insufficient Data": "gray"
}

fig_matrix = px.scatter(
    df_pivot_analisi,
    x='ADI',
    y='CV (Demand)',
    color='Forecastability (descr.)',
    color_discrete_map=color_map, # Enforce consistent colors
    title='Matrice ADI vs CV (Demand) - Classificazione Domanda',
    labels={'ADI': 'Average Inter-Demand Interval (ADI)', 'CV (Demand)': 'Coefficient of Variation (Demand Size) %'},
    hover_data=['Codice Articolo', 'Trend Slope', 'Seasonality'],#, 'Descrizione Articolo', 'Forecastability (descr.)', 'Totale'],
    height=700,
    #size='Valore totale',
    size_max=40
)

# Aggiungi linee soglia
fig_matrix.add_vline(x=ADI_LIMIT, line_width=2, line_dash="dash", line_color="red", annotation_text=f"ADI={ADI_LIMIT}")
fig_matrix.add_hline(y=CV_LIMIT*100, line_width=2, line_dash="dash", line_color="red", annotation_text=f"CV={CV_LIMIT*100}%")

# Imposta dimensione minima dei punti
fig_matrix.update_traces(marker=dict(sizemin=5))
st.plotly_chart(fig_matrix)

with st.expander("ℹ️ Legenda Classi di Forecastability"):
    st.markdown("""
    **Interpretazione delle Classi (Syntetos-Boylan):**
    
    *   **:green[Smooth] (Regolare)**: La domanda è frequente e poco variabile. È la condizione ideale per previsioni accurate.
    *   **:orange[Erratic] (Irregolare)**: La domanda è frequente ma le quantità variano molto. Difficile da prevedere con precisione.
    *   **:blue[Intermittent] (Intermittente)**: La domanda è poco frequente (avviene ogni tanto) ma quando c'è, le quantità sono costanti.
    *   **:red[Lumpy] (Sporadica/Grandi Ordini)**: La domanda è rara e le quantità sono molto variabili. Molto difficile da gestire, spesso richiede scorte di sicurezza elevate o gestione a commessa.
    *   **:violet[New] (Nuovi Articoli)**: Articoli con prima vendita negli ultimi 6 mesi. Storico insufficiente per classificazioni statistiche.
    *   **:grey[Insufficient Data]**: Dati storici insufficienti o serie non attive per permettere una classificazione affidabile.
    """)

# Tabella riepilogativa per classe di forecastability
st.subheader("Riepilogo per Classe di Forecastability")

# Verifica se esiste la colonna Totale, altrimenti calcolala
if 'Totale' not in df_pivot_analisi.columns:
    df_pivot_analisi['Totale'] = df_pivot_analisi["Mesi con valore ≠ 0"] * df_pivot_analisi["ADI"] # Dummy fallback se non c'è, ma dovrebbe esserci da step precedenti
    # Meglio ricalcolarla correttamente dai dati originali se possibile, ma usiamo quello che c'è
    # Nello step 5 c'era: df_pivot_analisi['Totale'] = df_pivot_analisi[colonne_date].sum(axis=1)
    # Verifichiamo se colonne_date è disponibile qui. Sì, definito a global scope o quasi.
    # Ma df_pivot_analisi è una copia.
    # Assumiamo di usare 'values' definita prima per ricalcolare se manca.
    pass

# Recalculate Totale to be sure (uses values dataframe defined earlier)
df_pivot_analisi['Totale_Qty'] = values.sum(axis=1)

summary_stats = df_pivot_analisi.groupby('Forecastability (descr.)').agg(
    Numero_Codici=('Forecastability (descr.)', 'count'),
    Quantita_Totale=('Totale_Qty', 'sum')
).reset_index()

total_codes = summary_stats['Numero_Codici'].sum()
total_qty = summary_stats['Quantita_Totale'].sum()

summary_stats['% Codici'] = (summary_stats['Numero_Codici'] / total_codes * 100).map('{:.1f}%'.format)
summary_stats['% Quantità'] = (summary_stats['Quantita_Totale'] / total_qty * 100).map('{:.1f}%'.format)

# Reorder columns
summary_stats = summary_stats[['Forecastability (descr.)', 'Numero_Codici', '% Codici', 'Quantita_Totale', '% Quantità']]
st.dataframe(summary_stats, use_container_width=True)

# Create two horizontal bar charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig_bar_count = px.bar(
        summary_stats,
        x='Numero_Codici',
        y='Forecastability (descr.)',
        color='Forecastability (descr.)',
        color_discrete_map=color_map,
        orientation='h',
        title='Numero di Codici per Classe',
        text='% Codici'
    )
    fig_bar_count.update_layout(showlegend=False)
    st.plotly_chart(fig_bar_count, use_container_width=True)

with col_chart2:
    fig_bar_qty = px.bar(
        summary_stats,
        x='Quantita_Totale',
        y='Forecastability (descr.)',
        color='Forecastability (descr.)',
        color_discrete_map=color_map,
        orientation='h',
        title='Quantità Totale per Classe',
        text='% Quantità'
    )
    fig_bar_qty.update_layout(showlegend=False)
    st.plotly_chart(fig_bar_qty, use_container_width=True)


# Forecasting con NeuralProphet
# ===========================================================

st.subheader('Forecasting con NeuralProphet / SBA', divider='orange')

# Selezione Classi da Prevedere
classi_disponibili = sorted(df_pivot_analisi['Forecastability (descr.)'].unique())
classi_selezionate = st.multiselect(
    "Seleziona le classi di forecastability da includere nell'analisi:",
    options=classi_disponibili,
    default=classi_disponibili,
    help="Scegli per quali categorie calcolare le previsioni."
)

if not classi_selezionate:
    st.warning("Seleziona almeno una classe per procedere.")
    st.stop()

# Filtra il dataset principale
df_forecast_input = df_pivot_analisi[df_pivot_analisi['Forecastability (descr.)'].isin(classi_selezionate)].copy()

st.write(f"Articoli selezionati per il forecasting: **{len(df_forecast_input)}**")

# Logica di Segmentazione: NeuralProphet vs Statistico
# NeuralProphet: Smooth, Erratic
# Statistico (Media Mobile): Intermittent, Lumpy, New, Insufficient Data

classi_np = ['Smooth', 'Erratic']
classi_stat = [c for c in classi_disponibili if c not in classi_np]

df_np_input = df_forecast_input[df_forecast_input['Forecastability (descr.)'].isin(classi_np)].copy()
df_stat_input = df_forecast_input[~df_forecast_input['Forecastability (descr.)'].isin(classi_np)].copy()

col_np, col_stat = st.columns(2)

# --- NEURAL PROPHET PREPARATION ---
with col_np:
    st.markdown("### 🧠 NeuralProphet (Smooth/Erratic)")
    st.caption(f"Articoli: {len(df_np_input)}")
    
    if not df_np_input.empty:
        # Preparazione formato ID, ds, y
        # Recupera la serie storica originale (df_venduto_pivot) filtrata
        codici_np = df_np_input['Codice Articolo'].tolist()
        df_np_series = df_venduto_pivot.loc[codici_np].reset_index()
        
        # Melt per avere formato long (ID, ds, y)
        df_np_long = df_np_series.melt(id_vars='Codice Articolo', var_name='ds', value_name='y')
        df_np_long.rename(columns={'Codice Articolo': 'ID'}, inplace=True)
        df_np_long['ds'] = pd.to_datetime(df_np_long['ds'])
        df_np_long = df_np_long.sort_values(['ID', 'ds'])
        
        st.dataframe(df_np_long.head(), use_container_width=True)
        st.success(f"Dati pronti per NeuralProphet: {len(df_np_long)} righe")

        # --- SEZIONE MANUTENZIONE (SIDEBAR) ---
        # Posizionata qui perché usa i dati già filtrati (Smooth/Erratic)
        st.sidebar.divider()
        st.sidebar.subheader("🛠️ Manutenzione Modelli")
        st.sidebar.caption("L'ottimizzazione analizza i top articoli Smooth/Erratic per trovare i parametri migliori.")

        if st.sidebar.button("🎯 Ottimizza NeuralProphet (Optuna)"):
            try:
                # Prepara campione: top 30 articoli Smooth/Erratic che abbiano almeno 12 mesi di storico
                # df_np_input contiene già solo Smooth/Erratic
                df_valid_history = df_np_input[df_np_input['Mesi con valore ≠ 0'] >= 12]
                
                if df_valid_history.empty:
                    st.sidebar.error("❌ Nessun articolo Smooth/Erratic ha almeno 12 mesi di storico. Impossibile ottimizzare.")
                else:
                    top_30_ids = df_valid_history.nlargest(30, 'Mean Demand')['Codice Articolo'].tolist()
                    df_sample = df_np_long[df_np_long['ID'].isin(top_30_ids)].copy()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    df_sample.to_excel(tmp.name, index=False)
                    tmp_path = tmp.name
                
                with st.sidebar.status("Ottimizzazione in corso...", expanded=True) as status:
                    st.write(f"Ricerca parametri ottimali su {len(top_30_ids)} articoli...")
                    
                    result = subprocess.run([sys.executable, "optimize_np.py", tmp_path], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        status.update(label="✅ Ottimizzazione Completata!", state="complete", expanded=False)
                    else:
                        status.update(label="❌ Ottimizzazione Fallita", state="error")
                
                # Mostra il log FUORI dallo status per evitare nested expanders
                st.sidebar.divider()
                st.sidebar.write("### 📋 Log Dettagliato")
                log_height = 300 if result.returncode != 0 else 150
                st.sidebar.text_area("Output Script", value=result.stdout + result.stderr, height=log_height)
                
                if result.returncode == 0:
                     st.sidebar.success("Parametri salvati. Cliccare su 'Reset forecasting' o ricaricare per applicare.")
                     st.cache_resource.clear()
                else:
                     st.sidebar.error("Errore durante l'ottimizzazione.")
                
                os.unlink(tmp_path)
            except Exception as e:
                st.sidebar.error(f"Errore tecnico: {e}")
    else:
        st.info("Nessun articolo per NeuralProphet.")

# --- STATISTICAL FORECAST PREPARATION ---
with col_stat:
    st.markdown("### 📊 Statistico (Syntetos-Boylan Approx.)")
    st.caption(f"Articoli: {len(df_stat_input)}")
    
    if not df_stat_input.empty:
        # Recupera metriche necessarie per SBA
        # Formula SBA: Forecast = (Mean_Demand / ADI) * (1 - alpha/2)
        # Assumiamo alpha = 0.1 (smoothing factor standard per livello) -> Factor = 0.95
        
        # Filtra colonne utili dal df_stat_input che è una copia di df_pivot_analisi filtrato
        df_sba = df_stat_input[['Mean Demand', 'ADI']].copy()
        
        # Gestione ADI < 1 (non dovrebbe succedere con logica attuale, ma per sicurezza)
        df_sba['ADI'] = df_sba['ADI'].replace(0, 1) 
        
        # Calcolo SBA
        alpha = 0.1
        correction_factor = (1 - alpha / 2)
        df_sba['Forecast_SBA_Mensile'] = (df_sba['Mean Demand'] / df_sba['ADI']) * correction_factor
        
        df_stat_forecast = pd.DataFrame({
            'Codice Articolo': df_stat_input['Codice Articolo'],
            'Forecast_SBA_Mensile': df_sba['Forecast_SBA_Mensile'].round(0),
            'Metodo': 'Syntetos-Boylan (alpha=0.1)'
        })
        df_stat_forecast['Forecast_SBA_Mensile'].fillna(0, inplace=True)
        df_stat_forecast['Forecast_SBA_Mensile'] = df_stat_forecast['Forecast_SBA_Mensile'].astype(int)
        st.dataframe(df_stat_forecast, use_container_width=True)
        st.success(f"Forecast SBA Calcolato per {len(df_stat_forecast)} articoli")
    else:
        st.info("Nessun articolo per Forecast Statistico.")


# Orizzonte Previsionale
periodi = st.number_input("Orizzonte previsionale [mesi], default = 4", min_value=1, max_value=12, value=4, step=1)

# --- REPORT OTTIMIZZAZIONE E SELEZIONE MODELLO ---
opt_data = load_optimal_params()
if opt_data:
    st.write("**Ottimizzazione Iperparametri:**")
    with st.expander("📊 Vedi Report Ottimizzazione (Optuna)"):
        st.write(f"**Ultimo aggiornamento:** {opt_data.get('last_update', 'N/A')}")
        mae = opt_data.get('best_mae', 'N/A')
        if isinstance(mae, (float, int)) and mae > 900000:
            mae_display = "Indefinito (Storico insufficiente)"
        else:
            mae_display = f"{mae:.2f}" if isinstance(mae, (float, int)) else str(mae)
        st.write(f"**Miglior Errore (Weighted MAE):** {mae_display}")
        st.json(opt_data.get('params', {}))
    
    st.write("**Configurazione Modello:**")
    model_choice = st.radio(
        "Scegli il modello NeuralProphet da utilizzare:",
        ["Modello Iron (Default)", "Modello Ottimizzato (Optuna)"],
        index=0,
        help="Il modello Iron usa parametri stabili predefiniti. Il modello Ottimizzato usa i parametri trovati da Optuna per articoli Smooth/Erratic."
    )
    use_optimized = (model_choice == "Modello Ottimizzato (Optuna)")
else:
    st.info("💡 Nessuna ottimizzazione trovata. Verrà utilizzato il modello Iron di default.")
    use_optimized = False

# Riepilogo finale
msg_model = f"""
**Configurazione attuale:**
- Orizzonte previsionale: {periodi} mesi
- Modello: {"Ottimizzato (Optuna)" if use_optimized else "Iron (Default)"}
"""
st.info(msg_model)

# --- GESTIONE STATO PULSANTE (RESET SU CAMBIO INPUT) ---
# Creiamo un "track record" per resettare il calcolo se cambiano le impostazioni
current_params = {
    "periodi": periodi,
    "use_optimized": use_optimized,
    "data_hash": len(df_np_long) # Semplice tracking del numero righe
}

if "last_params" not in st.session_state:
    st.session_state.last_params = current_params

# Se uno dei parametri cambia, resettiamo lo stato del forecasting
if st.session_state.last_params != current_params:
    st.session_state.forecasting_done = False
    st.session_state.last_params = current_params




# crea pulsante per procedere e gestisce session state
if 'forecasting_done' not in st.session_state:
    st.session_state.forecasting_done = False

col1, col2 = st.columns(2)
with col1:
    if st.button('✅ Procedi con il forecasting'): 
        st.session_state.forecasting_done = True

with col2:
    if st.button('🔄 Reset forecasting'):
        st.session_state.forecasting_done = False
        st.rerun()

if not st.session_state.forecasting_done:
    st.stop()
st.subheader('Preparazione del modello... attendere', divider='red')

m_np, metrica = opt_model(df_np_long, periodi=periodi, use_optimized=use_optimized)

st.success('Modello NeuralProphet creato con successo!')


@st.cache_data
def forecast(_model, df, horizon):
    # Restituisce i dataframe in scala reale (Pezzi)
    predicted = _model.predict(df)
    df_future_np = _model.make_future_dataframe(df, n_historic_predictions=True, periods=horizon)
    forecast_np = _model.predict(df_future_np)
    return predicted, forecast_np

orizzonte = periodi

set_log_level("ERROR")

st.subheader('Fit & Predict', divider='red')

predicted, forecast_np = forecast(m_np, df_np_long, orizzonte)

# --- IRON CLIPPING ---
# Applichiamo semplicemente un clip a 0 per garantire non-negatività sui risultati
def apply_iron_clipping(df_res):
    df_out = df_res.copy()
    cols_to_clip = ['yhat1'] + [c for c in df_out.columns if c.startswith('yhat')]
    for col in cols_to_clip:
        if col in df_out.columns:
            df_out[col] = df_out[col].clip(lower=0)
    return df_out

predicted = apply_iron_clipping(predicted)
forecast_np = apply_iron_clipping(forecast_np)

st.success('Forecasting effettuato con successo (Iron-Clipping)!')


st.write('Predicted: tabella con previsioni sui dati storici')
st.dataframe(predicted)

st.write('Forecast: tabella con previsioni sui dati storici e previsioni future')
st.dataframe(forecast_np)


col_fcst_1, col_fcst_2 = st.columns(2)

with col_fcst_1:
    st.write('Forecast: tabella con previsioni sui dati storici e previsioni future')
    st.dataframe(forecast_np)
    st.caption('Le colonne yhat1,..., yhatn, rappresentano le previsioni per i mesi futuri in base all\'orizzonte selezionato prodotte n periodi fa.')
    scarica_excel(forecast_np, 'forecast_neuralprophet.xlsx')
    st.caption('Nome file: forecast_neuralprophet.xlsx')

with col_fcst_2:
    # crea dataframe con ID, ds, y e previsioni
    predicted_da_scaricare = forecast_np[['ID', 'ds', 'y']].copy()
    for i in range(1, orizzonte + 1):
        if f'yhat{i}' in forecast_np.columns:
            predicted_da_scaricare[f'yhat{i}'] = forecast_np[f'yhat{i}']
    # Aggiungi colonna latest_forecast
    def get_latest_forecast(row):
        # Solo se y non è None (cioè per le previsioni future)
        if pd.isna(row['y']):
            # Cerca il primo valore yhat disponibile (yhat1, yhat2, etc.)
            for i in range(1, orizzonte + 1):
                yhat_col = f'yhat{i}'
                if yhat_col in predicted.columns and pd.notna(row[yhat_col]):
                    return row[yhat_col]
        return None
    predicted_da_scaricare['latest_forecast'] = predicted_da_scaricare.apply(get_latest_forecast, axis=1)
    predicted_da_scaricare.drop(columns = [col for col in predicted_da_scaricare.columns if col.startswith('yhat')], inplace=True)
    predicted_da_scaricare.rename(columns={'ds': 'Data', 'latest_forecast': 'Quantità Forecast','y': 'Quantità Actual'}, inplace=True)
    st.write('DataFrame con previsioni per orizzonte selezionato:')
    st.dataframe(predicted_da_scaricare)
    scarica_excel(predicted_da_scaricare, 'predicted_neuralprophet.xlsx')
    st.caption('Nome file: predicted_neuralprophet.xlsx')
    # per esportazione: magazzino = 100 e proprietà = 0   

st.subheader('Adattamento del forecasting a J-Galileo', divider='orange')
#st.dataframe(predicted_da_scaricare)

df_galileo = predicted_da_scaricare.copy()
df_galileo = df_galileo[['ID', 'Data', 'Quantità Forecast']]
# rimuove righe None
df_galileo = df_galileo.dropna(subset=['Quantità Forecast'])
# arrotonda e converte in intero
df_galileo['Quantità Forecast'] = df_galileo['Quantità Forecast'].round(0).astype(int)
df_galileo.rename(columns={'ID':'Articolo'}, inplace=True)

df_galileo['Quantità Forecast'] = df_galileo['Quantità Forecast'].astype(int)
df_galileo_pivot = df_galileo.pivot(index='Articolo', columns='Data', values='Quantità Forecast')
df_galileo_pivot.reset_index(inplace=True)
df_galileo_pivot['Proprietà'] = 0
df_galileo_pivot['Magazzino'] = 100
df_galileo_pivot['Proprietà'] = df_galileo_pivot['Proprietà'].astype(str)
df_galileo_pivot['Magazzino'] = df_galileo_pivot['Magazzino'].astype(str)
#ordina colonne: Articolo, Proprietà, Magazzino, e poi le date
df_galileo_pivot = df_galileo_pivot[['Articolo', 'Proprietà', 'Magazzino'] + [col for col in df_galileo_pivot.columns if col not in ['Articolo', 'Proprietà', 'Magazzino']]]

#df_galileo_pivot
# inserisci pulsante per scaricare df_galileo_pivot in excel
#scarica_excel(df_galileo_pivot, 'df_galileo_pivot.xlsx')


df_galileo_sba = df_stat_forecast.copy()
df_galileo_sba.rename(columns={'Codice Articolo':'Articolo'}, inplace=True)


# Creazione file unico con df_galileo_pivot e df_galileo_sba
df_galileo_unito = pd.concat([df_galileo_pivot, df_galileo_sba], ignore_index=True)
df_galileo_unito['Proprietà']=0
df_galileo_unito['Magazzino']=100
df_galileo_unito['Proprietà'] = df_galileo_unito['Proprietà'].astype(str)
df_galileo_unito['Magazzino'] = df_galileo_unito['Magazzino'].astype(str)
df_galileo_unito['Metodo'].fillna('NeuralProphet', inplace=True)

# Logica per riempire le colonne data con il valore di Forecast_SBA_Mensile per le righe SBA
cols_escluse = ['Articolo', 'Proprietà', 'Magazzino', 'Metodo', 'Forecast_SBA_Mensile', 'Codice Articolo']
date_cols = [c for c in df_galileo_unito.columns if c not in cols_escluse]

mask_sba = df_galileo_unito['Metodo'] == 'Syntetos-Boylan (alpha=0.1)'
for col in date_cols:
    df_galileo_unito.loc[mask_sba, col] = df_galileo_unito.loc[mask_sba, 'Forecast_SBA_Mensile']

# Rimuove la prima colonna data
df_galileo_unito = df_galileo_unito.drop(columns=date_cols[0])
#rimuove la colonna Forecast_SBA_Mensile
df_galileo_unito = df_galileo_unito.drop(columns=['Forecast_SBA_Mensile'])
# Rimuove le righe con valori 0 nelle colonne data
df_galileo_unito = df_galileo_unito[(df_galileo_unito[date_cols[1:]] != 0).all(axis=1)]


#st.write('df_galileo_unito')
#st.dataframe(df_galileo_unito)
# aggiunge a df_galileo_unito una colonna vuota 'Cliente/Fornitore' e la sposta in prima posizione
df_galileo_unito.insert(0, 'Cliente/Fornitore', '')
# aggiunnge a df_galileo_unito una colonna vuota 'Commessa' e la sposta in terza posizione
df_galileo_unito.insert(2, 'Commessa', '')
# aggiunge a df_galileo_unito una colonna vuota 'Sotto commessa' e la sposta in quarta posizione
df_galileo_unito.insert(3, 'Sotto commessa', '')
# rimuove la colonna Metodo
df_galileo_unito = df_galileo_unito.drop(columns=['Metodo'])
#st.write('df_galileo_unito')
st.dataframe(df_galileo_unito)

st.write('Numero codici articoli:', len(df_galileo_unito['Articolo'].unique()))
scarica_excel(df_galileo_unito, 'df_galileo_per_forecasting.xlsx')




################ TEST E VISUALIZZAZIONI

# st.subheader('test', divider='orange')

# serie_test = '00454'
# st.write('Serie di test usata per il modello:', serie_test)

# st.write('Last prediction for serie test:')
# last_pred = m_np.get_latest_forecast(forecast_np, df_name= serie_test)
# st.dataframe(last_pred)

# st.write('plot')
# fig_plot = m_np.plot(forecast_np, df_name= serie_test)
# st.plotly_chart(fig_plot, use_container_width=True)

# st.write('plot focus 1')
# fig_focus_1 = m_np.plot(forecast_np, df_name= serie_test, forecast_in_focus=1)
# st.plotly_chart(fig_focus_1, use_container_width=True)

# st.write('plot focus 4')
# fig_focus_4 = m_np.plot(forecast_np, df_name= serie_test, forecast_in_focus=4)
# st.plotly_chart(fig_focus_4, use_container_width=True)

# st.write('plot parameters')
# fig_parameters = m_np.plot_parameters(df_name= serie_test)
# st.plotly_chart(fig_parameters, use_container_width=True)

# st.write('plot components')
# fig_components = m_np.plot_components(forecast_np, df_name= serie_test, plotting_backend='plotly')
# st.plotly_chart(fig_components, use_container_width=True)

# st.write('plot latest forecast, include = periodi')
# fig_latest_forecast = m_np.plot_latest_forecast(forecast_np, df_name= serie_test,include_previous_forecasts=periodi, plotting_backend='plotly')
# st.plotly_chart(fig_latest_forecast, use_container_width=True)

# st.write('test')
# st.write(m_np.test(df_np_long))

############ FINE TEST E VISUALIZZAZIONI





st.subheader('Analisi errore di forecasting MAE (Mean Absolute Error) - NMAE (Normalized Mean Absolute Error) | ultimi 12 mesi', divider='orange')

# ==============================================================================
# 1. UNIFICAZIONE FORECAST (NeuralProphet + SBA Statistico)
# ==============================================================================

# A) NeuralProphet History
# ------------------------
df_np_history = pd.DataFrame()
if 'predicted' in locals() and predicted is not None and not predicted.empty:
    # predicted contiene: ID, ds, y, yhat1 
    # (solo per le serie processate con NP: Smooth/Erratic)
    
    # Filtriamo colonne necessarie
    cols_np = ['ID', 'ds', 'y', 'yhat1']
    if all(c in predicted.columns for c in cols_np):
        df_np_history = predicted[cols_np].copy()
        df_np_history.rename(columns={'yhat1': 'y_forecast', 'y': 'y_actual'}, inplace=True)
        df_np_history['Model_Type'] = 'NeuralProphet'

# B) Statistical History (SBA)
# ----------------------------
df_stat_history = pd.DataFrame()
if 'df_stat_forecast' in locals() and not df_stat_forecast.empty:
    # df_stat_forecast ha: Codice Articolo, Forecast_SBA_Mensile, Metodo
    # Dobbiamo creare lo "storico" confrontando questo valore costante con le vendite reali ultimi 12 mesi
    
    # Recuperiamo i dati reali per questi articoli (df_stat_input o df_venduto_pivot)
    codici_stat = df_stat_forecast['Codice Articolo'].unique()
    
    # Prendiamo dal pivot originale gli ultimi 12 mesi
    df_actual_stat = df_venduto_pivot.loc[codici_stat].copy()
    last_12_cols_stat = df_actual_stat.columns[-12:]
    df_actual_stat = df_actual_stat[last_12_cols_stat]
    
    # Melt per avere formato long (ID, ds, val)
    df_actual_stat_long = df_actual_stat.reset_index().melt(
        id_vars='Codice Articolo', 
        var_name='ds', 
        value_name='y_actual'
    )
    df_actual_stat_long.rename(columns={'Codice Articolo': 'ID'}, inplace=True)
    df_actual_stat_long['ds'] = pd.to_datetime(df_actual_stat_long['ds'])
    
    # Merge con il valore forecast costante
    df_stat_merged = df_actual_stat_long.merge(
        df_stat_forecast[['Codice Articolo', 'Forecast_SBA_Mensile']],
        left_on='ID', right_on='Codice Articolo', how='left'
    )
    
    df_stat_history = df_stat_merged[['ID', 'ds', 'y_actual', 'Forecast_SBA_Mensile']].copy()
    df_stat_history.rename(columns={'Forecast_SBA_Mensile': 'y_forecast'}, inplace=True)
    df_stat_history['Model_Type'] = 'SBA (Statistico)'

# C) Concatenazione
# -----------------
df_all_history = pd.concat([df_np_history, df_stat_history], ignore_index=True)

if df_all_history.empty:
    st.warning("Nessun dato storico disponibile per il calcolo dell'errore (né NP né Statistico).")
    st.stop()


# ==============================================================================
# 2. CALCOLO ERRORI (Unificato)
# ==============================================================================

# Filtra ultimi 12 mesi (calcolato sulla data massima del dataset unificato)
data_massima = df_all_history['ds'].max()
data_limite = (pd.to_datetime(data_massima) - pd.DateOffset(months=11)).strftime('%Y-%m-%d')
errore_latest = df_all_history[df_all_history['ds'] >= data_limite].copy()

# Ensure numeric
errore_latest['y_actual'] = pd.to_numeric(errore_latest['y_actual'], errors='coerce').fillna(0)
errore_latest['y_forecast'] = pd.to_numeric(errore_latest['y_forecast'], errors='coerce').fillna(0)

# Calcolo Errore Puntuale
errore_latest['latest_forecast_error'] = errore_latest['y_actual'] - errore_latest['y_forecast']

# st.write('Dettaglio Errori (ultimi 12 mesi) - Tutti i Modelli')
# st.dataframe(errore_latest.head())

# Calcolo di RSFE, MAD e TS per ogni ID
errore_latest_metriche = errore_latest.copy()

# Metrica RSFE (Running Sum of Forecast Errors): Somma algebrica degli errori
errore_latest_metriche['RSFE'] = errore_latest_metriche.groupby('ID')['latest_forecast_error'].cumsum()

# Metrica MAD (Mean Absolute Deviation): Media dei valori assoluti degli errori (running mean)
errore_latest_metriche['abs_error'] = errore_latest_metriche['latest_forecast_error'].abs()
errore_latest_metriche['MAD_running'] = errore_latest_metriche.groupby('ID')['abs_error'].transform(lambda x: x.expanding().mean())

# Metrica RMSE (Root Mean Squared Error): Calcolo dei quadrati per il riepilogo
errore_latest_metriche['squared_error'] = errore_latest_metriche['latest_forecast_error']**2

# Metrica TS (Tracking Signal): RSFE / MAD
errore_latest_metriche['TS'] = errore_latest_metriche['RSFE'] / errore_latest_metriche['MAD_running']

# df_show = errore_latest_metriche[['ID', 'ds', 'Model_Type', 'y_actual', 'y_forecast', 'latest_forecast_error', 'RSFE', 'MAD_running', 'TS']]
# st.write('**Dettaglio metriche di errore (running totals):**')
# st.dataframe(df_show)

# Tabella riassuntiva finale per ID
st.write('**Riepilogo Metriche per ID (RSFE, MAD, RMSE, TS):**')
df_metrics_summary = errore_latest_metriche.groupby(['ID', 'Model_Type']).agg({
    'latest_forecast_error': 'sum',
    'abs_error': 'mean',
    'squared_error': lambda x: np.sqrt(x.mean())
}).reset_index()

df_metrics_summary.columns = ['ID', 'Model_Type', 'RSFE', 'MAD', 'RMSE']
df_metrics_summary['TS'] = df_metrics_summary['RSFE'] / df_metrics_summary['MAD']

# Aggiunge informazioni materiali
df_metrics_summary = df_metrics_summary.merge(df_anagrafica, left_on='ID', right_on='Codice Articolo', how='left')
# Visualizza colonne core
cols_output = ['ID', 'Descrizione Articolo univoca', 'Model_Type', 'RSFE', 'MAD', 'RMSE', 'TS']
df_metrics_summary = df_metrics_summary[cols_output]
st.dataframe(df_metrics_summary)

scarica_excel(df_metrics_summary, 'riepilogo_metriche_forecasting.xlsx')
st.caption('Nome file: riepilogo_metriche_forecasting.xlsx')

#scarica file errore in excel
scarica_excel(errore_latest, 'errore_latest_forecasting_unified.xlsx')
st.caption('Nome file: errore_latest_forecasting_unified.xlsx')

# ==============================================================================
# 3. MAE & NMAE ANALYSIS
# ==============================================================================

# calcolo MAE
df_calcolo_mae = errore_latest[['ID','ds', 'y_actual', 'y_forecast','latest_forecast_error', 'Model_Type']].copy()
df_calcolo_mae['absolute_error'] = df_calcolo_mae['latest_forecast_error'].abs()
df_mae = df_calcolo_mae.groupby(['ID', 'Model_Type']).agg({'absolute_error': 'mean'}).reset_index()
df_mae.rename(columns={'absolute_error': 'MAE'}, inplace=True)

# calcolo NMAE (normalized MAE)
# calcolo Media Quantità Storica per ogni ID (sugli ultimi 12 mesi effettivi)
df_media_qty_storico = errore_latest.groupby('ID').agg({'y_actual': 'mean'}).reset_index()
df_media_qty_storico.rename(columns={'y_actual': 'media_qty_storico'}, inplace=True)

# unisci con df_mae
df_mae = df_mae.merge(df_media_qty_storico, on='ID')

# calcola NMAE
# Attenzione: se media_qty_storico è 0 (impossibile se attivo, ma..), NMAE è inf.
df_mae['NMAE'] = df_mae['MAE'] / df_mae['media_qty_storico'] * 100
df_mae = df_mae[['ID', 'Model_Type', 'media_qty_storico', 'MAE', 'NMAE']]
# unisci con anagrafica per avere descrizione
df_mae = df_mae.merge(df_anagrafica, left_on='ID', right_on='Codice Articolo', how='left')
cols = ['ID', 'Descrizione Articolo univoca', 'Model_Type', 'media_qty_storico', 'MAE', 'NMAE']
df_mae = df_mae[cols]


# Visualizza in colonne tabella e grafico
col_err_1, col_err_2 = st.columns(2)

with col_err_1:
    st.write('MAE e NMAE su ultimi 12 mesi per materiali raggruppati:')
    st.dataframe(df_mae)

with col_err_2:
    import plotly.figure_factory as ff
    # Drop NaN/Inf per il grafico di densità
    nmae_values = df_mae['NMAE'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if not nmae_values.empty and len(nmae_values) > 1:
        try:
            fig_nmae = ff.create_distplot(
                [nmae_values],
                group_labels=['NMAE'],
                bin_size=1,
                show_hist=False,
                colors=['orange']
            )
        except np.linalg.LinAlgError:
             st.warning("Varianza insufficiente per KDE NMAE. Mostro istogramma.")
             fig_nmae = px.histogram(x=nmae_values, nbins=20, labels={'x':'NMAE'}, color_discrete_sequence=['orange'])
        except Exception as e:
             st.warning(f"Errore grafico NMAE: {e}")
             fig_nmae = None
        if fig_nmae:
            fig_nmae.update_layout(
                title='Distribuzione del NMAE (%) - Scala Log',
                xaxis_title='NMAE (%)',
                yaxis_title='Densità',
                xaxis_type="log",
                height=700
            )
            st.plotly_chart(fig_nmae, use_container_width=True)
    else:
        st.info("Dati insufficienti per il grafico di densità NMAE.")


# crea scateerplot per NMAE vs Qty Storico e usa come dimensione del punto il MAE
st.subheader('NMAE vs Qty Storico per Materiali raggruppati', divider='orange')

# Crea categorie di dimensione per la legenda
df_mae_plot = df_mae.copy()
df_mae_plot['MAE_categoria'] = pd.cut(df_mae_plot['MAE'], bins=5, labels=['Molto Basso', 'Basso', 'Medio', 'Alto', 'Molto Alto'])

fig_nmae_vs_media = px.scatter(
    df_mae_plot,
    x='media_qty_storico',
    y='NMAE',
    size='MAE',  # Usa MAE come dimensione del punto
    color='MAE',  # Aggiungi scala colore per il MAE
    symbol='Model_Type', # Usa Model_Type per la forma del punto
    color_continuous_scale='Inferno',  # Scala di colori dal chiaro al scuro
    title='NMAE vs Qty Storico (dimensione e colore = MAE)',
    labels={'media_qty_storico': 'Qty Storico (pcs)', 'NMAE': 'NMAE (%)', 'MAE': 'MAE (pcs)'},
    hover_data=['ID', 'Descrizione Articolo univoca', 'Model_Type', 'MAE'],
    log_x=True, # Scala logaritmica per asse X (Qty)
    log_y=True, # Scala logaritmica per asse Y (NMAE)
    height=700,
    size_max=25  # Dimensione massima dei punti aumentata
)


# Aumenta la dimensione minima dei punti per renderli più visibili
fig_nmae_vs_media.update_traces(marker=dict(sizemin=5, line=dict(width=1, color='DarkSlateGrey')))

# Aggiorna il layout: Barra colori a destra, Legenda (simboli) in alto
fig_nmae_vs_media.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    coloraxis_colorbar=dict(
        title="MAE (pcs)",
        thicknessmode="pixels",
        thickness=15,
        lenmode="pixels",
        len=200,
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    )
)
st.plotly_chart(fig_nmae_vs_media, use_container_width=True)


# ANALISI PER SERIE
#==========================================================

# Treemap volumi
st.subheader('Treemap volumi per codice articolo', divider='orange')

# Assicurati che la colonna y sia numerica per il treemap
df_model_numeric = df_all_history.copy()  
df_model_numeric['y_actual'] = pd.to_numeric(df_model_numeric['y_actual'], errors='coerce')

# Aggrega i dati per ID per ottenere i totali
df_treemap_agg = df_model_numeric.groupby('ID')['y_actual'].sum().reset_index()

# Unisci con anagrafica per avere descrizione
df_treemap_agg = df_treemap_agg.merge(df_anagrafica, left_on='ID', right_on='Codice Articolo', how='left')

# Crea labels che includono ID e valore y
df_treemap_agg['labels_with_value'] = df_treemap_agg['ID']  + '<br>' + df_treemap_agg['Descrizione Articolo univoca'] + '<br>' + df_treemap_agg['y_actual'].astype(str) + ' pcs'

# Crea la treemap direttamente con i dati aggregati
fig_treemap = px.treemap(
    df_treemap_agg,
    path=['labels_with_value'],  # Usa le labels personalizzate
    values='y_actual',                  # Dimensione dei rettangoli proporzionale al totale
    width=1200, height=600,
    color='ID',                  # Colore basato sull'ID per colori distinti
    color_discrete_sequence=px.colors.qualitative.Set3  #Set3
)
st.plotly_chart(fig_treemap, use_container_width=True)


# ==============================================================================
# 4. ANALISI DETTAGLIATA PER SINGOLO ARTICOLO
# ==============================================================================
st.subheader('Analisi di dettaglio e Residui per Codice Articolo', divider='orange')

# Selezione codice (Tutti i codici previsti: NP + SBA)
lista_item = sorted(df_all_history['ID'].unique().tolist())
codice = st.selectbox('Seleziona codice articolo per analisi di dettaglio:', options=lista_item, index=0)

# Recupera Metadati e Metriche
descrizione_art = df_anagrafica[df_anagrafica['Codice Articolo'] == codice]['Descrizione Articolo univoca'].values[0] if not df_anagrafica[df_anagrafica['Codice Articolo'] == codice].empty else "N/D"
metriche_art = df_metrics_summary[df_metrics_summary['ID'] == codice].iloc[0] if not df_metrics_summary[df_metrics_summary['ID'] == codice].empty else None
tipo_modello = df_all_history[df_all_history['ID'] == codice]['Model_Type'].iloc[0]

st.markdown(f"**Codice:** `{codice}` | **Descrizione:** *{descrizione_art}* | **Modello Usato:** `{tipo_modello}`")

# Visualizzazione Metriche Rapide
if metriche_art is not None:
    mae_art = df_mae[df_mae['ID'] == codice]['MAE'].values[0] if not df_mae[df_mae['ID'] == codice].empty else 0
    nmae_art = df_mae[df_mae['ID'] == codice]['NMAE'].values[0] if not df_mae[df_mae['ID'] == codice].empty else 0
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("MAE (pcs)", f"{mae_art:.2f}")
    m_col2.metric("NMAE (%)", f"{nmae_art:.1f}%")
    m_col3.metric("Tracking Signal", f"{metriche_art['TS']:.2f}")
    m_col4.metric("Bias Totale (RSFE)", f"{metriche_art['RSFE']:.0f}")

st.divider()

# Tab 1: Analisi Forecast | Tab 2: Analisi Residui
tab1, tab2 = st.tabs(['📈 Analisi Forecast','🔍 Analisi dei Residui'])

with tab1:
    if tipo_modello == 'NeuralProphet':
        st.write("**Visualizzazione NeuralProphet (Dati storici e componenti):**")
        # Visualizza grafico principale di NeuralProphet
        fig_fcst = m_np.plot(forecast_np, df_name=codice)
        st.plotly_chart(fig_fcst, use_container_width=True)
        
        # Visualizza componenti (trend, stagionalità)
        st.write("**Componenti del modello:**")
        fig_fcst_components = m_np.plot_components(forecast_np, df_name=codice)
        st.plotly_chart(fig_fcst_components, use_container_width=True)
    else:
        st.write("**Visualizzazione SBA (Confronto Venduto vs Media Statistica):**")
        # Per SBA mostriamo il venduto reale degli ultimi 12 mesi vs il forecast costante
        df_plot_sba = df_all_history[df_all_history['ID'] == codice].copy()
        fig_sba = px.line(
            df_plot_sba, x='ds', y=['y_actual', 'y_forecast'],
            title=f'Venduto Reale vs Livello SBA per {codice}',
            labels={'ds': 'Data', 'value': 'Pezzi', 'variable': 'Tipo'},
            markers=True
        )
        st.plotly_chart(fig_sba, use_container_width=True)

with tab2:
    st.write("**Analisi dell'errore (Residuo = Reale - Previsto)**")
    
    # 1. Recupero Dati Residui
    # -----------------------
    # Dati unificati di errore (ultimi 12 mesi)
    df_res_art = errore_latest[errore_latest['ID'] == codice].copy()
    
    if df_res_art.empty:
        st.warning("Dati residui non disponibili per questo articolo.")
    else:
        # A) Residuo Standard (Unified)
        df_res_art['residual'] = df_res_art['latest_forecast_error']
        
        # B) Analisi Multi-Residuo (Solo per NeuralProphet se richiesto)
        df_residui_multi = pd.DataFrame()
        if tipo_modello == 'NeuralProphet':
            # Funzione originale per calcolare residui su diverse yhat (backtest)
            def calcola_residui_multi(df_pred, df_name):
                df_item = df_pred[df_pred['ID'] == df_name].copy()
                y_true = df_item['y']
                ds = df_item['ds']
                residui = {'ds': ds, 'y': y_true}
                yhat_columns = []
                for col in df_pred.columns:
                    if col.startswith('yhat'):
                        yhat = df_item[col]
                        residui[col] = y_true - yhat
                        yhat_columns.append(col)
                df_res_multi = pd.DataFrame(residui)
                if yhat_columns:
                    df_res_multi = df_res_multi.dropna(subset=yhat_columns, how='all')
                return df_res_multi
            
            df_residui_multi = calcola_residui_multi(predicted, codice)
            if not df_residui_multi.empty:
                # Versione melted per scatter plot multi-residuo
                df_res_plot = df_residui_multi.melt(id_vars=['ds', 'y'], var_name='horizon', value_name='residual')
                x_col = 'y'
                color_col = 'horizon'
            else:
                df_res_plot = df_res_art.copy()
                df_res_plot['residual'] = df_res_plot['latest_forecast_error']
                x_col = 'y_actual'
                color_col = None
        else:
            df_res_plot = df_res_art.copy()
            df_res_plot['residual'] = df_res_plot['latest_forecast_error']
            x_col = 'y_actual'
            color_col = None
        
        # 2. Plotting
        # -----------
        col_res1, col_res2 = st.columns(2)
        colors_res = px.colors.qualitative.Set1

        with col_res1:
            # Grafico Residui vs Valori Reali
            st.write("**Residui vs Valori Reali**")
            fig_res_vs_y = px.scatter(
                df_res_plot, x=x_col, y='residual', color=color_col,
                title=f'Residui vs Reale per {codice}',
                labels={x_col: 'Valore Reale', 'residual': 'Residuo (Reale-Prev)', 'horizon': 'Orizzonte (Step)'},
                color_discrete_sequence=px.colors.qualitative.Safe if color_col else [colors_res[0]]
            )
            fig_res_vs_y.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res_vs_y, use_container_width=True)

        with col_res2:
            # Grafico Residui nel Tempo
            st.write("**Residui nel Tempo**")
            fig_res_time = px.scatter(
                df_res_plot, x='ds', y='residual', color=color_col,
                title=f'Residui nel Tempo per {codice}',
                labels={'ds': 'Data', 'residual': 'Residuo', 'horizon': 'Orizzonte (Step)'},
                color_discrete_sequence=px.colors.qualitative.Safe if color_col else [colors_res[0]]
            )
            fig_res_time.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res_time, use_container_width=True)

        # Distribuzione Errore
        st.write("**Distribuzione dell'Errore (KDE)**")
        import plotly.figure_factory as ff
        
        # Se abbiamo dati multi-residuo (NP), mostriamo il grafico con subplot
        if not df_residui_multi.empty:
            st.caption("Analisi multi-residuo (yhat1, yhat2...) basata sui residui di backtest di NeuralProphet")
            multi_cols = [c for c in df_residui_multi.columns if c.startswith('yhat')]
            n_cols = len(multi_cols)
            from plotly.subplots import make_subplots
            fig_dist = make_subplots(rows=1, cols=n_cols, subplot_titles=[f'Distrib. {c}' for c in multi_cols], shared_yaxes=True)
            
            for i, col_res in enumerate(multi_cols):
                res_validi = df_residui_multi[col_res].dropna()
                if not res_validi.empty:
                    try:
                        fig_temp = ff.create_distplot([res_validi.values], [col_res], show_hist=False, show_rug=True, colors=[colors_res[i % len(colors_res)]])
                        for trace in fig_temp.data:
                            trace.showlegend = False
                            fig_dist.add_trace(trace, row=1, col=i+1)
                    except np.linalg.LinAlgError:
                        fig_dist.add_trace(go.Histogram(x=res_validi.values, marker_color=colors_res[i % len(colors_res)], showlegend=False), row=1, col=i+1)
                    except Exception:
                        pass
            
            fig_dist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            # Grafico distribuzione singolo per SBA o NP base
            res_val = df_res_art['residual'].dropna()
            if not res_val.empty:
                try:
                    fig_dist_single = ff.create_distplot([res_val.values], ['Residuo'], show_hist=True, show_rug=True)
                    fig_dist_single.update_layout(title="Distribuzione Errore (Ultimi 12 mesi)", height=400)
                    st.plotly_chart(fig_dist_single, use_container_width=True)
                except np.linalg.LinAlgError:
                    # Fallback per varianza nulla (es. tutti zeri) che impedisce il calcolo KDE
                    st.warning("Varianza insufficiente per il grafico di densità. Visualizzazione istogramma semplice.")
                    fig_hist = px.histogram(x=res_val.values, nbins=20, title="Distribuzione Errore (Istogramma Semplice)")
                    fig_hist.update_layout(xaxis_title="Residuo", yaxis_title="Conteggio", height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception as e:
                    st.warning(f"Impossibile calcolare grafico distribuzione: {e}")


# Aggiunta tabella "Ultimo Forecast" nel Tab 1 (post-hoc edit per integrazione)
with tab1:
    st.divider()
    st.write(f"**Tabella Prossimi Forecast per {codice}:**")
    if tipo_modello == 'NeuralProphet':
        # Estrazione del "latest forecast" dai backtest (prendiamo il primo yhat non nullo per ogni riga futura)
        df_item_fcst = forecast_np[forecast_np['ID'] == codice].copy()
        def get_single_latest(row):
            if pd.isna(row['y']):
                for i in range(1, orizzonte + 1):
                    yhat_col = f'yhat{i}'
                    if yhat_col in df_item_fcst.columns and pd.notna(row[yhat_col]):
                        return row[yhat_col]
            return None
        df_item_fcst['Forecast'] = df_item_fcst.apply(get_single_latest, axis=1)
        df_ultimo_forecast = df_item_fcst[df_item_fcst['Forecast'].notna()][['ds', 'Forecast']]
        df_ultimo_forecast.rename(columns={'ds': 'Data', 'Forecast': 'Forecast [pcs]'}, inplace=True)
        st.dataframe(df_ultimo_forecast, use_container_width=True)
    else:
        # Per SBA, proiettiamo il valore mensile per i prossimi 12 mesi
        mesi_futuri = pd.date_range(start=data_massima + pd.DateOffset(months=1), periods=orizzonte, freq='M')
        # mesi_futuri come anno-mese
        mesi_futuri = mesi_futuri.strftime('%Y-%m')
        val_sba = df_stat_forecast[df_stat_forecast['Codice Articolo'] == codice]['Forecast_SBA_Mensile'].values[0]
        df_futuro_sba = pd.DataFrame({'Data': mesi_futuri, 'Forecast [pcs]': val_sba})
        st.dataframe(df_futuro_sba, use_container_width=True)

st.stop()