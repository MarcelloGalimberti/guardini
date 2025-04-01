# Processo dati Guardini
# env neuraplprophet conda

# Per il deployment: creare ambiente con requirements.txt

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from neuralprophet import set_log_level


####### Impaginazione

st.set_page_config(layout="wide")

url_immagine = 'guardini.png'#?raw=true'

col_1, col_2 = st.columns([2, 4])

with col_1:
    st.image(url_immagine, width=400)

with col_2:
    st.title('Forecasting | orizzonte 4 mesi')

st.header('Caricamento dati | consumi.xlsx', divider='orange')

####### Caricamento dati

uploaded_consumi = st.file_uploader("Carica consumi ultimi 26 mesi") # nome file da caricare
if not uploaded_consumi:
    st.stop()
consumi_raw=pd.read_excel(uploaded_consumi, skiprows=[1], header=0) #, skiprows=[0,1,2,3,4],

#st.dataframe(consumi_raw)

# pre-processing

st.subheader('Pre-processing file consumi', divider='orange')
st.write('- consumi ultimi 12 mesi > 0')
st.write('- codici che hanno una serie storica significativa, pari a 20 mesi')
st.write('- codici non obsoleti')
st.write('- codici non SK')


colonne_mesi_raw = consumi_raw.columns[11:-1].tolist()
dict_mesi = {'Gen.':'.01.', 'Feb.':'.02.', 'Mar.':'.03.', 'Apr.':'.04.', 'Mag.':'.05.', 'Giu.':'.06.',
              'Lug.':'.07.', 'Ago.':'.08.', 'Set.':'.09.', 'Ott.':'.10.', 'Nov.':'.11.', 'Dic.':'.12.'}

lista_sostituita = []
for item in colonne_mesi_raw:
    for key, value in dict_mesi.items():
        item = item.replace(key, value)  # Sostituisci tutte le occorrenze nella stringa
    lista_sostituita.append(item)

lista_colonne = [item.replace("Consumi\n", "01") for item in lista_sostituita]

consumi_raw.drop(columns=['Giacenza','Indice di Rotazione','Scorta Sicurezza','Lotto Riordino','Media Mese Consumi','Scorta vs Media','Mesi di Cop. (Scorta)',
                          'Mesi di Cop. (Media)','Consumi','Fornitore Abituale'], inplace=True)

consumi_raw.columns = ['Codice', 'Descrizione']  + lista_colonne

consumi_raw[lista_colonne] = consumi_raw[lista_colonne].clip(lower=0)

consumi_raw.fillna(0, inplace=True)

consumi_raw['Consumi'] = consumi_raw[lista_colonne].sum(axis=1)

df_consumi = consumi_raw[consumi_raw['Consumi'] > 0]  

# applicazione filtri

primi_7_mesi = df_consumi.columns[2:9].tolist()
ultimi_12_mesi = df_consumi.columns[-13:-1].tolist()

df_consumi['Consumi_ultimi_12_mesi'] = df_consumi[ultimi_12_mesi].sum(axis=1)
df_consumi['Consumi_primi_7_mesi'] = df_consumi[primi_7_mesi].sum(axis=1)

df_consumi['test_ultimi_12_mesi'] = np.where(df_consumi['Consumi_ultimi_12_mesi'] == 0, 1, 0).astype(bool)
df_consumi['test_primi_7_mesi'] = np.where(df_consumi['Consumi_primi_7_mesi'] == 0, 1, 0).astype(bool)

df_consumi['eliminare'] = df_consumi['test_ultimi_12_mesi'] | df_consumi['test_primi_7_mesi']

df_consumi['3dgt_Descrizione'] = df_consumi['Descrizione'].str[:3]
df_consumi['2dgt_Codice'] = df_consumi['Codice'].str[:2]

df_consumi_filtro_1 = df_consumi[~df_consumi['eliminare']]
df_consumi_filtro_2 = df_consumi_filtro_1[df_consumi_filtro_1['3dgt_Descrizione']!='***']
df_consumi_filtro_3 = df_consumi_filtro_2[df_consumi_filtro_2['2dgt_Codice']!='SK']

colonne_db = df_consumi_filtro_3.columns[:-8].tolist()

df_filtrato = df_consumi_filtro_3[colonne_db]

# formato per neuralprophet

df_filtrato_long = df_filtrato.melt(id_vars=['Codice', 'Descrizione'], var_name='Data', value_name='Consumi')

#st.dataframe(df_filtrato_long)

df_dati_fcst=df_filtrato_long.copy()

df_dati_fcst['art_desc']=df_dati_fcst['Codice'] + '_' + df_dati_fcst['Descrizione']

df_dati_fcst.rename(columns={'Consumi':'Venduto'},inplace=True)

df_dati_fcst.drop(columns=['Codice','Descrizione'],inplace=True)


# lista item da prevedere
# inserire lunghezza lista
lista_item = df_dati_fcst['art_desc'].unique().tolist()

st.write('Codici da prevedere dopo pre-processing: ', len(lista_item))

df_dati_fcst['Data'] = pd.to_datetime(df_dati_fcst['Data'], format='%d.%m.%y')#, dayfirst=True)

df_dati_np = df_dati_fcst.copy()

df_dati_np.columns = ['ds','y','ID']

df_dati_np = df_dati_np.sort_values(by=['ID', 'ds'], ascending=[True, True])

st.subheader('Database processato', divider='orange')
st.dataframe(df_dati_np)


#@st.cache_resource # modificato da cache_data
def vanilla_model (df):
    set_random_seed(0)
    #m_np = NeuralProphet(n_lags=10) # - secondo test
    #m_np = NeuralProphet() # global - primo test
    m_np = NeuralProphet(
        #trend_global_local='local',
        #season_global_local='local',
        seasonality_mode='multiplicative',
        n_lags=12, n_forecasts=4,
        ar_layers=[8,8])
    m_np.set_plotting_backend('plotly')
    m_np.fit(df, freq='MS') # modifica
    return m_np

st.subheader('Preparazione del modello... attendere', divider='orange')

m_np = vanilla_model(df_dati_np)

@st.cache_data
def forecast(_model, df, horizon):
    predicted = _model.predict(df)
    df_future_np = _model.make_future_dataframe(df, n_historic_predictions=True, periods=horizon)
    forecast_np = _model.predict(df_future_np)
    return predicted, forecast_np


# caricamento orizzonte previsionale
#orizzonte = st.number_input("Orizzonte previsionale [mesi], default = 4", min_value=1, max_value=12, value=4, step=1)
#st.write("Orizzonte ", orizzonte)

orizzonte = 4

set_log_level("ERROR")

st.subheader('Parametri forecast', divider='orange')

predicted, forecast_np = forecast(m_np, df_dati_np, orizzonte)

# grafici
@st.cache_data
def parametri_forecast(_m):
    fig = _m.plot_parameters(figsize=(12,4))
    return fig
fig_param = parametri_forecast(m_np)
st.plotly_chart(fig_param, use_container_width=True)

st.subheader('Distribuzione errore di forecasting Mean Absolute Error - MAE', divider='orange')
# errore forecasting
errore = predicted.copy()
errore = errore[['ID','ds','y','yhat1','yhat2','yhat3','yhat4']]
errore['ds'] = errore['ds'].dt.strftime('%Y-%m-%d')
errore.dropna(subset=['yhat1', 'yhat2', 'yhat3', 'yhat4'], how='all', inplace=True)
colonne_errore = ['e_1','e_2','e_3','e_4']
for col, i in zip(colonne_errore, range(1,5)):
    errore['e_' + str(i)] =abs(errore['y'] - errore['yhat' + str(i)])
df_result = errore.groupby('ID').apply(lambda x: x[['e_1', 'e_2', 'e_3', 'e_4']].mean().mean()).reset_index(name='mae')
fig_errore = px.histogram(df_result, x='mae', nbins=75, title='Istogramma di MAE')
fig_errore.update_layout(
    xaxis_title='MAE',
    yaxis_title='Frequenza'
)
st.plotly_chart(fig_errore, use_container_width=True)


# salvataggio su excel dei file predicted e forecast
def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Foglio1')
    return output.getvalue()

# Crea il bottone per scaricare predicted
#predicted_data = to_excel_bytes(predicted)
#st.download_button(
#    label="📥 Scarica Predicted",
#    data=predicted_data,
#    file_name='predicted_test.xlsx',
#    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
#)

# Crea il bottone per scaricare forecast_np
#forecast_data = to_excel_bytes(forecast_np)
#st.download_button(
#    label="📥 Scarica Forecast",
#    data=forecast_data,
#    file_name='forecast_test.xlsx',
#    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
#)

# Treemap volumi
st.subheader('Treemap volumi', divider='orange')
fig_treemap = px.treemap(
    df_dati_np,
    path=['ID'],       # Gerarchia: qui viene mostrato solo il codice prodotto
    values='y',      # Dimensione dei rettangoli proporzionale alle vendite
    width=1200, height=600, #title='Treemap delle Vendite per Codice Prodotto'
    color='ID',       # Colore assegnato in base al codice prodotto
    #color_discrete_sequence=px.colors.qualitative.Pastel,

) #width=1200, height=600,
st.plotly_chart(fig_treemap, use_container_width=True)

# selezionare il codice da prevedere
st.subheader('Analisi forecast', divider='orange')

codice = st.selectbox('Seleziona codice', options=lista_item, index=0)

st.write("Codice selezionato: ", codice)

# grafici 
fig_fcst = m_np.plot(forecast_np, df_name=codice)
st.plotly_chart(fig_fcst, use_container_width=True)
fig_fcst_components = m_np.plot_components(forecast_np, df_name=codice)
st.plotly_chart(fig_fcst_components, use_container_width=True)


# forecast
st.subheader('forecast', divider='orange')
forecast_np_1 = forecast_np[forecast_np['y'].isna()].reset_index(drop=True)
forecast_np_1 = forecast_np_1[['ID','ds','y','yhat1','yhat2','yhat3','yhat4']]
yhat_columns = ['yhat1','yhat2','yhat3','yhat4'] # modificare se orizzonte diverso
forecast_np_1[yhat_columns] = forecast_np_1[yhat_columns].clip(lower=0)
forecast_np_1['y'] = forecast_np_1[['yhat1', 'yhat2', 'yhat3', 'yhat4']].bfill(axis=1).iloc[:, 0]
forecast_np_1 = forecast_np_1[['ID','ds','y']]
forecast_np_2 = forecast_np.merge(forecast_np_1, on=['ID','ds'], how='left')

def grafico_actual_forecast(df, item):
    fig = px.bar(df[df['ID'] == item], x='ds', y=['y_x','y_y'], title=item, labels={'y_x': 'Actual', 'y_y': 'Forecast'}) #template='plotly_dark'
    for trace in fig.data:
        if trace.name == 'y_x':
            trace.name = 'Actual'
        elif trace.name == 'y_y':
            trace.name = 'Forecast'
                  
    st.plotly_chart(fig, use_container_width=True)

grafico_actual_forecast(forecast_np_2, codice)

# Crea il bottone per scaricare forecast_np_2
forecast_data_2 = to_excel_bytes(forecast_np_2)
st.download_button(
    label="📥 Scarica Forecast",
    data=forecast_data_2,
    file_name='forecast_test_np_2.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)


st.stop()


