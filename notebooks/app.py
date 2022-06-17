# import modules
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import joblib
import xgboost as xgb
from streamlit_kpi_metric import metric, metric_row

# page config
st.set_page_config(page_title='Parcels Forecasting',layout='wide', initial_sidebar_state="expanded",)
st.title("Parcels Forecasting")

# urls
url_past_data = '../files/data/daytypelags2017_parcels.csv'
url_future_data = '../files/data/future.csv'
model = joblib.load('../files/save_model/XGB_model.sav')

# import data and declare variables
df_past = pd.read_csv(url_past_data, sep=",", index_col='date')
df_past.index = pd.to_datetime(df_past.index)
df_past = df_past[['date_c', 'n_parcels', 'typeDay']]
df_future = pd.read_csv(url_future_data, sep=",", index_col='date')
df_future.index = pd.to_datetime(df_future.index)
yhat = None

# plot function
def scatterPlot(start, end, df):
    df_scatterplot = df[(df.index >= dt.datetime.strptime(str(start), '%Y-%m-%d'))&(df.index <=  dt.datetime.strptime(str(end), '%Y-%m-%d'))]
    fig = plt.figure(figsize=(24, 8))
    p = sns.scatterplot(df_scatterplot.index, df_scatterplot['n_parcels'], hue=df_scatterplot['typeDay']).set( xlabel = "Date", ylabel = "Parcels")
    st.pyplot(fig)    

# show prediction function
def show_prediction(y_hat, df_f, df_p):
    if y_hat is not None:
      date = df_future.iloc[-1:].index[0]
      date2 = date + dt.timedelta(days=-1)
      increase = int(y_hat) - df_past.loc[str(date2)[0:10]]['n_parcels']
      st.write("# The prediction for the next day is")
      col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
      col5.metric(label="Parcels", value=str(int(y_hat)), delta = increase)
      df_week = df_past.iloc[-7:]['n_parcels']
      prediction = {'date':[str(date)[0:10]],'n_parcels':[int(y_hat)]}
      prediction = pd.DataFrame(prediction)
      prediction.index = prediction['date']
      prediction.drop('date', axis=1, inplace=True)
      prediction.index = pd.to_datetime(prediction.index)
      df_week_plot = pd.concat([df_week, prediction], axis=0)
      df_week_plot['nparcels'] = df_week_plot.fillna(0)[0] + df_week_plot.fillna(0)['n_parcels']
      df_week_plot['past_future'] = df_week_plot.apply(lambda x: 'past' if x[0] >= 0 else 'future', axis=1)
      fig = plt.figure(figsize=(24, 8))
      p1 = sns.lineplot(df_week_plot.index, df_week_plot['nparcels'], marker='o', 
                       hue=df_week_plot['past_future']).set(xlabel = "Date", ylabel = "Parcels")
      st.pyplot(fig) 

# show future data parameters function
def show_parameters(df):
    st.write('The characteristics of the next day are as follows:')
    if str(df.iloc[:1]['fest'].values) == '[True]':
      fest = 'Yes'
    else: fest = 'No'
    if str(df.iloc[:1]['saturday'].values) == '[True]' or str(df.iloc[:1]['sunday'].values) == '[True]':
      weekend = 'Yes'
    else: weekend = 'No'
    if str(df.iloc[:1]['blackFriday'].values) == '[True]':
      blackfriday = 'Yes'
    else: blackfriday = 'No'
    if str(df.iloc[:1]['COVID'].values) == '[True]':
      covid = 'Yes'
    else: covid = 'No'
    day = str(df.iloc[-1:].index[0])[0:11]
    metric_row(
        {
          "Next day": day
        }
    )
    metric_row(
        {
          "Holiday":  fest,
          "Weekend": weekend,
          "Black Friday": blackfriday,
          "COVID lockdown": covid,
        }
    )

# create sidebar
with st.sidebar:
    st.title('Options')
    with st.form(key='Historical plot'):
      d_ini = st.date_input("Start date", min_value=dt.datetime.strptime('2017-01-01', '%Y-%m-%d'), 
                            max_value=dt.datetime.strptime('2022-03-31', '%Y-%m-%d'), 
                            value=dt.datetime.strptime('2021-01-01', '%Y-%m-%d'))
      d_fin = st.date_input("End date", min_value=dt.datetime.strptime('2017-01-01', '%Y-%m-%d'), 
                            max_value=dt.datetime.strptime('2022-03-31', '%Y-%m-%d'), 
                            value=dt.datetime.strptime('2022-03-31', '%Y-%m-%d'))
      if d_ini > d_fin:
        st.error('Error: End date must be greater or equal than start date')
      submit_button_predict = st.form_submit_button(label='Upload plot')
      
    with st.form(key='Prediction'):
      st.write('Launch forecasting model')
      submit_button_execute = st.form_submit_button(label='Execute prediction')

# create prediction button
if submit_button_execute:
  df_future_pred = df_future.reset_index()
  df_future_pred.drop('date', axis=1, inplace=True)
  df_future_pred['fest'] = df_future_pred.apply(lambda x: 1 if x['fest'] else 0, axis=1)
  df_future_pred['saturday'] = df_future_pred.apply(lambda x: 1 if x['fest'] else 0, axis=1)
  df_future_pred['sunday'] = df_future_pred.apply(lambda x: 1 if x['fest'] else 0, axis=1)
  df_future_pred['blackFriday'] = df_future_pred.apply(lambda x: 1 if x['fest'] else 0, axis=1)
  df_future_pred['blackFridayWeek'] = df_future_pred.apply(lambda x: 1 if x['fest'] else 0, axis=1)
  df_future_pred['COVID'] = df_future_pred.apply(lambda x: 1 if x['fest'] else 0, axis=1)
  df_future_pred['Christmas'] = df_future_pred.apply(lambda x: 1 if x['fest'] else 0, axis=1)
  yhat = model.predict(df_future_pred)


# interface components

st.title('Historical data')

st.write('Number of parcels per day')
scatterPlot(d_ini, d_fin, df_past)

st.title('Predict')
st.write('Forecast number of parcels for the next day.')

show_parameters(df_future)

show_prediction(yhat, df_future, df_past)
