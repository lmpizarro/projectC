import pandas as pd
import numpy as np
from datetime import datetime

url = 'https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos'


Data_BONO = pd.read_excel('cashflows_BONO.xls', sheet_name='DATA')
BONO = pd.read_html(url, thousands='.', decimal=',')[0]
BONO['Ticker']=BONO['Símbolo']
BONO['Precio']=pd.to_numeric(BONO['ÚltimoOperado'])
BONO['Vol']=pd.to_numeric(BONO['MontoOperado'])
precios_BONO = BONO[['Ticker', 'Precio', 'Vol']]
precios_BONO.set_index('Ticker', inplace=True)

Data_BONO = pd.read_excel('cashflows_BONO.xls', sheet_name='DATA')

usdars = 323
try:
    usdars = float(input(f'input usdars ({usdars}): '))
except:
    pass
cashflows={}
out_df = pd.DataFrame()
list_ars = []
from datetime import datetime


for ticker in Data_BONO.Ticker:
    today = datetime.now()
    iD = ticker
    multiplier = 1 / usdars
    if ticker[-1] in ['C', 'D']:
        sufix = ticker[-1]
        ticker = ticker[:-1]
        iD = ticker + sufix
        multiplier = 1
    else:
        list_ars.append(ticker)

    CF = pd.read_excel('cashflows_BONO.xls', sheet_name=ticker, parse_dates=['Fecha de pago'], date_parser=pd.to_datetime)
    CF = CF.replace(',', '.', regex=True)
    CF.Renta = pd.to_numeric(CF.Renta)
    CF['Amortizacion'] = pd.to_numeric(CF['Amortización'])
    CF.drop(columns=['Amortización'], inplace=True)
    CF = CF[CF['Fecha de pago'] >= today]
    cashflows[iD] = CF

    total_cash_flow = cashflows[iD].Renta.sum() +  cashflows[iD].Amortizacion.sum()
    precio = multiplier * precios_BONO.loc[iD].Precio 
    yild =  (total_cash_flow - precio) / precio

    time_to_maturity = (CF.iloc[-1]['Fecha de pago'] - CF.iloc[0]['Fecha de pago']).days / 365
    time_to_maturity = (CF.iloc[-1]['Fecha de pago'] - today).days  / 365

    d = {'Ticker': iD, 'precio':precio, 'maturity': time_to_maturity,
         'cashFlow': total_cash_flow, 'tasaPromedio': yild / time_to_maturity, 
         'cashPromedio': total_cash_flow/time_to_maturity}

    df_d = pd.DataFrame([d])
    out_df = pd.concat([out_df, df_d], ignore_index=True)

out_df.set_index('Ticker', inplace=True)
out_df['pctlinDur'] = out_df.precio / out_df.cashPromedio / out_df.maturity
out_df['linDur'] = out_df.precio / out_df.cashPromedio
print(out_df.loc[list_ars].sort_values(by='tasaPromedio'))

tickers = ['AL41', 'AL30', 'AE38', 'GD46']

df_pays = cashflows['AL29']
df_pays['Total'] = df_pays[['Renta', 'Amortizacion']].sum(axis=1)
df_pays.drop(columns=['Renta', 'Amortizacion'], inplace=True)

for ticker in tickers:
    cf = cashflows[ticker]
    cf = cf[cf['Fecha de pago'] > today]
    cf['Total'] = cf[['Renta', 'Amortizacion']].sum(axis=1)
    cf.drop(columns=['Renta', 'Amortizacion'], inplace=True)
    df_pays = cf.join(df_pays.set_index('Fecha de pago'), on='Fecha de pago', lsuffix=f'{ticker}') 
    df_pays.fillna(0, inplace=True)
    df_pays['Total'] = df_pays.Total + df_pays[f'Total{ticker}']
    df_pays.drop(columns=[f'Total{ticker}'], inplace=True)


import matplotlib.pyplot as plt
df_pays.set_index('Fecha de pago', inplace=True)
plt.plot(df_pays.Total, 'x-')
plt.show()
