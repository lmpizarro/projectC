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
for ticker in Data_BONO.Ticker:
    today = datetime.now()
    iD = ticker
    multiplier = 1 / usdars
    if ticker[-1] in ['C', 'D']:
        sufix = ticker[-1]
        ticker = ticker[:-1]
        iD = ticker + sufix
        multiplier = 1
    CF = pd.read_excel('cashflows_BONO.xls', sheet_name=ticker)

    CF = CF.replace(',', '.', regex=True)
    CF.Renta = pd.to_numeric(CF.Renta)
    CF['Amortizacion'] = pd.to_numeric(CF['Amortización'])
    CF.drop(columns=['Amortización'], inplace=True)
    CF = CF[CF['Fecha de pago'] >= today]
    cashflows[iD] = CF

    total_cash_flow = cashflows[iD].Renta.sum() +  cashflows[iD].Amortizacion.sum()
    precio = multiplier * precios_BONO.loc[iD].Precio 
    yild =  total_cash_flow / precio

    time_to_maturity = (CF.iloc[-1]['Fecha de pago'] - CF.iloc[0]['Fecha de pago']).days / 365
    time_to_maturity = (CF.iloc[-1]['Fecha de pago'] - today).days  / 365

    print(f'{iD} ', round(precio, 2), 
            round(total_cash_flow,2), 
            round(100 *  yild / time_to_maturity, 2), 
            round(total_cash_flow/time_to_maturity,2))

