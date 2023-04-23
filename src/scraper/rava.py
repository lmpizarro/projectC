from config import urls
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

def scrap_bonos_rava(especie):
    url = f"{urls['bonos_rava']}/{especie}"
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='html.parser')
    table = soup.find('main').find('perfil-p')

    res = json.loads(table.attrs[':res'])
    return res


def scrap_cedear_rava():
    url = urls['cedears']
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='html.parser')
    table = soup.find('main').find('cedears-p')

    body = json.loads(table.attrs[':datos'])['body']
    symbolos = []
    for b in body:
        symbolos.append(b['simbolo'])
    return symbolos


import pandas as pd

def dolares(tipo='CCL', desde='2020-04-20', hasta='2023-04-23'):
    if tipo == 'MAY':
        # Mayorista ordenado de menor a mayor comienza 07 03 2013
        url = f'https://mercados.ambito.com//dolar/mayorista/grafico/{desde}/{hasta}'
    elif tipo == 'CCL':
        # CCL ordenado de mayor a menor comienza 07 03 2013
        url = f'https://mercados.ambito.com/dolarrava/cl/historico-general/{desde}/{hasta}'
    elif tipo == 'MEP':
        # MEP ordenado de mayor a menor comienza 24 03 2020
        url = f'https://mercados.ambito.com/dolarrava/mep/historico-general/{desde}/{hasta}'
    elif tipo == 'OFI':
        url = f'https://mercados.ambito.com/dolar/oficial/grafico/{desde}/{hasta}'
    elif tipo == 'NAC':
        url = f'https://mercados.ambito.com/dolarnacion/historico-general/{desde}/{hasta}'



    df_dolar = pd.read_json(url)
    if tipo == 'NAC':
        df_dolar = df_dolar[1:]
        df_dolar = df_dolar.apply(lambda x: x.str.replace(',','.'))
        df_dolar[[1,2]] = df_dolar[[1, 2]].astype('float64')
        df_dolar[1] = (df_dolar[1] + df_dolar[2]) / 2
        df_dolar = df_dolar[[0,1]]

    if tipo == 'CCL' or tipo == 'MEP':
        df_dolar = df_dolar.reindex(index=df_dolar.index[::-1]).reset_index()
        df_dolar = df_dolar.drop(columns=['index'])
        df_dolar = df_dolar.apply(lambda x: x.str.replace(',','.'))
        df_dolar = df_dolar[:-1]
        df_dolar[[1]] = df_dolar[[1]].astype('float64')


    if tipo ==  'MAY' or tipo == 'OFI':
        df_dolar = df_dolar[1:]
    elif tipo == 'MEP' or tipo == 'CCL':
        df_dolar = df_dolar[:-1]

    df_dolar = df_dolar.rename(columns={0: 'fecha', 1: tipo})

    return df_dolar

# df = dolares(tipo='OFI')
# print(df.tail())
# df = dolares(tipo='NAC')
# print(df.tail())
# df = dolares(tipo='MEP')
# print(df.tail())
def dolar_may_ccl(hasta='2023-04-20'):
    df_may = dolares(tipo='MAY', hasta=hasta)
    # print(df_may.tail())
    df_ccl = dolares(tipo='CCL', hasta=hasta)
    # print(df_ccl.tail())

    results = df_may.merge( df_ccl, on='fecha', how='left')
    results.dropna(inplace=True)
    results['brecha'] = (results['CCL'] - results['MAY']) / results['MAY']
    results['brecha'] = pd.to_numeric(results['brecha'])
    return results

def test01():
    dol_may_ccl = dolar_may_ccl()
    print(dol_may_ccl.brecha.min(), dol_may_ccl.brecha.max(), dol_may_ccl.brecha.mean())
    min_arg = dol_may_ccl.brecha.argmin(skipna=True)
    max_arg = dol_may_ccl.brecha.argmax(skipna=True)

    print('min ', dol_may_ccl.iloc[min_arg].fecha)
    print('max ', dol_may_ccl.iloc[max_arg].fecha)

    # plt.plot(dol_may_ccl.brecha)
    # plt.plot(dol_may_ccl.brecha.rolling(60).mean())
    # plt.show()

def variables_bcra(tipo='cer', desde='2016-04-20'):
    url = "https://www.bcra.gob.ar/PublicacionesEstadisticas/Principales_variables_datos.asp"
    settings = {
        'cer': {"Serie": "3540", "Detalle": "CER (Base 2.2.2002=1)"},
        'badlar': {"Serie": "7935", "Detalle": "BADLAR en pesos de bancos privados (en  e.a.)"},
        'TEAPolMon': {"Serie": "7936", "Detalle": "Tasa de Política Monetaria (en  e.a.)"},
        'mayorista': {"Serie": "272", "Detalle": "Tipo de Cambio Mayorista ($ por US$) Comunicación A 3500 - Referencia"},
        'TEAPF': {"Serie": "7939", "Detalle": "Tasa mínima para plazos fijos de personas humanas hasta $10 millones (en  e.a. para depósitos a 30 días)"}
    }
    today = datetime.now().date()
    month = f'0{today.month}' if today.month >= 1 and today.month <= 9 else f'{today.month}'
    hasta = f'{today.year}-{month}-{today.day}'
    data = {"primeravez": "1",
            "fecha_desde": desde,
            "fecha_hasta": hasta,
            "serie": settings[tipo]['Serie'],
            "series1": "0",
            "series2": "0",
            "series3": "0",
            "series4": "0",
            "detalle": settings[tipo]['Detalle']
            }
    resp = requests.post(url=url, data=data, headers={'User-Agent': 'Mozilla/5.0'})

    r_text = resp.text

    df_cer = pd.read_html(r_text, thousands='.')[0]

    df_cer = df_cer.apply(lambda x: x.str.replace(',','.'))
    df_cer[['Valor']] = df_cer[['Valor']].astype('float64')
    df_cer.set_index('Fecha', inplace=True)
    return df_cer

df_cer = variables_bcra('cer')
print(df_cer.head())
print(df_cer.tail())


df_cer = variables_bcra('mayorista')
print(df_cer.head())
print(df_cer.tail())

import yfinance as yf

tickers = ['GGAL', 'GGAL.BA', 'AAPL.BA', 'AAPL', 'ARS=X']

df_close = yf.download(tickers, start="2018-04-20", auto_adjust=True)['Close']
df_close['ccl1'] = 10 * df_close['GGAL.BA'] / df_close['GGAL']
df_close['ccl2'] = 10 * df_close['AAPL.BA'] / df_close['AAPL']
df_close['ccl'] = .5*(df_close.ccl1 + df_close.ccl2)
df_close = df_close[['ccl', 'ARS=X']]
df_close['gap'] = (df_close['ccl'] - df_close['ARS=X']) / df_close['ARS=X']
df_close.dropna(inplace=True)

print(df_close.head())
print(df_close.tail())
import matplotlib.pyplot as plt

plt.plot(df_close.gap)
plt.show()
