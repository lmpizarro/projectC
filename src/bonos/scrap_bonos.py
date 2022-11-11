import requests
import pickle
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import numpy as np
import numpy_financial as npf
from datetime import date
from bonds import ytm_discrete, ytm_continuous, m_duration
import json

DAYS_IN_A_YEAR = 360
ONE_BPS = 0.0001

bonos = {
            "GD29": {"pagos": [('09/01/23', 0.5, 0), ('09/07/23', 0.5, 0),
                               ('09/01/24', 0.5, 0), ('09/07/24', 0.5, 0),
                               ('09/01/25', 0.5, 10), ('09/07/25', 0.45, 10),
                               ('09/01/26', 0.4, 10), ('09/07/26', 0.35, 10),
                               ('09/01/27', 0.3, 10), ('09/07/27', 0.25, 10),
                               ('09/01/28', 0.2, 10), ('09/07/28', 0.15, 10),
                               ('09/01/29', 0.1, 10), ('09/07/29', 0.05, 10),
                               ], "pay_per_year":2},
            }


nombre_bonos = {'GD' :[30,35,38,41,46]}

def process_csv(nombre_bonos):
    for tipo in nombre_bonos:
        for anio in nombre_bonos[tipo]:
            df_35 = pd.read_csv(f"flujoFondos_{tipo}{anio}.csv")

            bono = {}
            pagos = []
            for row in df_35.iterrows():
                fecha = row[1]["Fecha de pago"].split('/')
                fecha = '/'.join([fecha[2], fecha[1], fecha[0][2:]])
                renta = row[1]["Renta"]
                amort = row[1]["Amortización"]
                pagos.append((fecha, renta, amort))
            bono['pagos'] = pagos
            bono["pay_per_year"] = 2
            ticker = row[1]['Ticker']
            print(ticker)
            bonos[ticker] = bono
            if anio == 30:
                a_ticker = f'AL{anio}'
                bonos[a_ticker] = bono.copy()
            elif anio == 38:
                a_ticker = f'AE{anio}'
                bonos[a_ticker] = bono.copy()
            elif anio == 35:
                a_ticker = f'AL{anio}'
                bonos[a_ticker] = bono.copy()
            elif anio == 41:
                a_ticker = f'AL{anio}'
                bonos[a_ticker] = bono.copy()

    with open('bonos.pkl', 'wb') as fp:
        pickle.dump(bonos, fp)

def scrap_bonos_rava():
    url = "https://www.rava.com/cotizaciones/bonos"

    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='html.parser')
    table = soup.find('main').find('bonos-p')
    
    body = json.loads(table.attrs[':datos'])['body']
    # print(body[0].keys())
    my_keys = ['simbolo', 'ultimo']
    my_body = []
    for b in body:
        my_body.append({my_key: b[my_key] for my_key in my_keys})

    symbols = {}
    for e in my_body:
        simbolo = e['simbolo']
        symbols[simbolo] = {k:e[k] for k in e if k !='simbolo'}

    return symbols


def scrap_cash_flows(ticker):
    url = f"https://bonistas.com/md/{ticker}"

    # Create object page
    page = requests.get(url)


    # parser-lxml = Change html to Python friendly format
    # Obtain page's information
    soup = BeautifulSoup(page.text, 'lxml')


    div = soup.find('div', class_='col-lg-6 d-none d-lg-block d-xl-block')
    table = div.find('table')

    rows = table.findChildren(['th', 'tr'])

    header = ["FECHA", "SALDO", "CUPÓN", "AMORTIZACIÓN", "TOTAL"]

    csv = ','.join(header) + '\n'
    for row in rows:
        cells = row.findChildren('td')
        line = ''
        for cell in cells:
            value = cell.string
            line += "%s," % value
        line = line[:-1]
        if line != '':
            csv += "%s\n" % line
    csv += "\n"

    csv_file = StringIO(csv)
    df = pd.read_csv(csv_file)
    
    today = [pd.Timestamp(date.today())] * len(df)
    to_day = pd.Series(today)
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df['T'] = (df['FECHA'] - to_day)
    df['T'] = df['T'].dt.days.astype('int16') / DAYS_IN_A_YEAR

    return df

if __name__ == '__main__':
    tickers = ['AL29D', 'AL30D', 'AE38D', 'AL41D', 'AL35D', 'GD29D', 'GD30D', 'GD46D', 'GD38D', 'GD35D', 'GD41D']

    print('tkr', '  precio', '  tir', '  m_dur', '  cash', '  amort', '  cupon')
    for ticker in tickers:
        close_day = scrap_bonos_rava()
        price = float(close_day[ticker]['ultimo'])


        df = scrap_cash_flows(ticker)
        tir_ = ytm_discrete(df, price)

        total_amort = df['AMORTIZACIÓN'].sum()
        total_cupo = df['CUPÓN'].sum()

        total = total_amort + total_cupo

        duration = m_duration(df, tir_)
        print(f'{ticker},  {round(price,2)},  {round(tir_, 2)},  {round(duration, 2)},  ' 
              f'{round(total,2)},  {round(total_amort,2)},  {round(total_cupo,2)}')
    
