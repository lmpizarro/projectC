import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import numpy as np
import numpy_financial as npf
from datetime import date
from bonds import ytm_discrete, ytm_continuous, m_duration
import json

N_DAYS = 360

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
    df['T'] = df['T'].dt.days.astype('int16') / N_DAYS

    return df

tickers = ['AL29D', 'AL30D', 'AE38D', 'AL41D', 'AL35D', 'GD29D', 'GD30D', 'GD46D', 'GD38D', 'GD35D']

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
    
