from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import copy
import json

urls = {"nasdaq100": "https://www.slickcharts.com/nasdaq100",
        "dowjones": "https://www.slickcharts.com/dowjones",
        "sp500": "https://www.slickcharts.com/sp500",
        "sectors": "https://topforeignstocks.com/indices/components-of-the-sp-500-index",
        "cedears": "https://www.rava.com/cotizaciones/cedears",
        "bonos_rava": "https://www.rava.com/perfil",
        "bonistas_com": "https://bonistas.com"
        }

def scrap_bonos_rava(especie):
    url = f"{urls['bonos_rava']}/{especie}"
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='html.parser')
    table = soup.find('main').find('perfil-p')

    res = json.loads(table.attrs[':res'])
    return res


def scrap_bonistas_ticker(especie):
    url = f"{urls['bonistas_com']}/md/{especie}"
    """
       https://towardsdatascience.com/a-guide-to-scraping-html-tables-with-pandas-and-beautifulsoup-7fc24c331cf7
    """
    dfs = pd.read_html(url)

    for df in dfs:
        print(df)

def scrap_bonistas_main():
    url = f"{urls['bonistas_com']}"

    dfs = pd.read_html(url)
    tickers_index = [0, 2, 4, 8, 10]
    maps = {0:'tasa fija badlar',
            2:'CER',
            4:'USD',
            6: 'CABLE',
            8: 'LEDES',
            10: 'LECER',
            12: 'MEP CCL ARG',
            13: 'MEP CCL NY'}
    tickers = []
    bonos = {}
    for index, df in enumerate(dfs):
        if index < 11 and not (index % 2):
            bonos[index] = df
            if index in tickers_index:
                tickers.extend(list(df.Ticker))
        elif index > 11 and index <= 13:
            bonos[index] = df
            if index in tickers_index:
                tickers.extend(list(df.Ticker))

    for index in bonos:
        print(f'-- {index} ---------- {maps[index].upper()} --------')
        print(bonos[index])

    print(tickers)


def scrap_slick_chart(url, constituents) -> Dict[str, Any]:
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='lxml')
    table = soup.find('table')
    for row in table.findAll('tr')[1:]:
        tds = row.findAll('td')
        ticker = tds[1]
        weight = float(tds[3].text)
        price = float(''.join(tds[4].text.strip().split(',')))
        ticker = ticker.a.get('href').split('/')[2]
        if ticker not in constituents:
            constituents[ticker] = {}
        constituents[ticker]['weight'] = weight
        constituents[ticker]['price'] = price
    return constituents

def list_sp500():
    return scrap_slick_chart(urls['sp500'], {}).keys()

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


def scrap_finviz(constituents, max_n=10):
    N = 0
    for ticker in copy.deepcopy(constituents):
        print(ticker)
        N += 1
        try:
            url = f'https://finviz.com/quote.ashx?t={ticker}'
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.text, features='lxml')
            table = soup.find('table', {'class': 'snapshot-table2'})
            for row in table.findAll('tr')[0:]:
                tds = row.findAll('td')
                for i in range(0, len(tds), 2):
                    k = tds[i].text
                    v = tds[i+1].text
                    if k not in constituents[ticker]:
                        constituents[ticker][k] = v
        except AttributeError as err:
            del constituents[ticker]
            print(err)
        if N == max_n:
            break

    return constituents



def scrap_sp500(folder:str, file_name:str, max_n:int = 10):
    url_sectors = urls['sectors']
    url_sp500 = urls['sp500']


    resp = requests.get(url_sectors, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='lxml')
    table = soup.find('table')

    constituents = {}
    for row in table.findAll('tr')[1:]:
        tds = row.findAll('td')
        sector = tds[3].text
        ticker = tds[2].text
        name   = tds[1].text
        constituents[ticker] = {'sector': sector, 'name': name}

    resp = requests.get(url_sp500, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='lxml')
    table = soup.find('table')
    for row in table.findAll('tr')[1:]:
        tds = row.findAll('td')
        ticker = tds[1]
        weight = float(tds[3].text)
        price = float(''.join(tds[4].text.strip().split(',')))
        ticker = ticker.a.get('href').split('/')[2]
        constituents[ticker]['weight'] = weight
        constituents[ticker]['price'] = price

    for ticker in copy.deepcopy(constituents):
        if '.' in ticker:
            new_ticker = ticker.replace('.', '-')
            constituents[new_ticker] = constituents.pop(ticker)
            ticker = new_ticker

    nasdaq100 = scrap_slick_chart(urls['nasdaq100'], {})
    cedears = scrap_cedear_rava()

    N = 0

    for ticker in constituents:
        N += 1
        print(ticker, N, ticker in cedears)
        try:
            url = f'https://finviz.com/quote.ashx?t={ticker}'
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.text, features='lxml')
            table = soup.find('table', {'class': 'snapshot-table2'})
            for row in table.findAll('tr')[0:]:
                tds = row.findAll('td')
                for i in range(0, len(tds), 2):
                    k = tds[i].text
                    v = tds[i+1].text
                    if v == 'S&P 500':
                        v = 'S&P500'
                    if k == 'Index' and ticker in nasdaq100:
                        v = f'{v} NDQ100'

                    if k not in constituents[ticker]:
                        constituents[ticker][k] = v
        except AttributeError as err:
            print(err)
        constituents[ticker]['cedear'] = 1 if ticker in cedears else 0
        if N == max_n:
            break

    df = pd.DataFrame.from_dict(constituents, orient='index')
    file_path = folder / file_name

    df.to_csv(file_path)

def read_csv(folder:str, file_name:str):
    file_path = folder / file_name
    df = pd.read_csv(file_path)
    df.rename( columns={'Unnamed: 0':'ticker'}, inplace=True )
    return df

def filter_df(folder:str, file_name:str):
    df = read_csv(folder, file_name)

    rslt_df = df[df['Dividend'] != '-']
    rslt_df = df[df['weight'] > .5]

    print(len(rslt_df))
    for i in range(0, 10):
        rslt_df = rslt_df[rslt_df['Dividend %'] > f'{i}%']
        print(i, len(rslt_df))

def filter_df1(folder:str, file_name:str):
    df = read_csv(folder, file_name)

    def tr_pct(x):
        if type(x) == str:
            try:
                x = float(x[:-1])
            except Exception:
                pass
        return x

    def tr_str(x):
        try:
            x = float(x)
        except Exception:
            pass
        return x

    df['Perf YTD'] = df['Perf YTD'].transform(tr_pct)
    df['Perf Year'] = df['Perf Year'].transform(tr_pct)
    df['Change'] = df['Change'].transform(tr_pct)
    df['SMA200'] = df['SMA200'].transform(tr_pct)
    df['SMA50'] = df['SMA50'].transform(tr_pct)
    df['SMA20'] = df['SMA20'].transform(tr_pct)
    df['Dividend %'] = df['Dividend %'].transform(tr_pct)
    df['Dividend'] = df['Dividend'].transform(tr_str)
    df['P/E'] = df['P/E'].transform(tr_str)
    df['Beta'] = df['Beta'].transform(tr_str)
    # df['Recom'] = df['Recom'].transform(tr_str)

    rslt_df = df[df['cedear'] == 1]
    # rslt_df = rslt_df[rslt_df['Perf YTD'] > 0]
    rslt_df = rslt_df[rslt_df['Recom'] < "2.0"]
    # rslt_df = rslt_df[rslt_df['Dividend'] != '-']
    # rslt_df = rslt_df[rslt_df['Dividend'] > 0]
    # rslt_df = rslt_df[rslt_df['Dividend %'] > 1.0]
    # rslt_df = rslt_df[rslt_df['weight'] > .1]
    # rslt_df = rslt_df[rslt_df['SMA200'] < 0 ]
    # rslt_df = rslt_df[rslt_df['P/E'] < 15 ]
    rslt_df = rslt_df[rslt_df['Beta'] > .8 ]
    rslt_df = rslt_df[rslt_df['Beta'] < 1.4 ]
    # rslt_df = rslt_df[rslt_df['SMA50'] < 0 ]

    print(rslt_df.head())
    print(len(rslt_df))
    print(rslt_df['ticker'])


from pathlib import Path

def main():
    p = Path(__file__)
    p = p.parent.parent / "data"

    p.mkdir(exist_ok=True)

    file_name = 'sp500-2022-08-13.csv'

    scrap_sp500(p, file_name, max_n=1000)

    # filter_df1(p, file_name)

def cedear_not_in_sp500(max_n=10):
    cedears = scrap_cedear_rava()
    sp500 = scrap_slick_chart(urls['sp500'], {}).keys()
    cedear_not_sp500 = {}
    for t in cedears:
        if t not in sp500:
            cedear_not_sp500[t] = {}

    df = pd.DataFrame.from_dict(scrap_finviz(cedear_not_sp500, max_n=max_n), orient='index')
    return df

def test01():
    main()
    ce = scrap_cedear_rava()
    print(len(ce))
    print(ce)

    print('\n')
    print('\n')
    c = cedear_not_in_sp500(max_n=10)

    print(c.tail())

if __name__ == '__main__':
    scrap_bonistas_main()