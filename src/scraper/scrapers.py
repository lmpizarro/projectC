import requests
from bs4 import BeautifulSoup
import pandas as pd
import copy

urls = {"nasdaq100": "https://www.slickcharts.com/nasdaq100",
        "dowjones": "https://www.slickcharts.com/dowjones", 
        "sp500": "https://www.slickcharts.com/sp500",
        "sectors": "https://topforeignstocks.com/indices/components-of-the-sp-500-index"}


def scrap_slick_chart(url, constituents):
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

    N = 0
    for ticker in constituents:
        N += 1
        print(ticker, N)
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

from pathlib import Path

if __name__ == '__main__':
    p = Path(__file__)
    p = p.parent.parent / "data"

    p.mkdir(exist_ok=True)

    file_name = 'sp500-2022-07-30b.csv'

    scrap_sp500(p, file_name, max_n=1000)

    filter_df(p, file_name)    