import requests
from bs4 import BeautifulSoup
import pandas as pd
import copy


def scrap_sp500(folder:str, file_name:str, max_n:int = 10):
    url1 = 'https://topforeignstocks.com/indices/components-of-the-sp-500-index/'
    url2 = 'https://www.slickcharts.com/sp500'


    resp = requests.get(url1, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='lxml')
    table = soup.find('table')

    constituents = {}
    for row in table.findAll('tr')[1:]:
        tds = row.findAll('td')
        sector = tds[3].text
        ticker = tds[2].text
        name   = tds[1].text

        constituents[ticker] = {'sector': sector, 'name': name}
    
    resp = requests.get(url2, headers={'User-Agent': 'Mozilla/5.0'})
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
                    if k not in constituents[ticker]:
                        constituents[ticker][k] = v
        except AttributeError as err:
            print(err)
        if N == max_n:
            break

    df = pd.DataFrame.from_dict(constituents, orient='index')

    print(df.head())
    file_path = folder + '/' + file_name

    df.to_csv(file_path)

def read_csv(folder:str, file_name:str):
    file_path = folder + '/' + file_name
    df = pd.read_csv(file_path)
    df.rename( columns={'Unnamed: 0':'ticker'}, inplace=True )
    return df

def filter_df(folder:str, file_name:str):
    df = read_csv(folder:str, file_name:str)

    rslt_df = df[df['Dividend'] != '-']
    rslt_df = df[df['weight'] > .5]

    print(len(rslt_df))
    for i in range(0, 10):
        rslt_df = rslt_df[rslt_df['Dividend %'] > f'{i}%']
        print(i, len(rslt_df))

