import requests
import copy
from bs4 import BeautifulSoup
import pandas as pd

class FinViz:

    @staticmethod
    def get_df_screener(url: str):
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text, features='lxml')
        table = soup.findAll('table')
        df = pd.read_html(str(table))[-2]
        maper = {}
        for i,e in enumerate(df.iloc[0]):
            maper[df.keys()[i]] = e

        df.rename(columns=maper, inplace=True)
        df.drop(0, inplace=True)

        print(df.head())


    @staticmethod
    def generate_urls_scrap_finviz():
        index = [111, 121, 161, 131, 141, 171]
        sub_title = ["overview", "valuation", "financial", "ownership", "performance", "technical"]
        index_view = dict(zip(index, sub_title))

        capital = {'mega': 21, 'large': 81, 'mid': 81, 'small': 81}

        urls = []
        for j in capital:
            for i in range(1, capital[j] + 1, 20):
                for k in index_view:
                    url = f'https://finviz.com/screener.ashx?v={k}&f=cap_{j}&ft=4&o=-marketcap&r={i}'
                    urls.append(url)

        return urls

    @staticmethod
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



if __name__ == '__main__':
    urls = FinViz.generate_urls_scrap_finviz()
    for url in urls:
        FinViz.get_df_screener(url)