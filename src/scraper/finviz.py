import requests
import copy
from bs4 import BeautifulSoup
import pandas as pd
from collections import namedtuple

Section = namedtuple("Sections", "url section")

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


        return df


    @staticmethod
    def generate_urls_scrap_finviz():

        index = [111, 121, 161, 131, 141, 171]
        sections = ["overview", "valuation", "financial", "ownership", "performance", "technical"]

        index = [121, 161, 131, 141, 171]
        sections = ["valuation", "financial"]
        index_to_sections = dict(zip(index, sections))

        company_sizes = {'mega': 21, 'large': 181, 'mid': 81, 'small': 81}

        urls = []
        for company_size in company_sizes:
            for page in range(1, company_sizes[company_size] + 1, 20):
                section_pages = []
                for index_section in index_to_sections:
                    section = index_to_sections[index_section]
                    url = f'https://finviz.com/screener.ashx?v={index_section}&f=cap_{company_size}&ft=4&o=-marketcap&r={page}'
                    section_pages.append(Section(url, section))

                urls.append(section_pages)

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
    df = pd.DataFrame()
    for sections in urls:
        for section in sections:
            if section.section == 'valuation':
                keys = ['Ticker', 'P/E', 'P/B', 'P/S', 'PEG']
                df1 = FinViz.get_df_screener(section.url)[keys]
                df1.set_index('Ticker', inplace=True)
            else:
                keys = ['Ticker', 'Dividend', 'Debt/Eq']
                df2 = FinViz.get_df_screener(section.url)[keys]
                df2.set_index('Ticker', inplace=True)


        df3 = df1.join(df2)
        df = pd.concat([df,df3])
    df.to_csv('factors.csv')