import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import nct
from scipy.stats import norm
from scipy.stats import cauchy
from collections import namedtuple

CDF = namedtuple("CDF", "x cdf pdf")
Describe = namedtuple('Describe', 'min max sum mean std loc scale')


stocks = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BRK-B', 'TSLA', 'META', 'JNJ',
        'V', 'TSM', 'XOM', 'UNH', 'WMT', 'JPM', 'MA', 'PG', 'LLY', 'CVX', 'HD',
        'ASML', 'ABBV', 'GLOB', 'MELI']

stocks = stocks

class Downloader:
    def __init__(self, start='2017-01-01', stocks=stocks) -> None:
        self.stocks = stocks
        self.start = start
        self.ewm: pd.DataFrame
        self.log_returns: pd.DataFrame
        self.ewm_log_returns: pd.DataFrame
        self.df_distance: pd.DataFrame

    def download(self):
        self.yf_data = yf.download(self.stocks, start=self.start, auto_adjust=True)['Close']
        self.yf_data.dropna(inplace=True)

    def calc_log_return(self):
        self.log_returns = pd.DataFrame()
        for stock in self.stocks:
            self.log_returns[stock] = np.log(self.yf_data[stock]/self.yf_data[stock].shift(1))
        self.log_returns.dropna(inplace=True)

    def calc_ewma(self):
        self.ewm = pd.DataFrame()
        for stock in self.stocks:
            self.ewm[stock] = self.yf_data[stock].ewm(alpha=.9).mean()

    def calc_ewm_log_returns(self):
        self.ewm_log_returns = pd.DataFrame()
        for stock in self.stocks:
            self.ewm_log_returns[stock] = self.log_returns[stock].ewm(alpha=.9).mean()

    def calc_dist(self) -> pd.DataFrame:
        self.df_distance = pd.DataFrame()
        for stock in self.stocks:
            close = self.yf_data[stock]
            log = np.log(close/close.shift(1))
            log.dropna(inplace=True)

            mean_log = log.ewm(alpha=.9).mean()
            mean_log_2 = mean_log * mean_log
            log2_ewm = (log*log).ewm(alpha=.9).mean()
            dist = (log - mean_log) / np.sqrt(log2_ewm - mean_log_2)
            dist.dropna(inplace=True)
            self.df_distance[stock] = dist

        self.df_distance = np.sqrt(2*(1-self.df_distance.corr()))

    def returns_gt_threshold(self, pct_loc=1, pct_scale=2):
        dict_gt = {}
        for stock in self.stocks:
            loc, scale = norm.fit(data=self.log_returns[stock])
            d = self.log_returns[stock][self.log_returns[stock] >= pct_loc*loc + pct_scale*scale]
            dict_gt[stock] = Downloader.describe(d, loc=loc, scale=scale)
        return dict_gt

    @staticmethod
    def describe(d: pd.DataFrame, loc, scale):
        return Describe(max=d.max(), mean=d.mean(), sum=d.sum(), min=d.min(), std=d.std(), loc=loc, scale=scale)

    def returns_lt_threshold(self, pct_loc=1, pct_scale=2):
        dict_lt = {}
        for stock in self.stocks:
            loc, scale = norm.fit(data=self.log_returns[stock])
            d = self.log_returns[stock][self.log_returns[stock] <= pct_loc*loc - pct_scale*scale]
            dict_lt[stock] = Downloader.describe(d, loc=loc, scale=scale)
        return dict_lt

    def returns_in_range(self, min=0, max=2):
        dict_in_range = {}
        for stock in self.stocks:
            loc, scale = norm.fit(data=self.log_returns[stock])
            d = \
                self.log_returns[stock][(self.log_returns[stock] >= loc - min*scale) & (self.log_returns[stock] <= loc + max*scale)]
            dict_in_range[stock] = Downloader.describe(d, loc=loc, scale=scale)
        return dict_in_range


    @staticmethod
    def create_histogram(yf_data, bins=252):
        """Return an histogram """
        count, bins = np.histogram(yf_data, bins=bins)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        return CDF(bins[1:], cdf, count)

    def create_histograms(self):
        dict_hist = {}
        for stock in self.stocks:
            dict_hist[stock] = Downloader.create_histogram(self.log_returns[stock])
        return dict_hist

def distance_to_spy(distance):
    weights = []
    tickers = list(distance.keys())
    for i, k in enumerate(distance.keys()):
        for j in range(i+1, len(distance.keys())):
            c = k
            r = distance.keys()[j]
            weights.append((r, c, distance.loc[r][c]))


    weights.sort(key=lambda a: a[2], reverse=True)
    dist = {}
    for w in weights:
        if w[0] == 'SPY' or w[1] == 'SPY':
            dist[w[0]] = [1/w[2]]

    dist = pd.DataFrame(dist).T

    dist.rename(columns={0: 'dist'}, inplace=True)

    return dist


def test():

    dwldr = Downloader()

    yf_data = dwldr.download()

    print(dwldr.yf_data.head())

    dwldr.calc_ewma()
    dwldr.calc_log_return()
    dwldr.calc_ewm_log_returns()
    dwldr.calc_dist()
    gt = dwldr.returns_gt_threshold()
    lt = dwldr.returns_lt_threshold()
    in_r = dwldr.returns_in_range()
    hist = dwldr.create_histograms()

    # print(dwldr.ewm.head())
    # print(dwldr.log_returns.head())
    # print(dwldr.ewm_log_returns.head())
    # print(dwldr.df_distance.head())
    # print(gt.keys())
    factors = []
    for k in lt:
        factor = {}
        factor['ticker'] = k
        factor['lt'] = np.abs(lt[k].mean)
        factor['gt'] = gt[k].mean
        factor['in_range'] = in_r[k].mean
        factor['loc'] = gt[k].loc
        factor['scale'] = gt[k].scale
        factors.append(factor)

    # print(hist['AAPL'].pdf, hist['AAPL'].cdf, hist['AAPL'].x)
    df = pd.DataFrame(factors)
    # df[['a', 'b', 'c']] = df[['lt', 'gt', 'in_range']] - df.loc[0][['lt', 'gt', 'in_range']]
    df['lt'] = (1/df['lt'])/(1/df['lt']).sum()
    df['scale'] = (1/df['scale'])/(1/df['scale']).sum()
    df['gt'] = (df['gt'])/(df['gt']).sum()
    df['in_range'] = (df['in_range'])/(df['in_range']).sum()
    df['loc'] = (df['loc'])/(df['loc']).sum()

    df.drop(index=0, axis=1, inplace=True)
    df.set_index('ticker', inplace=True)

    dist = distance_to_spy(dwldr.df_distance)
    df = df.join(dist, rsuffix='dist')
    df['dist'] = (1/df['dist'])/(1/df['dist']).sum()

    df['sum'] = df[['lt', 'gt', 'in_range', 'dist']].sum(axis=1)
    df['sum'] = (df['sum'])/(df['sum']).sum()

    df = df.sort_values(by='sum')[10:]
    df['sum'] = df[['lt', 'gt', 'in_range', 'dist']].sum(axis=1)
    df['sum'] = (df['sum'])/(df['sum']).sum()

    print(df)
    # print(dist)



test()
