import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import nct
from scipy.stats import norm
from scipy.stats import cauchy
from collections import namedtuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans


CDF = namedtuple("CDF", "x cdf pdf")
Describe = namedtuple('Describe', 'min max sum mean std loc scale')


stocks = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BRK-B', 'TSLA', 'META', 'JNJ',
        'V', 'TSM', 'XOM', 'UNH', 'WMT', 'JPM', 'MA', 'PG', 'LLY', 'CVX', 'HD',
        'ASML', 'ABBV', 'GLOB', 'MELI', 'MRK', 'NVO', 'KO', 'AVGO', 'BABA', 'ORCL', 'PEP', 'PFE',
        'PM', 'AZN', 'TMO', 'BAC', 'COST', 'NVS', 'CSCO', 'MCD', 'SHEL', 'CRM', 'NKE', 'ACN',
        'DIS', 'TMUS', 'ABT', 'DHR', 'ADBE', 'LIN', 'VZ']

stocks = stocks

class Downloader:
    def __init__(self, start='2017-01-01', stocks=stocks) -> None:
        self.stocks = stocks
        self.start = start
        self.ewm: pd.DataFrame
        self.log_returns: pd.DataFrame
        self.ewm_log_returns: pd.DataFrame
        self.df_distance: pd.DataFrame
        self.properties: pd.DataFrame
        self.yf_data_close: pd.DataFrame

    def download(self):
        self.yf_data_close = yf.download(self.stocks, start=self.start, auto_adjust=True)['Close']
        self.yf_data_close.dropna(inplace=True)

    def calc_log_return(self):
        self.log_returns = pd.DataFrame()
        for stock in self.stocks:
            self.log_returns[stock] = np.log(self.yf_data_close[stock]/self.yf_data_close[stock].shift(1))

        self.log_returns.dropna(inplace=True)


    def calc_props(self):

        properties = []
        for stock in self.stocks:
            props = {}
            props['ticker'] = stock
            spy = np.array(self.log_returns['SPY'])
            reg_spy = LinearRegression().fit(spy.reshape(-1,1), np.array(self.log_returns[stock]))
            props['beta'] = reg_spy.coef_[0]
            props['alfa'] = reg_spy.intercept_

            props['mean'] = np.mean(self.log_returns[stock])
            props['std'] = np.std(self.log_returns[stock])
            props['VaR_95_vc'] = norm.ppf(0.05, props['mean'], props['std'])
            # props['VaR_99_vc'] = norm.ppf(0.01, props['mean'], props['std'])
            # props['VaR_95_hs'] = self.log_returns[stock].quantile(0.05)
            # props['VaR_99_hs'] = self.log_returns[stock].quantile(0.01)
            d = self.log_returns[stock][self.log_returns[stock] < 0]
            # props['lt_z_mean'] = d.mean()
            props['lt_z_med'] = d.median()
            props['lt_z_std'] = d.std()
            d = self.log_returns[stock][self.log_returns[stock] > 0]
            # props['gt_z_mean'] = d.mean()
            props['gt_z_med'] = d.median()
            props['gt_z_std'] = d.std()

            properties.append(props)
        df_dist_spy = self.distance_to_spy()
        print(df_dist_spy)

        self.properties = pd.DataFrame(properties)
        self.properties.set_index('ticker', inplace=True)
        self.properties = self.properties.join(df_dist_spy, rsuffix='dist')


    def calc_ewma(self):
        self.ewm = pd.DataFrame()
        for stock in self.stocks:
            self.ewm[stock] = self.yf_data_close[stock].ewm(alpha=.9).mean()

    def calc_ewm_log_returns(self):
        self.ewm_log_returns = pd.DataFrame()
        for stock in self.stocks:
            self.ewm_log_returns[stock] = self.log_returns[stock].ewm(alpha=.9).mean()

    def calc_dist(self) -> pd.DataFrame:
        self.df_distance = pd.DataFrame()
        for stock in self.stocks:
            close = self.yf_data_close[stock]
            log = np.log(close/close.shift(1))
            log.dropna(inplace=True)

            mean_log = log.ewm(alpha=.9).mean()
            mean_log_2 = mean_log * mean_log
            log2_ewm = (log*log).ewm(alpha=.9).mean()
            dist = (log - mean_log) / np.sqrt(log2_ewm - mean_log_2)
            dist.dropna(inplace=True)
            self.df_distance[stock] = dist

        self.df_distance = np.sqrt(2*(1-self.df_distance.corr()))

    def returns_gt_threshold(self, pct_loc=1, pct_scale=1.82):
        dict_gt = {}
        for stock in self.stocks:
            loc, scale = norm.fit(data=self.log_returns[stock])
            d = self.log_returns[stock][self.log_returns[stock] >= pct_loc*loc + pct_scale*scale]
            dict_gt[stock] = Downloader.describe(d, loc=loc, scale=scale)
        return dict_gt

    @staticmethod
    def describe(d: pd.DataFrame, loc, scale):
        return Describe(max=d.max(), mean=d.mean(), sum=d.sum(), min=d.min(), std=d.std(), loc=loc, scale=scale)

    def returns_lt_threshold(self, pct_loc=1, pct_scale=1.82):
        dict_lt = {}
        for stock in self.stocks:
            loc, scale = norm.fit(data=self.log_returns[stock])
            d = self.log_returns[stock][self.log_returns[stock] <= pct_loc*loc - pct_scale*scale]
            dict_lt[stock] = Downloader.describe(d, loc=loc, scale=scale)
        return dict_lt


    def returns_gt_zero(self):
        return self.returns_gt_threshold(0,0)

    def returns_lt_zero(self):
        return self.returns_lt_threshold(0,0)

    def returns_in_range(self, pct_loc= 1, min=0, max=2):
        dict_in_range = {}
        for stock in self.stocks:
            loc, scale = norm.fit(data=self.log_returns[stock])
            d = \
                self.log_returns[stock][(self.log_returns[stock] >= pct_loc*loc - min*scale) & (self.log_returns[stock] <= pct_loc*loc + max*scale)]
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

    def distance_to_spy(self):
        self.calc_dist()
        weights = []
        tickers = list(self.df_distance.keys())
        for i, k in enumerate(self.df_distance.keys()):
            for j in range(i+1, len(self.df_distance.keys())):
                c = k
                r = self.df_distance.keys()[j]
                weights.append((r, c, self.df_distance.loc[r][c]))


        weights.sort(key=lambda a: a[2], reverse=True)
        dist_to_spy = {'SPY':0}
        for w in weights:
            if w[0] == 'SPY' or w[1] == 'SPY':
                dist_to_spy[w[0]] = [1/w[2]]

        dist_to_spy = pd.DataFrame(dist_to_spy).T

        dist_to_spy.rename(columns={0: 'dist'}, inplace=True)

        return dist_to_spy


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
    dwldr.download()

    dwldr.calc_log_return()
    dwldr.calc_props()
    print(dwldr.properties)
    normalizer = Normalizer()
    kmeans = KMeans(n_clusters=10, max_iter=1000)
    pipeline = make_pipeline(normalizer,kmeans)
    pipeline.fit(dwldr.properties)

    print(kmeans.inertia_)
    labels = pipeline.predict(dwldr.properties)

    df = pd.DataFrame({'labels': labels, 'companies': dwldr.properties.index})


    print(print(df.sort_values('labels')))

    exit()

    dwldr.calc_ewma()
    dwldr.calc_ewm_log_returns()
    dwldr.calc_dist()
    gt = dwldr.returns_gt_zero()
    lt = dwldr.returns_lt_zero()
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
        factor['loc'] = gt[k].loc
        factor['scale'] = lt[k].std
        factors.append(factor)

    # print(hist['AAPL'].pdf, hist['AAPL'].cdf, hist['AAPL'].x)
    df = pd.DataFrame(factors)
    # df[['a', 'b', 'c']] = df[['lt', 'gt', 'in_range']] - df.loc[0][['lt', 'gt', 'in_range']]
    df['lt'] = (1/df['lt'])/(1/df['lt']).sum()
    df['scale'] = (1/df['scale'])/(1/df['scale']).sum()
    df['gt'] = (df['gt'])/(df['gt']).sum()
    df['loc'] = (df['loc'])/(df['loc']).sum()

    df.drop(index=0, axis=1, inplace=True)
    df.set_index('ticker', inplace=True)

    dist = distance_to_spy(dwldr.df_distance)
    df = df.join(dist, rsuffix='dist')
    df['dist'] = (1/df['dist'])/(1/df['dist']).sum()

    df['sum'] = df[['lt', 'gt', 'dist']].sum(axis=1)
    df['sum'] = (df['sum'])/(df['sum']).sum()

    df = df.sort_values(by='sum')[10:]
    df['sum'] = df[['lt', 'gt',  'dist']].sum(axis=1)
    df['sum'] = (df['sum'])/(df['sum']).sum()

    print(df)
    # print(dist)
test()
