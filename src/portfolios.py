from typing import List, Tuple
import numpy as np
from calcs import (cumsum, returns, vars, cross_vars, cross_matrix)
from denoisers.butter.filter import min_lp
from plot.ploter import plot_stacked

def get_cross_var_keys(symbols):
    keys = []
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            s1 = symbols[i]
            s2 = symbols[j]
            key = f'{s1}_{s2}'
            keys.append(key)
    return keys

def get_ewma_keys(symbols):
    keys = []
    for s in symbols:
        key_var = s + '_ewma'
        keys.append(key_var)
    return keys

def get_return_keys(symbols):
    keys = []
    for s in symbols:
        keys.append(s)
    return keys

def get_filt_keys(symbols):
    keys = []
    for s in symbols:
        key_var = s + '_filt'
        keys.append(key_var)
    return keys

def get_cumsum_keys(symbols):
    keys = []
    for s in symbols:
        key_var = s + '_csum'
        keys.append(key_var)
    return keys

 
def get_matrix(symbols, row_item):
    a = np.zeros(len(symbols)*len(symbols))
    a = a.reshape(len(symbols), len(symbols))
    for i in range(len(symbols)):
        key = f'{symbols[i]}_ewma'
        a[i,i] = row_item[key]
        for j in range(i+1, len(symbols)):
            key = f'{symbols[i]}_{symbols[j]}'
            a[i,j] = row_item[key] 
            a[j,i] = row_item[key]
    return a

def equal_weight_port(symbols, years=10, name='equal'):

    data_risk = []
    data_rel = []

    df = download(symbols=symbols, years=years)
    df = cross_matrix(symbols=symbols, df=df)
    df_rets = df[get_filt_keys(symbols)]


    w = np.array([1/len(symbols)] * len(symbols))
    for index, row in df.iterrows():
        
        return_ = np.matmul(w, df_rets.loc[index])
        a = get_matrix(symbols, row_item=row)
        risk = np.matmul(w, np.matmul(a, w))
        data_risk.append(risk)
        data_rel.append(return_ / risk)

    df[f'{name}_port'] = np.array(data_risk)
    df[f'{name}_rela'] = np.array(data_rel)

    return df, w

def min_ewma_port(symbols:List[str], years=10, name: str='inv'):
    df = download(symbols=symbols, years=years)
    df_rets = cross_matrix(symbols=symbols, df=df)
    # df_rets = df[get_filt_keys(symbols)]

    data = []
    w_old = np.zeros(len(symbols))
    N = 0
    w = w_old
    for index, row in df_rets.iterrows():
        a = get_matrix(symbols, row_item=row)
        data.append(np.matmul(w, np.matmul(a, w)))

        s_var = (1/row[[e+'_ewma' for e in symbols]]).sum()
        if s_var != 0:
            whts = [(1/row[e+'_ewma'])/s_var for e in symbols if s_var != 0 and row[e+'_ewma'] != 0]
        if len(whts) == len(symbols):
            w = np.array(whts)
        
            diff_ = np.abs(w - w_old).sum()
            if diff_ > 0.05:
                w_old = w
                N = N + 1  

    df_rets[f'{name}_port'] = np.array(data)

    return df_rets, w


import yfinance as yf
from datetime import date

def download(symbols, years=10, denoise=False, YTD=True, end=None):
    symbols.sort()
    symbols = [s.strip() for s in symbols]
    c_year = date.today().year
    c_month = date.today().month
    if YTD: c_month = 1

    begin = f'{c_year-years}-{c_month}-2'
    df = yf.download(symbols, begin, end)['Adj Close']
    print(df.tail())


    if denoise:

        if 'BIL' in symbols:
            symbols.remove('BIL')
        df = min_lp(symbols, df)
        df.drop(columns=symbols, inplace=True)

        re = {f'{s}_deno':s for s in symbols}
        df.rename(columns=re, inplace=True)
    
    print(df.tail())

    return df

def custom_port(ct_port: List[Tuple[str, float]],
                years=10,
                name='cust'):
    """
        ct_port [(symbol0, pct0),(...)...] 
    """

    c_port = sorted(ct_port, key=lambda tup: tup[0])
    symbols = [t[0] for t in ct_port]
    weights = np.asarray([w[1] for w in ct_port])
    weights = weights / weights.sum()
    df = download(symbols, years=years)

    df_rets = returns(symbols, df)
    
    df_c = weights * df_rets
    df_c[f'{name}'] = df_c.sum(axis=1)
    df_c[f'{name}_csum'] = df_c[f'{name}'].cumsum()

    return df_c[[f'{name}_csum', f'{name}']], c_port

def mrkt_returns(name='MRKT', years=10):
    """
        return MRKT daily returns
        return csum weighted market component
        return components and weight
    """
    c_port = [("^DJI", .1), ("^GSPC", .7), ('^IXIC', .1), ('^RUT', .1)]
    df, c_port = custom_port(c_port, years=years, name=name)
    return df, c_port


def symbols_returns(symbols, years=10):
    """
        include market returns
    """
    df = download(symbols, years=years, denoise=False)

    df_mrkt, _ = mrkt_returns(years=years)
    df_rets = returns(symbols, df)

    df_rets['MRKT'] = df_mrkt['MRKT']

    symbols.append('MRKT')

    return df_rets

import copy
def cum_returns(df_rets):
    df = copy.deepcopy(df_rets)
    symbols = df_rets.keys()
    df_accum = cumsum(symbols, df)
    df_accum.drop(columns=symbols, inplace=True)

    # keys symbol_csum
    return df_accum


def mrkt_diffs(df_rets):
    df_cu = cum_returns(df_rets)

    symbols = list(df_cu.keys())
    symbols.remove('MRKT_csum')

    for s in symbols:
        df_cu[s] = (df_cu[s])/df_cu['MRKT_csum']

    df_cu.drop(columns=['MRKT_csum'])
    re = {s:s.split('_')[0]for s in symbols}
    df_cu.rename(columns=re, inplace=True)


    return df_cu


def variances(df_rets, lmbd=.94, ewma=True):
    df = copy.deepcopy(df_rets)
    symbols = df_rets.keys()
    df = vars(symbols, df, lmbd, ewma=ewma)
    df = cross_vars(symbols, df, lmbd, ewma=ewma)
    df.drop(columns=symbols, inplace=True)

    df.drop(columns=[f'{s}_filt' for s in symbols], inplace=True)
    return df

import matplotlib.pyplot as plt
import seaborn as sns


def test_download():
    symbols = ['MSFT', 'KO']

    dfs = download(symbols=symbols, years=10)

    print(dfs.tail())

def test_mrkt_returns():
    df, c_port = mrkt_returns()
    print(df.tail())

    plt.plot(df['MRKT'])
    plt.show()

    plt.plot(df['MRKT_csum'])
    plt.show()

def test_covaria():

    symbols = ['AAPL', 'MSFT', 'AMZN', 'KO']
    df_rets = symbols_returns(symbols, years=10)
    covaria = variances(df_rets, ewma=False)
    plot_stacked(symbols, covaria, k='_ewma', title='covaria')
    plot_stacked(symbols, covaria, k='_MRKT', title='covaria_mrkt', skip=True)

    from numpy import linalg as LA
    symbols.remove('MRKT')
    for index, row in covaria.iterrows():

        a = get_matrix(symbols, row_item=row)

        try:
            # print(LA.cond(a), bb)
            bb = min(LA.svd(a, compute_uv=False))*min(LA.svd(LA.inv(a), compute_uv=False))
        except Exception as err:
            print(index, err, a)

    print(covaria.head())

def tracker01(symbols):
    df = download(symbols=symbols, years=14)
    df_rets = returns(symbols, df)
    df_c = cumsum(symbols=symbols, df=df_rets)

    symbols.remove('SPY')

    for s in symbols:
        df_c[f'{s}_d'] = df_c['SPY_csum'] - df_c[f'{s}_csum']

    print(df_c[[f'{s}_d' for s in symbols]].head())

    w_old = np.array([1/len(symbols)]*len(symbols))
    data = []
    counter = 0 
    for index, row in df_c.iterrows():

        r = row[[f'{s}_d' for s in symbols]]
        rx = r.copy()
        rx = rx + np.abs(r.min())*2
        rx_inv = (1 / rx ) ** 1
        rx_inv_sum = rx_inv.sum()
        w_inv = rx_inv / rx_inv_sum

        new_d = np.dot(w_old,row[[f'{s}_csum' for s in symbols]])
        data.append(new_d)
        counter += 1
        if not counter%250:
            diference = df_c.SPY_csum.loc[index] - np.dot(w_inv,row[[f'{s}_csum' for s in symbols]])
            print(f'update {counter} {diference} {100*np.abs(np.array(w_old) - np.array(w_inv)).sum()}')
            w_old = w_inv
    
    df_c['data'] = np.array(data)
    plt.plot(df_c.data)
    plt.plot(df_c.SPY_csum)
    plt.show()

from scipy import signal

def tracker02(symbols):
    df = download(symbols=symbols, years=14, denoise=False)
    print(df.tail())
    df_rets = returns(symbols, df)
    df_c = cumsum(symbols=symbols, df=df_rets)

    def weights(x, mu, sigma):
        rx_inv = np.exp(-(x-mu)**2/(sigma**2)) / (sigma*np.sqrt(np.pi))
        return rx_inv / rx_inv.sum()

    def minimizer(ref_time_serie, row, Np=20):
        col_time_series = row[[f'{s}_csum' for s in symbols]]
        rx = row[[f'{s}_d' for s in symbols]]
        sigma = rx.max() - rx.min()
        range_mu = np.linspace(2*rx.min(), 2*rx.max(), Np)
        winvs = [weights(rx, m, sigma) for m in range_mu]
        # winvs = [weights(rx, 0, m) for m in np.linspace(rx_range, 4*rx_range, 10)]
        dict_minimizer = {}
        for w_inv in winvs:
            k_diff = int(1e6*float(error(ref_time_serie, 
                                             col_time_series, 
                                             w_inv)))

            dict_minimizer[np.abs(k_diff)] = w_inv
        best_key = min(dict_minimizer.keys())
        w_inv = dict_minimizer[best_key]
        difference = error(ref_time_serie, 
                           col_time_series, 
                           w_inv)
 
        return w_inv, difference

    def error(spy_row, row, w_inv):
        return spy_row - np.dot(w_inv,row)

    symbols.remove('SPY')

    for s in symbols:
        df_c[f'{s}_d'] = df_c['SPY_csum'] - df_c[f'{s}_csum']

    print(df_c[[f'{s}_d' for s in symbols]].head())

    w_old = np.array([1/len(symbols)]*len(symbols))
    data = []
    counter = 0 
    for index, row in df_c.iterrows():


        new_d = np.dot(w_old,row[[f'{s}_csum' for s in symbols]])
        data.append(new_d)

        counter += 1
        if not counter%60:

            ref_time_serie = df_c.SPY_csum.loc[index]
           
            w_inv, difference = minimizer(ref_time_serie, row, Np=60)

            print(f'update {difference} {100*np.abs(np.array(w_old) - np.array(w_inv)).sum()}')
            w_old = w_inv
    
    df_c['data'] = np.array(data)
    plt.plot(df_c.data)
    plt.plot(df_c.SPY_csum)
    plt.show()


def min_distances():

    import pandas as pd
    from scraper.scrapers import scrap_cedear_rava, list_sp500
    ce = scrap_cedear_rava()
    
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    oldests_sp500 = list(sp500[sp500.Founded < '2002'].Symbol)
    print(len(oldests_sp500))

    symbols = [s for s in  oldests_sp500 if s in ce]

    symbols = symbols
    symbols.append('SPY')

    print(len(symbols))
    

    df = download(symbols=symbols, years=20)
    df_rets = returns(symbols, df, log__=True)
    df_rets = cumsum(symbols, df_rets)

    symbols.remove('SPY')

    distances = {}
    for s in symbols:
        df_rets[s] = np.abs(df_rets[s] - df_rets.SPY)
        ds = df_rets[s].sum()
        distances[s] = ds
    distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    return list(distances.keys())

def test_tracker():

    symbols = ['BIL', 'HON', 'MSFT', 'PEP', 'XOM', 'KO', 'TXN', 'MO', 'XOM', 'SPY']
    # symbols = ['BIL', 'HON', 'CL', 'XOM', 'SPY']
    symbols_min = min_distances()
    symbols = symbols_min[:10]
    # symbols.extend(symbols_min[-5:])
    symbols.append('BIL')
    symbols.append('SPY')
    tracker02(symbols)


if __name__ == '__main__':
    test_tracker()    
    exit()
    symbols = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'BRK-B', 'KO', 'NVDA', 'JNJ', 'META', 'PG', 'MELI', 'PEP', 'AVGO']
    symbols = ['AAPL', 'MSFT', 'AMZN', 'KO']


    df_rets = symbols_returns(symbols, years=10)
    cum_ret = cum_returns(df_rets)

    print(cum_ret.tail(10))
    
    plot_stacked(symbols, df_rets, k='', title='returns')
    plot_stacked(symbols, cum_ret, k='_csum', title='c_returns')
    sns.pairplot(df_rets)
    plt.show()

    
