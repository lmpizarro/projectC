import sys
sys.path.append('..')

import  matplotlib.pyplot as plt
from portfolios import (cum_returns, download,
                        mrkt_returns,
                        vars_covars,
                        symbols_returns,
                        get_cross_matrix,
                        cross_matrix)

from betas import (rolling_beta_fussion, beta_by_ewma)

from calcs import returns
from plot.ploter import plot_stacked
import seaborn as sns

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
    covaria = vars_covars(df_rets, mode='garch')
    plot_stacked(symbols, covaria, k='_ewma', title='covaria')
    plot_stacked(symbols, covaria, k='_MRKT', title='covaria_mrkt', skip=True)

    from numpy import linalg as LA
    symbols.remove('MRKT')
    for index, row in covaria.iterrows():

        a = get_cross_matrix(symbols, row_item=row)

        try:
            # print(LA.cond(a), bb)
            bb = min(LA.svd(a, compute_uv=False))*min(LA.svd(LA.inv(a), compute_uv=False))
        except Exception as err:
            print(index, err, a)

    print(covaria.head())

def test_download():
    symbols = ['MSFT', 'KO']

    dfs = download(symbols=symbols, years=10)

    print(dfs.tail())

def test_cum_returns():
    symbols = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'BRK-B', 'KO', 'NVDA', 'JNJ', 'META', 'PG', 'MELI', 'PEP', 'AVGO']
    symbols = ['AAPL', 'MSFT', 'AMZN', 'KO']


    df_rets = symbols_returns(symbols, years=10)
    cum_ret = cum_returns(df_rets)

    print(cum_ret.tail(10))

    plot_stacked(symbols, df_rets, k='', title='returns')
    plot_stacked(symbols, cum_ret, k='_csum', title='c_returns')
    sns.pairplot(df_rets)
    plt.show()


def test_denoise():
    symbols = ['BIL', 'HON', 'CL']

    df:pd.DataFrame = download(symbols, denoise=True)

    df_rets = returns(symbols, df)
    var_s = vars_covars(df_rets)

    var_s['sum_ewma'] = var_s.mean(axis=1)
    print(var_s.tail())
    symbols.append('sum')
    plot_stacked(symbols,var_s, k='_ewma')
    plt.show()

def test_fama_french():
    import datetime
    from pandas_datareader import famafrench


    end = datetime.datetime(2022,6,30)
    start = datetime.datetime(2012, 6, 30)
    ds = famafrench.FamaFrenchReader('F-F_Research_Data_Factors_daily', start, end).read()
    df_f_f = ds[0]
    df_f_f[['Mkt-RF', 'SMB', 'HML', 'RF']] = df_f_f[['Mkt-RF', 'SMB', 'HML', 'RF']] / 100
    print(ds[0].tail())
    print(ds[0].keys())

    df = download(['AAPL', 'PG'])
    df_rets = returns(['AAPL', 'PG'], df)

    import pandas as pd
    print(df_rets.tail())
    m = pd.merge(df_rets, df_f_f, on='Date')
    print(m.tail())
    plt.plot(m.cumsum())
    plt.show()

def test_betas():
    symbols = ['TSLA', 'KO', 'AAPL', 'SPY', 'AVGO']

    df = download(symbols)
    df1 = df.copy()
    # df['SPY'] = np.random.normal(.01, .1, size=len(df)) + np.random.normal(.01, .001, size=len(df))
    # df['BIL'] = 5 * df['SPY']
    # df['KO'] = .5 * df['SPY']

    df.rename(columns={'SPY':'MRKT'}, inplace=True)
    symbols.remove('SPY')
    symbols.append('MRKT')

    df_betas = rolling_beta_fussion(df, N=120)

    symbols.remove('MRKT')
    plt.plot(df_betas[symbols])
    plt.show()

def test_beta_covar():
    symbols = ['TSLA', 'KO', 'AAPL', 'SPY', 'AVGO']


    df = download(symbols)
    df.rename(columns={'SPY':'MRKT'}, inplace=True)

    symbols.remove('SPY')

    symbols.append('MRKT')

    df = cross_matrix(symbols, df)
    betas = beta_by_ewma(symbols, df)

    print(betas.tail())

if __name__ == '__main__':
    test_betas()