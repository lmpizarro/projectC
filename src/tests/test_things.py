import sys
sys.path.append('..')

import  matplotlib.pyplot as plt
from portfolios import (cum_returns, download,
                        mrkt_returns,
                        vars_covars,
                        symbols_returns,
                        get_cross_matrix)
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




if __name__ == '__main__':
    test_cum_returns()