import copy
import numpy as np
import matplotlib.pyplot as plt
from plot.ploter import plot_stacked
from portfolios import symbols_returns 
import pandas as pd


def index_start_end(start='2000-01-01', end='2022-06-30', freq='6M'):
    time_index = pd.date_range(start=start, end=end, freq=freq)
    time_index_from = list(time_index[:-1])
    time_index_to = list(time_index[1:])

    return time_index_from, time_index_to


def get_start_end(df):
    day0 = df.head(1).iloc[0].name
    day1 = df.tail(1).iloc[0].name
    diff_ = (day1 - day0) / np.timedelta64(1, 'D')
    start = str(day0).split()[0]
    end = str(day1).split()[0]

    return start, end


def array_periods(t_len=105, per_len=10):
    periods = int(t_len / per_len) + 1
    if periods > 1:
        l_periods = [i for i in range(periods) for _ in range(per_len)]

        l_periods = l_periods[:-(len(l_periods) - t_len)]
    else:
        raise ValueError

    return np.array(l_periods)


def periodic_returns(df_rets):
    df_c = copy.deepcopy(df_rets)

    symbols = list(df_rets.keys())
    symbols.remove('period')
    for s in symbols:
        df_c[s] = df_c[['period', s]].groupby('period').cumsum()

    return df_c

def add_periods(df_rets, len_period=int(252/3)):

    df_c = copy.deepcopy(df_rets)
    df_c['period'] = array_periods(len(df_rets), len_period)

    df_c['END'] = df_c['period'].diff(-1)
    df_c['BEGIN'] = df_c['period'].diff()

    df_end = df_c[df_c['END'] != 0]
    df_beg = df_c[df_c['BEGIN'] != 0]
    
    df_c.drop(columns=['END'], inplace=True)
    df_c.drop(columns=['BEGIN'], inplace=True)

    return df_c, df_beg.index, df_end.index

def test_periodic_returns():

    symbols = ['AAPL', 'MSFT', 'AMZN', 'KO']
    weights = np.array([1/len(symbols)]*len(symbols))
    df_rets = symbols_returns(symbols, years=20)

    df_c = weights * df_rets.drop(columns=['MRKT'])
    df_c['sum'] = df_c.sum(axis=1)
    df_c['csum'] = df_c['sum'].cumsum()

    print(df_c.head(20))
    
    plt.plot(df_c['csum'])
    plt.show()

    plot_stacked(symbols, df_rets, k='', title='returns')
    plt.show()

    df_pridc, beg_index, end_index = add_periods(df_rets, len_period=10)

    df_pridc = periodic_returns(df_pridc)

    df_pridc.drop(columns=['MRKT', 'period'], inplace=True)

    for i, beg in enumerate(beg_index):
        en_d = end_index[i]
        # print((df_rets[beg:en_d].mean()*len(df_rets[beg:en_d])))

    df_c = weights * df_pridc
    df_c['sum'] = df_c.sum(axis=1)

    return_ends = df_c.loc[end_index]
    return_ends['csum'] = return_ends['sum'].cumsum()
    print(return_ends)

    plt.plot(return_ends['csum'])
    plt.show()


    plt.plot(df_c['sum'])
    plt.axhline(y=0)
    plt.show()

    from_, to_ = get_start_end(df_rets)

    ti_from, ti_to = index_start_end(start=from_, end=to_)

    for i, tt in enumerate(ti_from):
        # print(tt, ti_to[i])
        pass

if __name__ == '__main__':
    test_periodic_returns()