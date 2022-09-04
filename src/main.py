from datetime import datetime, timedelta
from re import A
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from calcs import cumsum
from denoisers.butter.filter import min_lp, butter
from plot.ploter import plot_stacked
from portfolios import (min_ewma_port,
                        equal_weight_port,
                        get_cross_matrix,
                        get_cross_var_keys,
                        get_ewma_keys,
                        vars_covars,
                        returns,
                        download)



rf = 0.015
import scipy.optimize as sco

def max_SR_opt(mean_returns, cov_matrix, rf_rate, n, display = False):

    def get_ret_vol_sr(weights, mean_returns, cov_matrix, rf_rate):
        weights = np.array(weights)
        ret = np.sum(np.array(mean_returns) * weights)
        vol = np.sqrt(np.dot(weights.T,np.dot(np.array(cov_matrix),weights)))
        sr = (ret-rf)/vol
        return np.array([ret,vol,sr])

    # minimize negative Sharpe Ratio
    def neg_sharpe(weights, mean_returns, cov_matrix, rf_rate):
        return -get_ret_vol_sr(weights, mean_returns, cov_matrix, rf_rate)[2]

    # check allocation sums to 1
    def check_sum(weights):
        return np.sum(weights) - 1

    # create constraint variable
    cons = ({'type':'eq','fun':check_sum})

    # create weight boundaries
    bounds = ((0,1),)*n

    # initial guess
    init_guess = [1/n]*n

    opt_results = sco.minimize(neg_sharpe, init_guess,
                               method='SLSQP', bounds = bounds,
                               constraints = cons,
                               args = (mean_returns, cov_matrix, rf_rate),
                               options = {'disp': display})

    weights = pd.Series(np.round(opt_results.x,2), index = mean_returns.index)
    return weights, opt_results.fun


def sharpe_fun(returns, a, w, rfr):
    ret = np.dot(w.T, np.array(returns)) - rfr
    risk =  np.sqrt(np.dot(w.T, np.dot(a, w)))
    return ret / risk

def weights_func(size):
    w = np.random.uniform(size=size)
    w = w / w.sum()
    wc = 1 - w
    wc = wc / wc.sum()

    return w, wc

def max_sharpe(symbols, df, rfr=0.0001, r=False):
    np.random.seed(1)
    N_index = 0
    N_max = 12
    for index, row in df.iterrows():

        N_index += 1
        a = get_cross_matrix(symbols, row_item=row)
        returns = row[[e+'_ewm' for e in symbols]]

        if not N_index % 60:
            if r:
                w, fun = max_SR_opt(returns, a, rfr, len(symbols))
                print(np.array(w), fun, index)
            else:
                max_s = -1000000
                max_w = None
                N = 0
                M = 0
                while True:
                    M += 1
                    if M == 3000000:
                        break

                    w, wc = weights_func()

                    sharpe = sharpe_fun(returns, a , w, rfr)
                    sharpe2 = sharpe_fun(returns, a , wc, rfr)

                    if sharpe2 > sharpe:
                        sharpe = sharpe2

                    if sharpe > max_s:
                        max_s = sharpe
                        max_w = w
                        N +=1
                        if N > N_max:
                            break

                if max_w is not None:
                    print(100*np.round(max_w,2), M, N, str(index).split(' ')[0])
                else:
                    print(max_w, index)

def control_var(symbols, df):
    keys = get_cross_var_keys(symbols)
    keys.extend(get_ewma_keys(symbols))
    s = df[keys].sum(axis=1)
    f_ema = (s).ewm(span=15).mean()
    s_ema = s.ewm(span=150).mean()
    control_ = ((f_ema - s_ema)<0)
    return control_

def control_var_key(symbol, df):
    s = df[symbol+'_var']
    f_ema = s.ewm(span=15).mean()
    s_ema = s.ewm(span=150).mean()
    control_ = ((f_ema - s_ema)<0)
    return control_

symbols = ['KO', 'PEP', 'PG', 'AAPL', 'JNJ', 'AMZN', 'DE', 'CAT', 'META', 'MSFT', 'ADI']
symbols = ['MSFT', 'AVGO', 'PG', 'PEP', 'AAPL', 'KO', 'LMT', 'TSLA', 'ADI', 'MELI', 'JNJ', 'SPY', 'AMZN', 'META']


def test_equal_weight():

    symbols = ['PG', 'PEP', 'AAPL']
    np.random.seed(1)
    symbols.sort()

    df_min = min_ewma_port(symbols)

    df_equal = equal_weight_port(symbols)
    print(df_equal.keys())

    diff__ = df_equal['risk']- df_min['risk']
    print(diff__.mean())

    plt.plot(df_min['risk'])
    plt.plot(df_equal['risk'], 'k')
    plt.show()
    plt.plot(diff__)
    plt.show()

    plt.plot(df_equal['returns'].cumsum())
    plt.show()
    plt.plot(df_min['returns'].cumsum(), 'g')
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

def test_options():
    """
        AAPL220826C00070000
        symbol = AAPL
        year = 2022
        date = 26
        month = 08
        C = call
        00070000 = 70   9999.9999
    """
    aapl = yf.Ticker('AAPL')
    options_dates = aapl.options

    strikes = pd.DataFrame()
    for d in options_dates:
        ex_date = datetime.strptime(d, "%Y-%m-%d").date()
        today = datetime.now().date()
        time_to_expire = ex_date - today

        calls: pd.DataFrame = aapl.option_chain(d).calls
        vol_strike = calls[['strike', 'impliedVolatility']]
        vol_strike = vol_strike.set_index('strike')
        vol_strike.rename(columns={'impliedVolatility':time_to_expire.days}, inplace=True)
        print(vol_strike)
        if strikes.empty:
            strikes = vol_strike
        else:
            strikes = pd.merge(strikes, vol_strike, on='strike')

    strikes.fillna(0, inplace=True)
    print(calls.keys())
    print(strikes.tail())
    plt.plot(strikes.index, strikes[strikes.keys()])
    plt.show()
    from matplotlib import cm
    x = np.array(strikes.columns, dtype="float64")
    y = strikes.index
    X,Y = np.meshgrid(x,y)
    Z = np.array(strikes, dtype="float64")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.hot, linewidth=0, antialiased=True)
    # ax.plot_wireframe(X, Y, Z, rstride=40, cstride=40)
    """
    https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    Understanding the Particle Filter | | Autonomous Navigation, Part 2
    https://www.youtube.com/watch?v=NrzmH_yerBU

    https://www.youtube.com/watch?v=aUkBa1zMKv4
    https://www.youtube.com/watch?v=YBeVDxTHiYM

    https://www.youtube.com/watch?v=O-lAJVra1PU
    Particle Filter Tutorial With MATLAB Part 1: Student Dave

    https://www.youtube.com/watch?v=OM5iXrvUr_o
    Particle Filter Algorithm

    https://www.youtube.com/watch?v=r-bHqY5gaPw
    CS 188 Lecture 19: Particle Filtering

    https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/notebooks/basic_tutorial.html


    """
    plt.show()

if __name__ == '__main__':
    test_equal_weight()
    exit()
    symbols = ['MSFT', 'AVGO', 'PG', 'BIL', 'SPY']
    symbols = ['BIL', 'HON', 'CL', 'AVGO', 'PG', 'PEP', 'XOM', 'KO', 'TXN', 'MO', 'XOM',
               'PM', 'KO', 'LMT', 'INTC', 'ADI', 'HD', 'JNJ', 'WFC', 'MCD', 'T', 'GS', 'WMT',
               'SPY']

    symbols = ['AAPL', 'TSLA', 'SPY']

    df:pd.DataFrame = download(symbols, denoise=False)
    df = min_lp(symbols, df)
    df.drop(columns=symbols, inplace=True)
    df.rename(columns={f'{s}_deno':s for s in symbols}, inplace=True)



    df_rets = returns(symbols, df)

    print(df_rets.tail())
    plot_stacked(symbols, df_rets, k='', start=250)


    # df_rets = df_rets.ewm(alpha=.05).mean()
    # df_rets = butter(symbols, df_rets, 70)
    # df_rets = min_lp(symbols, df_rets)
    # plot_stacked(symbols, df_rets, k='_deno', start=250)

    print(df_rets.tail())
    exit()

    df = download(symbols=symbols, years=10)
    df_rets = returns(symbols, df)
    data_neg = {}
    data_gt = {}
    for s in symbols:
        lt_zero = df_rets[s][df_rets[s] < 0]
        gt_zero = df_rets[s][df_rets[s] > 0]
        if s not in data_neg:
            data_neg[s] = {'ticker':s}
        if s not in data_gt:
            data_gt[s] = {'ticker':s}


        data_neg[s]['mean'] = lt_zero.mean()
        data_neg[s]['dev'] = lt_zero.std()
        data_neg[s]['count'] = lt_zero.count()

        data_gt[s]['mean'] = gt_zero.mean()
        data_gt[s]['dev'] = gt_zero.std()
        data_gt[s]['count'] = gt_zero.count()

    df_neg = pd.DataFrame.from_dict(data_neg, orient='index' )
    df_neg = df_neg.set_index('ticker')
    df_gt = pd.DataFrame.from_dict(data_gt, orient='index')
    df_gt = df_gt.set_index('ticker')

    # df_gt['count'] = df_gt['count'] / df_gt['count'].loc['SPY']
    # df_gt['mean'] = df_gt['mean'] / df_gt['mean'].loc['SPY']
    # df_gt['dev'] = df_gt['dev'] / df_gt['dev'].loc['SPY']
    # df_gt['sum'] = df_gt.sum(axis=1)
    # df_gt.drop('SPY', inplace=True)

    # df_neg['count'] = df_neg['count'].loc['SPY'] / df_neg['count']
    # df_neg['mean'] = df_neg['mean'].loc['SPY'] / df_neg['mean']
    # df_neg['dev'] = df_neg['dev'].loc['SPY'] / df_neg['dev']
    # df_neg['sum'] = df_neg.sum(axis=1)
    # df_neg.drop('SPY', inplace=True)

    # df_gt['total'] = df_neg['sum'] + df_gt['sum']
    # total = df_gt.total[symbols].sum()

    # df_gt.total = df_gt.total / total
    print(df_neg)
    print(df_gt)
    # print(total)
    # print(df_gt['total'][symbols].sum())

    exit()
    df = yf.download(symbols, '2015-2-1')['Adj Close']
    df_prices = copy.deepcopy(df)



    lmbd = .94
    df_ewma = vars_covars(df, lmbd, mode='teor')

    print(df_prices.tail())
    print(df_ewma.tail())

    plot_stacked(symbols, df_ewma, '_ewma')

    print(df_ewma.keys())
    print(df.keys())



    # max_sharpe(symbols, df)
    # min_var(symbols, df)

    #plt.plot(df.KO_var)
    #plt.plot(df.PG_var> df.KO_var)
    #plt.plot(df.AAPL_var > df.KO_var, 'r')
    # plt.plot(df.AAPL_var/ df.AAPL_ewm, 'r')
    plt.plot(df.PG_csum, 'g')
    plt.show()

    print(df.KO.describe())
    control_ = control_var_key('PG', df)
    df.PG = df.PG*control_

    plt.plot(df.PG.cumsum())
    plt.show()

    # probar n portfolios aleatorios
    # https://medium.com/@Piotr_Szymanski/arithmetic-vs-log-stock-returns-in-python-7f7c3cff125


    ret_df = df[get_return_keys(symbols)]

    port_ret = ret_df.dot(equal_weights)
    dff = port_ret.ewm(alpha=1-lmbd).mean()
    dff = (dff**2).ewm(alpha=1-lmbd).mean()
    port = port_ret.cumsum()

    plt.plot(dff)

    for i in range(3):
        w, wc = weights_func(len(symbols))
        port_ret = ret_df.dot(w)
        dff = port_ret.ewm(alpha=1-lmbd).mean()
        dff = (dff**2).ewm(alpha=1-lmbd).mean()
        plt.plot(dff)

    plt.show()

    print(port.tail())