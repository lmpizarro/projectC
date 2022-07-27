import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from plot.ploter import plot_stacked

def calc_returns(symbols, df, deno=False):
    for s in symbols:
        if deno:
            pass

        df[s] = np.log(df[s]/df[s].shift(1))
    
    df.fillna(0, inplace=True)
    return df

def calc_csum(symbols, df):
    for s in symbols:
        key_csum = s + '_csum'
        df[key_csum] = df[s].cumsum()

    df.fillna(0, inplace=True)

    return df

def calc_var(symbols, df, lmbd=.99, ewma=True):
    for s in symbols:
        key_filt = s + '_filt'
        key_ewma = s + '_ewma'

        # filtered returns
        df[key_filt] = df[s].ewm(alpha=1-lmbd).mean()

        # ec cc 21        
        df[key_ewma] = (df[s]**2).ewm(alpha=1-lmbd).mean()
        if not ewma:        
            df[key_ewma] -= df[key_filt]**2 
        
        
    df.fillna(0, inplace=True)
    return df

def calc_cross(symbols, df, lmbd=.99, ewma=True, deno=False):
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            s1 = symbols[i]
            s2 = symbols[j]
            key = f'{s1}_{s2}'
            if deno:
                df[key] = df[s1+'_deno'] * df[s2+'_deno']
            else:
                df[key] = df[s1] * df[s2]

            df[key] = df[key].ewm(alpha=1-lmbd).mean()
            if not ewma:
                df[key] -= (df[key]).ewm(alpha=1-lmbd).mean()
    df.fillna(0, inplace=True)
    return df


def calc_matrix(symbols, df, lmbd, ewma=True, deno=False):
    df_rets = calc_returns(symbols, df, deno)
    df_rets = calc_var(symbols, df_rets, lmbd, ewma=ewma)
    df_rets = calc_cross(symbols, df_rets, lmbd, ewma=ewma)
    return df_rets

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
 
def get_matrix(symbols, sample):
    a = np.zeros(len(symbols)*len(symbols))
    a = a.reshape(len(symbols), len(symbols))
    for i in range(len(symbols)):
        key = f'{symbols[i]}_ewma'
        a[i,i] = sample[key]
        for j in range(i+1, len(symbols)):
            key = f'{symbols[i]}_{symbols[j]}'
            a[i,j] = sample[key] 
            a[j,i] = sample[key]
    return a

def equal_weight_port(symbols, df):
    data = []
    w = np.array([1/len(symbols)] * len(symbols))
    for _, row in df.iterrows():

        a = get_matrix(symbols, sample=row)
        data.append(np.matmul(w, np.matmul(a, w)))

    df['equal_port'] = np.array(data)

    return df, w

def min_ewma_port(symbols, df):
    data = []
    w_old = np.zeros(len(symbols))
    N = 0
    w = w_old
    for index, row in df.iterrows():

        a = get_matrix(symbols, sample=row)
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

    df['ewma_port'] = np.array(data)

    return df, w

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
        a = get_matrix(symbols, sample=row)
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


np.random.seed(1)
symbols = ['KO', 'PEP', 'PG', 'AAPL', 'JNJ', 'AMZN', 'DE', 'CAT', 'META', 'MSFT', 'ADI']
symbols = ['PG', 'PEP', 'AAPL']

equal_weights = np.array([1/len(symbols)] * len(symbols))

df = yf.download(symbols, '2015-2-1')['Adj Close']
df_prices = copy.deepcopy(df)

lmbd = .94
ewma = False

df_rets = calc_matrix(symbols, df, lmbd, ewma=ewma)

print(df_prices.tail())
print(df_rets.tail())

plot_stacked(symbols, df_rets, '_filt')
plot_stacked(symbols, df_rets, '_ewma')

df, w = min_ewma_port(symbols,df)
df, w1 = equal_weight_port(symbols,df)
print(w, w1)
plt.plot(df['ewma_port'])
plt.plot(df['equal_port'], 'k')

plt.show()



exit()


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

