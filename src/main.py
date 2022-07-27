import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

def plot_stacked(df):
    fig, axs = plt.subplots(len(symbols))
    fig.suptitle('Vertically stacked subplots')

    for i in range(len(symbols)):
        key = symbols[i] + '_var'
        axs[i].plot(df[key])
        axs[i].yaxis.set_label_position("right")
        axs[i].set_ylabel(symbols[i])

    plt.show()

def calc_var(symbols, df, lmbd=.99, ewma=True, deno=False):
    for s in symbols:
        key_var = s + '_var'
        key_ewma = s + '_ewm'
        key_csum = s + '_csum'

        if deno:
            s = s + '_deno'

        df[s] = np.log(df[s]/df[s].shift(1))
        df[key_ewma] = df[s].ewm(alpha=1-lmbd).mean()
        
        df[key_var] = (df[s]**2).ewm(alpha=1-lmbd).mean()
        if not ewma:        
            df[key_var] -= df[key_ewma]**2 
        
        df[key_csum] = df[s].cumsum()
        
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
    return df

def get_cross_var_keys(symbols):
    keys = []
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            s1 = symbols[i]
            s2 = symbols[j]
            key = f'{s1}_{s2}'
            keys.append(key)
    return keys

def get_var_keys(symbols):
    keys = []
    for s in symbols:
        key_var = s + '_var'
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
        key = f'{symbols[i]}_var'
        a[i,i] = sample[key]
        for j in range(i+1, len(symbols)):
            key = f'{symbols[i]}_{symbols[j]}'
            a[i,j] = sample[key] 
            a[j,i] = sample[key]
    return a

def min_var(symbols, df):
    data = []
    w_old = np.zeros(len(symbols))
    N = 0
    w = w_old
    for index, row in df.iterrows():

        a = get_matrix(symbols, sample=row)
        data.append(np.matmul(w, np.matmul(a, w)))

        s_var = (1/row[[e+'_var' for e in symbols]]).sum()
        if s_var != 0:
            whts = [(1/row[e+'_var'])/s_var for e in symbols if s_var != 0 and row[e+'_var'] != 0]
        if len(whts) == len(symbols):
            w = np.array(whts)
        
            diff_ = np.abs(w - w_old).sum()
            if diff_ > 0.05:

                w_old = w
                print(np.round(w,2), index)
                N = N + 1
            
    print(N, len(df))        

    df['port1'] = np.array(data)
    plt.plot(df['port1'])
    plt.show()

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
    keys.extend(get_var_keys(symbols))
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
symbols = ['KO', 'PEP']

equal_weights = np.array([1/len(symbols)] * len(symbols))

df = yf.download(symbols, '2015-2-1')['Adj Close']
df_prices = copy.deepcopy(df)

lmbd = .90
ewma = False
df_rets = calc_var(symbols, df, lmbd, ewma=ewma)
df_rets.fillna(0, inplace=True)
df_rets = calc_cross(symbols, df, lmbd, ewma=ewma)
df_rets.fillna(0, inplace=True)

print(df_prices.tail())


for s in symbols:

    plt.plot(df_rets[s+'_csum'])
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

