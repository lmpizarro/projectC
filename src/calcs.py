import numpy as np
from arch import arch_model

def returns(symbols, df, log__=False):

    if log__:
        df = np.log(df/df.shift(1))
    else:
        df = df.pct_change()

    df.dropna(inplace=True)
    return df

def cumsum(symbols, df):
    for s in symbols:
        key_csum = s + '_csum'
        df[key_csum] = df[s].cumsum()

    df.fillna(0, inplace=True)

    return df

def ewma_vars(symbols, df, lmbd=.99, mode='ewma'):
    for s in symbols:
        key_filt = s + '_filt'
        key_ewma = s + '_ewma'

        # filtered returns
        df[key_filt] = df[s].ewm(alpha=1-lmbd).mean()

        # ec cc 21
        df[key_ewma] = (df[s]**2).ewm(alpha=1-lmbd).mean()
        if mode == 'teor':
            df[key_ewma] -= df[key_filt]**2
        elif mode == 'garch':
            am = arch_model(100*df[s]**2, p=1, o=1, q=1, power=1.0, dist="StudentsT")
            res = am.fit(update_freq=5)
            df[key_ewma] = res.conditional_volatility / 100


    df.fillna(0, inplace=True)
    return df

def ewma_cross_vars(symbols, df, lmbd=.99, mode='ewma', deno=False):
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
            if mode == 'teor':
                df[key] -= (df[key]).ewm(alpha=1-lmbd).mean()
            elif mode == 'garch':
                am = arch_model(100*df[s1]*df[s2], p=1, o=1, q=1, power=1.0, dist="StudentsT")
                res = am.fit(update_freq=5)
                df[key] = res.conditional_volatility / 100


    df.fillna(0, inplace=True)
    return df

import pandas as pd

def beta_by_ewma(symbols, df_cov):
    beta = pd.DataFrame()
    symbols.remove('MRKT')
    for s in symbols:
        k_cov_mrkt = f'{s}_MRKT'
        beta[s] = df_cov[k_cov_mrkt] / df_cov['MRKT_ewma']

    return beta

def cross_matrix(symbols, df, lmbd=0.94, mode='ewma', deno=False):
    df_rets = returns(symbols, df, deno)
    df_rets = ewma_vars(symbols, df_rets, lmbd, mode=mode)
    df_rets = ewma_cross_vars(symbols, df_rets, lmbd, mode=mode)
    return df_rets

