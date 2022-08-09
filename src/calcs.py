import numpy as np

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

def vars(symbols, df, lmbd=.99, ewma=True):
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

def cross_vars(symbols, df, lmbd=.99, ewma=True, deno=False):
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

def beta_by_ewma(symbols, df_cov):
    symbols.remove('MRKT') 
    for s in symbols:
        k_cov_mrkt = f'{s}_MRKT'
        df_cov[f'B_{s}'] = df_cov[k_cov_mrkt] / df_cov['MRKT_ewma']
    
    return df_cov

def cross_matrix(symbols, df, lmbd=0.94, ewma=True, deno=False):
    df_rets = returns(symbols, df, deno)
    df_rets = vars(symbols, df_rets, lmbd, ewma=ewma)
    df_rets = cross_vars(symbols, df_rets, lmbd, ewma=ewma)
    return df_rets

