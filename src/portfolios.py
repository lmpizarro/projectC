from typing import List, Tuple
import numpy as np
from calcs import cumsum

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

def equal_weight_port(symbols, df, name='equal'):
    df = cumsum(symbols, df)
    data_risk = []
    data_rel = []
    ret_keys = get_filt_keys(symbols)
    rets = df[ret_keys]

    w = np.array([1/len(symbols)] * len(symbols))
    for index, row in df.iterrows():
        
        return_ = np.matmul(w, rets.loc[index])
        a = get_matrix(symbols, row_item=row)
        risk = np.matmul(w, np.matmul(a, w))
        data_risk.append(risk)
        data_rel.append(return_ / risk)

    df[f'{name}_port'] = np.array(data_risk)
    df[f'{name}_rela'] = np.array(data_rel)

    return df, w

def min_ewma_port(symbols, df, name='inv'):
    data = []
    w_old = np.zeros(len(symbols))
    N = 0
    w = w_old
    for index, row in df.iterrows():

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

    df[f'{name}_port'] = np.array(data)

    return df, w


import yfinance as yf
from datetime import date
import copy
from calcs import cross_matrix

def custom_port(ct_port: List[Tuple[str, float]], 
                years=10,
                name='c_50_35_15'):

    c_port = sorted(ct_port, key=lambda tup: tup[0])
    symbols = [t[0] for t in ct_port]
    weights = np.asarray([w[1] for w in ct_port])
    weights = weights / weights.sum()
    c_year = date.today().year

    begin = f'{c_year-years}-1-2'

    df = yf.download(symbols, begin)['Adj Close']

    lmbd = .94
    ewma = False

    df_rets = cross_matrix(symbols, df, lmbd, ewma=ewma)
    df_rets = cumsum(symbols, df_rets)

    df_csum = df_rets[get_cumsum_keys(symbols)]
    
    df_c = weights * df_csum
    df_c['port_cum'] = df_c.sum(axis=1)

    return df_c, c_port

if __name__ == '__main__':
    c_port = [("^DJI", .15), ("^GSPC", .6), ('^IXIC', .25)]


    df, c_port = custom_port(c_port, years=20)
    

    print(df.tail())