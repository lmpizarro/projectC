from datetime import datetime
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pickle
import numpy as np

DAYS_IN_A_YEAR = 364

import scipy.optimize

class Fit:
    @staticmethod
    def monoExp(x, m, t, b):
        return m * np.exp(-t * x) + b

    @staticmethod
    def optimizeExp(rs, vs, p0):
        params, cv = scipy.optimize.curve_fit(Fit.monoExp, rs, vs, p0)
        m, t, b = params

        return m, t, b
    
    @staticmethod
    def polyModel(rs, vs, degree=8):
        return np.poly1d(np.polyfit(rs, vs, degree))


class FromYF:
    """
    13 Week Treasury Bill (^IRX)
    Treasury Yield 10 Years (^TNX)
    Treasury Yield 30 Years (^TYX)
    Treasury Yield 5 Years (^FVX)
    """
    @staticmethod
    def referencias(tickers=['EEM', 'GGAL', '^TNX', '^TYX', '^FVX', '^IRX']):
        t = yf.download(tickers,  '2020-01-02')['Adj Close']
        t['fecha'] = t.index
        usd_bonds = ['^TNX', '^TYX', '^FVX', '^IRX']
        t[usd_bonds] = t[usd_bonds] / 100

        return t


class LeerCSVS:
    @staticmethod
    def ccl():
        df3 = pd.read_csv('ccl.csv')
        df3.fillna(method='ffill', inplace=True)
        df3['fecha'] = pd.to_datetime(df3['fecha'],  format='%Y-%m-%d')
        df3.rename(columns={'ultimo': 'ccl'}, inplace=True)
        df3 = df3[['fecha','ccl']]
        return df3

    @staticmethod
    def mayorista():
        df = pd.read_csv('dol-mayor.csv')
        df['fecha'] = pd.to_datetime(df['fecha'],  format='%Y-%m-%d')
        df.rename(columns={'ultimo': 'mayorista'}, inplace=True)
        return df[['mayorista', 'fecha']]

    @staticmethod
    def riesgo_pais():
        df2 = pd.read_csv('riesgo_pais.csv')
        df2.fillna(method='ffill', inplace=True)
        df2['fecha'] = pd.to_datetime(df2['fecha'],  format='%Y-%m-%d')
        df2.rename(columns={'ultimo': 'riesgo'}, inplace=True)
        df2 = df2[['fecha','riesgo']]
        return df2

    @staticmethod
    def merval():
        df2 = pd.read_csv('merval.csv')
        df2.fillna(method='ffill', inplace=True)
        df2['fecha'] = pd.to_datetime(df2['fecha'],  format='%Y-%m-%d')
        df2.rename(columns={'cierre': 'merval'}, inplace=True)
        df2 = df2[['fecha','merval']]
        return df2


    @staticmethod
    def leer_bonos():
        bonos_dict = {}
        bonos = ['al30', 'al30d', 'al41', 'al41d', 'gd41', 'gd41d', 'gd30', 'gd30d', 'gd38', 'gd38d']
        for bono in bonos:
            print(bono)
            df1 = pd.read_csv(f'{bono}.csv')
            df1['fecha'] = pd.to_datetime(df1['fecha'],  format='%Y-%m-%d')
            df1.rename(columns={'cierre': bono, 'volumen': f'vol_{bono}'}, inplace=True)
            df1[bono].replace(to_replace=0, method='ffill', inplace=True)
            df1.fillna(method='ffill', inplace=True)
            df1 = df1[['fecha',bono, f'vol_{bono}']]
            bonos_dict[bono] = df1

        return bonos_dict

def create_df_wrk():
    merval = LeerCSVS.merval()
    dolar_may = LeerCSVS.mayorista()
    refs = FromYF.referencias()
    bonos_dict = LeerCSVS.leer_bonos()
    df_riesgo_pais = LeerCSVS.riesgo_pais()
    df_ccl = LeerCSVS.ccl()

    df_merge = bonos_dict['al30'].merge(df_riesgo_pais, on='fecha')
    df_merge = df_merge.merge(bonos_dict['gd30'], on='fecha')
    df_merge = df_merge.merge(bonos_dict['al41'], on='fecha')
    df_merge = df_merge.merge(bonos_dict['gd41'], on='fecha')

    df_merge = df_merge.merge(bonos_dict['al30d'], on='fecha')
    df_merge = df_merge.merge(bonos_dict['gd30d'], on='fecha')
    df_merge = df_merge.merge(bonos_dict['al41d'], on='fecha')
    df_merge = df_merge.merge(bonos_dict['gd41d'], on='fecha')



    df_merge = df_merge.merge(bonos_dict['gd38'], on='fecha')
    df_merge = df_merge.merge(bonos_dict['gd38d'], on='fecha')
    
    df_merge = df_merge.merge(refs, on='fecha')
    df_merge = df_merge.merge(dolar_may, on='fecha')
    df_merge = df_merge.merge(merval, on='fecha')
    df_merge = df_merge.merge(df_ccl, on='fecha')


    df_merge['al30ccl'] = df_merge.al30 / df_merge.ccl
    df_merge['al41ccl'] = df_merge.al41 / df_merge.ccl
    df_merge['gd41ccl'] = df_merge.gd41 / df_merge.ccl
    df_merge['gd30ccl'] = df_merge.gd30 / df_merge.ccl
    df_merge['gd38ccl'] = df_merge.gd38 / df_merge.ccl

    return df_merge

def delta_time_years(date2: str, date1):
    end_date = datetime.strptime(date2, "%d/%m/%y").date()
    time_to_finish = end_date - date1.to_pydatetime().date()
    time_to_finish = time_to_finish.total_seconds()/(3600*24*DAYS_IN_A_YEAR)

    return time_to_finish

def valor_bono_disc(pair_pagos, tasa, pagos_p_a=2):
    valor = 0
    for e in pair_pagos:
        v = e[1]/np.power(1+tasa/pagos_p_a, pagos_p_a*e[0])
        valor += v 
    return valor

def get_nominals(bono, today):

    pagos = bono["pagos"]
    init_date = pagos[0][0]

    time_to_finish = delta_time_years(pagos[len(pagos)-1][0], today) 
    pairs_time_pagos = []
    for pago in pagos:
        dia_pago = datetime.strptime(pago[0], "%d/%m/%y").date()
        if dia_pago < today.date():
            continue
        time_to_pago = delta_time_years(pago[0], today)
        renta = pago[1]
        amortizacion = pago[2]

        r_mas_a = renta + amortizacion

        pairs_time_pagos.append((time_to_pago, r_mas_a))

    return pairs_time_pagos

def curva_v_r(bono, fecha):
    rs = np.linspace(0.0001, 1.0, 100)
    vs = np.zeros(100)
    pair_time_pagos = get_nominals(bono, fecha)   
    for i, r in enumerate(rs):
        vs[i] = valor_bono_disc(pair_time_pagos, r)
    return rs, vs
 
def corr_bono(bono, df_merge):
    with open('bonos.pkl', 'rb') as f:
        bonos = pickle.load(f)

    estructura = bonos[bono.upper()]
    tir = np.zeros(df_merge.shape[0]) 
    for i,r in df_merge.iterrows():
        tasa, precio = curva_v_r(estructura, r.fecha)
        ftir_precio = Fit.polyModel(precio, tasa)
        tir[i] = ftir_precio(r.al30d)
    df_merge[f'tir_{bono}d'] = tir

    m_corr = df_merge[['mayorista', 'merval', 'ccl', 'EEM', f'{bono}d', f'{bono}ccl', 
         'riesgo', '^TNX', '^TYX', '^FVX', '^IRX', f'tir_{bono}d']].corr()

    return m_corr


if __name__ == '__main__':
   
    df_merge = create_df_wrk()
    with open('df_wrk.pkl', 'wb') as f:
         pickle.dump(df_merge, f, protocol=pickle.HIGHEST_PROTOCOL )
    
    with open('df_wrk.pkl', 'rb') as f:
        df_merge = pickle.load(f)

    fig, ax = plt.subplots()
    ax.plot(df_merge.fecha, df_merge.al30d, 'g', label='al30d')
    ax.plot(df_merge.fecha, df_merge.al30ccl, 'b', label='al30ccl')
    ax.plot(df_merge.fecha, df_merge.gd30d, 'r', label='gd30d')
    ax.plot(df_merge.fecha, df_merge.gd30ccl, 'orange', label='gd30ccl')

    ax.legend(handlelength=4)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(df_merge.fecha, df_merge.gd30d / df_merge.al30d, 'r', label='ratio gd30d:al30d')

    ax.legend(handlelength=4)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(df_merge.fecha, df_merge.al41d, 'g', label='al41d')
    ax.plot(df_merge.fecha, df_merge.al41ccl, 'b', label='al41ccl')
    ax.plot(df_merge.fecha, df_merge.gd41d, 'r', label='gd41d')
    ax.plot(df_merge.fecha, df_merge.gd41ccl, 'orange', label='gd41ccl')

    ax.legend(handlelength=4)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(df_merge.fecha, df_merge.gd41d / df_merge.al41d, 'r', label='ratio gd41d:al41d')

    ax.legend(handlelength=4)
    plt.show()

    bono = 'al30'
    m_corr_al30 = corr_bono(bono, df_merge)

    print(m_corr_al30)
    bono = 'al41'
    m_corr_al41 = corr_bono(bono, df_merge)

    print(m_corr_al41)


    for bono in ['al30d', 'al41d']:
        for b_usa in ['^TNX', '^TYX', '^FVX']:
            slope, intercept, r, p, std_err = stats.linregress(df_merge[b_usa], df_merge[f'tir_{bono}'])
            fig, ax = plt.subplots()

            if bono == 'al41d':
                poly = Fit.polyModel(df_merge[b_usa], df_merge[f'tir_{bono}'], degree = 3)
                r_bono = poly(df_merge[b_usa])
                ax.scatter(df_merge[b_usa], r_bono,
                           label="poly")

            ax.scatter(df_merge[b_usa], df_merge[f'tir_{bono}'], c="green", alpha=0.5, marker=r'$\clubsuit$',
                       label="scatter")
            ax.scatter(df_merge[b_usa], df_merge[b_usa] * slope + intercept,
                       label="regresion")
            ax.set_xlabel(b_usa)
            ax.set_ylabel(bono)
            ax.legend()
            plt.show()

            print(f'{bono} vs {b_usa}  slope, intercept, r, p, std_err')
            print(slope, intercept, r, p, std_err)
