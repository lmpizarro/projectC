import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def ccl():
    df3 = pd.read_csv('ccl.csv')
    df3.fillna(method='ffill', inplace=True)
    df3['fecha'] = pd.to_datetime(df3['fecha'],  format='%Y-%m-%d')
    df3.rename(columns={'ultimo': 'ccl'}, inplace=True)
    df3 = df3[['fecha','ccl']]
    return df3

def riesgo_pais():
    df2 = pd.read_csv('riesgo_pais.csv')
    df2.fillna(method='ffill', inplace=True)
    df2['fecha'] = pd.to_datetime(df2['fecha'],  format='%Y-%m-%d')
    df2.rename(columns={'ultimo': 'riesgo'}, inplace=True)
    df2 = df2[['fecha','riesgo']]
    return df2

def leer_bonos():
    bonos_dict = {}
    bonos = ['al30', 'al30d', 'gd30', 'gd30d']
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

def referencias(tickers=['EEM', 'GGAL', 'YPF']):
    t = yf.download(tickers,  '2020-01-02')['Adj Close']
    t['fecha'] = t.index

    return t

if __name__ == '__main__':

    refs = referencias()


    bonos_dict = leer_bonos()
    df_riesgo_pais = riesgo_pais()
    df_ccl = ccl()

    df_merge = bonos_dict['al30'].merge(df_riesgo_pais, on='fecha')
    df_merge = df_merge.merge(df_ccl, on='fecha')
    df_merge = df_merge.merge(bonos_dict['gd30d'], on='fecha')
    df_merge = df_merge.merge(bonos_dict['al30d'], on='fecha')
    df_merge = df_merge.merge(bonos_dict['gd30'], on='fecha')
    df_merge = df_merge.merge(refs, on='fecha')

    df_merge['al30usd'] = df_merge.al30 / df_merge.ccl
    df_merge['gd30usd'] = df_merge.gd30 / df_merge.ccl
    print(df_merge.tail())

    plt.plot(df_merge.al30usd-df_merge.gd30d)
    plt.show()

    plt.plot(df_merge.riesgo)
    plt.show()

    m_corr = df_merge[['YPF', 'GGAL','EEM', 'al30d', 'al30usd', 'gd30d', 'gd30usd', 'ccl', 'riesgo']].corr()

    print(m_corr)


