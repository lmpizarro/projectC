from scrapers import scrap_bonos_rava
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def cash_flow(flujo, laminas: int = 1000):
    today = datetime.now()
    flujo.fillna(0, inplace=True)
    flujo['fecha'] = pd.to_datetime(flujo['fecha'],format= '%Y-%m-%d')
    flujo[['cupon', 'renta', 'amortizacion']] = laminas * flujo[['cupon', 'renta', 'amortizacion']]
    flujo['acumulado'] = flujo.cupon.cumsum()
    flujo_inicial = -flujo.acumulado.iloc[0]
    flujo['cupon_precio'] = flujo.cupon / flujo_inicial
    flujo['acumu_precio'] = flujo.acumulado / flujo_inicial
    flujo['dias_cupon'] = (flujo.fecha - today).dt.days

    print(flujo)

def test():
    res = scrap_bonos_rava('gd29')

    coti_hist = pd.DataFrame(res['coti_hist'])

    print(coti_hist.head())
    flujo = pd.DataFrame(res['flujofondos']['flujofondos'])

    dolar = (res['flujofondos']['dolar'])
    tir = (res['flujofondos']['tir'])
    duration = (res['flujofondos']['duration'])

    cash_flow(flujo)
    print(res.keys())

    print(res['cotizaciones'][0])

    print(res['cuad_tecnico'])

def coti_hist(res):
    return pd.DataFrame(res['coti_hist'])

def ratio_bond_usd(year: str = 29):
    hist_gd = coti_hist(scrap_bonos_rava(f'gd{year}'))
    key_bond = f'al{year}' if year != 38 else f'ae{year}'
    hist_al = coti_hist(scrap_bonos_rava(key_bond))

    cierre_gd = hist_gd[['fecha', 'cierre', 'usd_cierre']]
    cierre_al = hist_al[['fecha', 'cierre', 'usd_cierre']]
    mrg = pd.merge(cierre_al, cierre_gd, on='fecha', suffixes=(f'_al{year}', f'_gd{year}'))
    mrg[f'usd_al{year}'] = mrg[f'cierre_al{year}']/mrg[f'usd_cierre_al{year}']
    mrg[f'usd_gd{year}'] = mrg[f'cierre_gd{year}']/mrg[f'usd_cierre_gd{year}']
    mrg[f'ratio_{year}'] = mrg[f'cierre_gd{year}']/mrg[f'cierre_al{year}']
    mrg[f'ratio_usd_{year}'] = mrg[f'usd_cierre_gd{year}']/mrg[f'usd_cierre_al{year}']

    mrg[f'ewm_ratio_{year}'] = mrg[f'ratio_{year}'].ewm(alpha=0.1).mean()
    mrg[f'z_signal_{year}'] = mrg[f'ratio_{year}'] - mrg[f'ewm_ratio_{year}']
    mrg[f'ewm_z_signal_{year}'] = mrg[f'z_signal_{year}'].ewm(alpha=0.25).mean()

    print(mrg[f'ratio_{year}'].mean())
    st_dev = mrg[f'z_signal_{year}'].std()
    print(mrg.iloc[-1][f'ratio_{year}'])

    plt.title(f'g_{year}:al_{year}')
    plt.plot(mrg[f'z_signal_{year}'])
    plt.plot(mrg[f'ewm_z_signal_{year}'])

    plt.axhline(y=0.0, color='k', linestyle='-')
    plt.axhline(y=1.5*st_dev, color='y', linestyle='-')
    plt.axhline(y=st_dev, color='r', linestyle='-')
    plt.axhline(y=-st_dev, color='g', linestyle='-')
    plt.axhline(y=-1.5*st_dev, color='y', linestyle='-')
    plt.show()

def bonos_dolar():
    for y in [29, 30, 35, 38, 41]:
        ratio_bond_usd(y)

def bono_pesos(ticker: str = 'PARP'):
    res = scrap_bonos_rava(ticker)
    hist_gd = coti_hist(res)
    
    try:
        flujo = pd.DataFrame(res['flujofondos']['flujofondos'])
        if len(flujo) > 0:
            cash_flow(flujo)
    except:
        pass


    tir = 0; duration = 0
    try:
        tir = (res['flujofondos']['tir'])
    except:
        pass
    try:
        duration = (res['flujofondos']['duration'])
    except:
        pass

    print(f'{ticker} {tir} {duration}')
    plt.title(ticker)
    try:
        if 'usd_cierre' in hist_gd:
            plt.plot(hist_gd.usd_cierre)
        else:
            plt.plot(hist_gd.cierre)
    except:
        return

    plt.show()


def test_pesos():
    duales = ['TDJ23', 'TDL23', 'TDS23', 'TV23', 'TV24']
    txs =  ['TX23', 'T2X3', 'TX24', 'T2X4', 'TX25', 'TX26', 'TX28']
    en_pesos = ['CUAP', 'DICP', 'DIP0', 'PARP', 'BA37D', 'BDC24', 'BDC28', 'PBA25', 'TO26', 'TO23']
    for ticker in en_pesos:
        bono_pesos(ticker)

test_pesos()