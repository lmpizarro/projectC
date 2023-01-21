from scrapers import scrap_bonos_rava
import pandas as pd

def test():
    res = scrap_bonos_rava('gd41')

    coti_hist = pd.DataFrame(res['coti_hist'])

    print(coti_hist.head())

    flujo = pd.DataFrame(res['flujofondos']['flujofondos'])
    dolar = (res['flujofondos']['dolar'])
    tir = (res['flujofondos']['tir'])
    duration = (res['flujofondos']['duration'])

    print(flujo.head())
    print(res.keys())

    print(res['cotizaciones'][0])

    print(res['cuad_tecnico'])

def coti_hist(res):
    return pd.DataFrame(res['coti_hist'])

def ratio(year: str = 29):
    hist_gd = coti_hist(scrap_bonos_rava(f'gd{year}'))
    hist_al = coti_hist(scrap_bonos_rava(f'al{year}'))

    cierre_gd = hist_gd[['fecha', 'cierre', 'usd_cierre']]
    cierre_al = hist_al[['fecha', 'cierre', 'usd_cierre']]
    print(cierre_gd)
    print(cierre_al)

    mrg = pd.merge(cierre_al, cierre_gd, on='fecha', suffixes=(f'_al{year}', f'_gd{year}'))
    mrg[f'usd_al{year}'] = mrg[f'cierre_al{year}']/mrg[f'usd_cierre_al{year}']
    mrg[f'usd_gd{year}'] = mrg[f'cierre_gd{year}']/mrg[f'usd_cierre_gd{year}']
    mrg[f'ratio_{year}'] = mrg[f'cierre_gd{year}']/mrg[f'cierre_al{year}']
    mrg[f'ratio_usd_{year}'] = mrg[f'usd_cierre_gd{year}']/mrg[f'usd_cierre_al{year}']
    print(mrg)

ratio(41)
