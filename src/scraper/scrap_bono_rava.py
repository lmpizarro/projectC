from scrapers import scrap_bonos_rava
import pandas as pd


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

