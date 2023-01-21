from bs4 import BeautifulSoup
import pandas as pd
import requests
import json

urls = {"bonos": "https://www.rava.com/perfil"}


def scrap_bonos_rava(especie):
    url = f"{urls['bonos']}/{especie}"
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='html.parser')
    table = soup.find('main').find('perfil-p')

    res = json.loads(table.attrs[':res'])
    return res

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

