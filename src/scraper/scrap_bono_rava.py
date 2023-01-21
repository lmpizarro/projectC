
from bs4 import BeautifulSoup
import pandas as pd
import requests
import json

urls = {"bonos": "https://www.rava.com/perfil"}


def scrap_bonos_rava(especie):
    url = urls['bonos'] + '/' + especie
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='html.parser')
    table = soup.find('main').find('perfil-p')
    
    coti_hist = json.loads(table.attrs[':res'])['coti_hist']
    for e in coti_hist:
        print(e)

scrap_bonos_rava('gd41')


