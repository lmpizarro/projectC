from config import urls
import requests
from bs4 import BeautifulSoup
import pandas as pd
import copy
import json
from finviz import FinViz

def scrap_bonos_rava(especie):
    url = f"{urls['bonos_rava']}/{especie}"
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='html.parser')
    table = soup.find('main').find('perfil-p')

    res = json.loads(table.attrs[':res'])
    return res


def scrap_cedear_rava():
    url = urls['cedears']
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, features='html.parser')
    table = soup.find('main').find('cedears-p')

    body = json.loads(table.attrs[':datos'])['body']
    symbolos = []
    for b in body:
        symbolos.append(b['simbolo'])
    return symbolos


