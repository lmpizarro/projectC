import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import pandas as pd

from config import urls


def scrap_bonos_rava(especie):
    url = f"{urls['bonos_rava']}/{especie}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, features="html.parser")
    table = soup.find("main").find("perfil-p")

    res = json.loads(table.attrs[":res"])
    return res


def scrap_cedear_rava():
    url = urls["cedears"]
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, features="html.parser")
    table = soup.find("main").find("cedears-p")

    body = json.loads(table.attrs[":datos"])["body"]
    
    return [b["simbolo"] for b in body]

if __name__ == '__main__':
    result = scrap_bonos_rava('TX28')
    coti_hist = pd.DataFrame(result["coti_hist"])

    print(coti_hist)

    cedears  = scrap_cedear_rava()

    print(cedears)