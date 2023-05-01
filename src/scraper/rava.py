import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import pandas as pd

from config import urls

DAYS_IN_A_YEAR = 360

def coti_hist(res):
    return pd.DataFrame(res["coti_hist"])


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


def cash_flow(flujo):
    today = datetime.now()
    flujo.fillna(0, inplace=True)
    flujo["fecha"] = pd.to_datetime(flujo["fecha"], format="%Y-%m-%d")
    flujo["acumulado"] = flujo.cupon.cumsum()
    flujo_inicial = -flujo.acumulado.iloc[0]
    flujo["cupon_precio"] = flujo.cupon / flujo_inicial
    flujo["acumu_precio"] = flujo.acumulado / flujo_inicial
    flujo["dias_cupon"] = (flujo.fecha - today).dt.days
    flujo['years_to_coupon'] = flujo["dias_cupon"] / DAYS_IN_A_YEAR

    return flujo


def test():
    res = scrap_bonos_rava("gd29")

    coti_hist = pd.DataFrame(res["coti_hist"])

    print(coti_hist.head())
    flujo = pd.DataFrame(res["flujofondos"]["flujofondos"])

    dolar = res["flujofondos"]["dolar"]
    tir = res["flujofondos"]["tir"]
    duration = res["flujofondos"]["duration"]

    cash_flow(flujo)
    print(res.keys())

    print(res["cotizaciones"][0])

    print(res["cuad_tecnico"])


if __name__ == "__main__":
    result = scrap_bonos_rava("TX28")
    coti_hist = coti_hist(result)

    print(coti_hist)

    cedears = scrap_cedear_rava()

    print(cedears)

    test()
