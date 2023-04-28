from config import urls, url_treasury_by_year
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import pandas as pd


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
    symbolos = []
    for b in body:
        symbolos.append(b["simbolo"])
    return symbolos


import pandas as pd


def dolares(tipo="CCL", desde="2020-04-20", hasta="2023-04-23"):
    if tipo == "MAY":
        # Mayorista ordenado de menor a mayor comienza 07 03 2013
        url = f"https://mercados.ambito.com//dolar/mayorista/grafico/{desde}/{hasta}"
    elif tipo == "CCL":
        # CCL ordenado de mayor a menor comienza 07 03 2013
        url = f"https://mercados.ambito.com/dolarrava/cl/historico-general/{desde}/{hasta}"
    elif tipo == "MEP":
        # MEP ordenado de mayor a menor comienza 24 03 2020
        url = f"https://mercados.ambito.com/dolarrava/mep/historico-general/{desde}/{hasta}"
    elif tipo == "OFI":
        url = f"https://mercados.ambito.com/dolar/oficial/grafico/{desde}/{hasta}"
    elif tipo == "NAC":
        url = (
            f"https://mercados.ambito.com/dolarnacion/historico-general/{desde}/{hasta}"
        )

    df_dolar = pd.read_json(url)
    if tipo == "NAC":
        df_dolar = df_dolar[1:]
        df_dolar = df_dolar.apply(lambda x: x.str.replace(",", "."))
        df_dolar[[1, 2]] = df_dolar[[1, 2]].astype("float64")
        df_dolar[1] = (df_dolar[1] + df_dolar[2]) / 2
        df_dolar = df_dolar[[0, 1]]

    if tipo == "CCL" or tipo == "MEP":
        df_dolar = df_dolar.reindex(index=df_dolar.index[::-1]).reset_index()
        df_dolar = df_dolar.drop(columns=["index"])
        df_dolar = df_dolar.apply(lambda x: x.str.replace(",", "."))
        df_dolar = df_dolar[:-1]
        df_dolar[[1]] = df_dolar[[1]].astype("float64")

    if tipo == "MAY" or tipo == "OFI":
        df_dolar = df_dolar[1:]
    elif tipo == "MEP" or tipo == "CCL":
        df_dolar = df_dolar[:-1]

    df_dolar = df_dolar.rename(columns={0: "fecha", 1: tipo})

    return df_dolar


# df = dolares(tipo='OFI')
# print(df.tail())
# df = dolares(tipo='NAC')
# print(df.tail())
# df = dolares(tipo='MEP')
# print(df.tail())
def dolar_may_ccl(hasta="2023-04-20"):
    df_may = dolares(tipo="MAY", hasta=hasta)
    # print(df_may.tail())
    df_ccl = dolares(tipo="CCL", hasta=hasta)
    # print(df_ccl.tail())

    results = df_may.merge(df_ccl, on="fecha", how="left")
    results.dropna(inplace=True)
    results["brecha"] = (results["CCL"] - results["MAY"]) / results["MAY"]
    results["brecha"] = pd.to_numeric(results["brecha"])
    return results


def test01():
    dol_may_ccl = dolar_may_ccl()
    print(dol_may_ccl.brecha.min(), dol_may_ccl.brecha.max(), dol_may_ccl.brecha.mean())
    min_arg = dol_may_ccl.brecha.argmin(skipna=True)
    max_arg = dol_may_ccl.brecha.argmax(skipna=True)

    print("min ", dol_may_ccl.iloc[min_arg].fecha)
    print("max ", dol_may_ccl.iloc[max_arg].fecha)

    # plt.plot(dol_may_ccl.brecha)
    # plt.plot(dol_may_ccl.brecha.rolling(60).mean())
    # plt.show()


def variables_bcra(tipo="cer", desde="2016-04-20"):
    url = "https://www.bcra.gob.ar/PublicacionesEstadisticas/Principales_variables_datos.asp"
    settings = {
        "cer": {"Serie": "3540", "Detalle": "CER (Base 2.2.2002=1)"},
        "badlar": {
            "Serie": "7935",
            "Detalle": "BADLAR en pesos de bancos privados (en  e.a.)",
        },
        "TEAPolMon": {
            "Serie": "7936",
            "Detalle": "Tasa de Política Monetaria (en  e.a.)",
        },
        "mayorista": {
            "Serie": "272",
            "Detalle": "Tipo de Cambio Mayorista ($ por US$) Comunicación A 3500 - Referencia",
        },
        "TEAPF": {
            "Serie": "7939",
            "Detalle": "Tasa mínima para plazos fijos de personas humanas hasta $10 millones (en  e.a. para depósitos a 30 días)",
        },
        "inflacion": {"Serie": "7931", "Detalle": "Inflación mensual (variación en )"},
        "inflacionIA": {
            "Serie": "7932",
            "Detalle": "Inflación interanual (variación en i.a.)",
        },
        "reservas": {
            "Serie": "246",
            "Detalle": "Reservas Internacionales del BCRA (en millones de dólares - cifras provisorias sujetas a cambio de valuación)",
        },
    }
    today = datetime.now().date()
    month = (
        f"0{today.month}" if today.month >= 1 and today.month <= 9 else f"{today.month}"
    )
    hasta = f"{today.year}-{month}-{today.day}"
    data = {
        "primeravez": "1",
        "fecha_desde": desde,
        "fecha_hasta": hasta,
        "serie": settings[tipo]["Serie"],
        "series1": "0",
        "series2": "0",
        "series3": "0",
        "series4": "0",
        "detalle": settings[tipo]["Detalle"],
    }
    resp = requests.post(url=url, data=data, headers={"User-Agent": "Mozilla/5.0"})

    r_text = resp.text

    df_cer = pd.read_html(r_text, thousands=".")[0]
    print(df_cer.tail())

    if tipo != "reservas":
        df_cer = df_cer.apply(lambda x: x.str.replace(",", "."))
    df_cer["Fecha"] = pd.to_datetime(df_cer["Fecha"], format="%d/%m/%Y").dt.date
    df_cer[["Valor"]] = df_cer[["Valor"]].astype("float64")
    df_cer.set_index("Fecha", inplace=True)
    df_cer.rename(columns={"Valor": tipo}, inplace=True)
    return df_cer


import yfinance as yf


def ccl_gap():
    tickers = ["GGAL", "GGAL.BA", "AAPL.BA", "AAPL", "ARS=X"]
    df_close = yf.download(tickers, start="2012-04-20", auto_adjust=True)["Close"]
    df_close["ccl1"] = 10 * df_close["GGAL.BA"] / df_close["GGAL"]
    df_close["ccl2"] = 10 * df_close["AAPL.BA"] / df_close["AAPL"]
    df_close["ccl"] = 0.5 * (df_close.ccl1 + df_close.ccl2)
    df_close = df_close[["ccl", "ARS=X"]]
    df_close["gap"] = (df_close["ccl"] - df_close["ARS=X"]) / df_close["ARS=X"]
    df_close.dropna(inplace=True)

    return df_close

class USBonds:
    def __init__(self, year: int = 2023) -> None:
        self.year = year
        self.yield_curve = self.treasury_yield_curve()

    def treasury_yield_curve(self):
        string_terms = [
            "1 Mo",
            "2 Mo",
            "3 Mo",
            "4 Mo",
            "6 Mo",
            "1 Yr",
            "2 Yr",
            "3 Yr",
            "5 Yr",
            "7 Yr",
            "10 Yr",
            "20 Yr",
            "30 Yr",
        ]

        resp = requests.get(
            url_treasury_by_year(year=self.year), headers={"User-Agent": "Mozilla/5.0"}
        )
        treas_df = pd.read_html(resp.text)

        filter_keys = ["Date"]
        filter_keys.extend(string_terms)
        treas_df = treas_df[0][filter_keys]

        def spliter(r: str):
            s = r.split(" ")
            if s[1] == "Mo":
                d = 30
            else:
                d = 365
            k = int(s[0])

            return d * k

        treas_df["Date"] = pd.to_datetime(treas_df["Date"], format="%m/%d/%Y").dt.date
        string_term_to_days_term = {r: spliter(r) for r in string_terms}
        days_term = string_term_to_days_term.values()
        treas_df.rename(columns=string_term_to_days_term, inplace=True)
        treas_df['mean'] = treas_df[days_term].mean(axis=1)
        treas_df.set_index('Date', inplace=True)

        return treas_df

    def today_mean(self):
        today_year = datetime.now().year
        if self.year != today_year:
            self.year = today_year
            self.yield_curve = self.treasury_yield_curve()
        return self.yield_curve['mean'].iloc[-1]

    def last_curve_points(self):

        terms = np.array(list(self.yield_curve.keys())[0:-1])
        rates = np.array(list(self.yield_curve.tail().iloc[-1])[0:-1])

        return terms, rates

def test_ccl():
    import matplotlib.pyplot as plt

    # df_cer = variables_bcra('cer')
    # print(df_cer.head())
    # print(df_cer.tail())
    df_cer = variables_bcra("reservas", desde="2001-12-17")

    print(df_cer.tail())
    plt.plot(df_cer)
    plt.show()

    # df_cer = variables_bcra('mayorista')
    # # print(df_cer.head())
    # print(df_cer.tail())

    df_ccl = ccl_gap()

    print(df_ccl.head())
    print(df_ccl.tail())

    plt.plot(df_ccl.gap)
    plt.plot(df_ccl.gap.rolling(200).mean())
    plt.show()

import numpy as np
usbond = USBonds()
df_treas = usbond.yield_curve
print(df_treas.tail())
print(usbond.today_mean())
terms, rates = usbond.last_curve_points()

daily_rates = rates / 365
term_rates = daily_rates * terms

print(terms)
print(rates)

import matplotlib.pyplot as plt

plt.plot(df_treas['mean'])
plt.show()

plt.plot(terms, rates)
plt.show()
