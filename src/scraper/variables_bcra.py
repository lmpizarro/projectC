import requests
import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

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
    "inflacion": {"Serie": "7931",
                  "Detalle":"Inflación mensual (variación en )"}
}

def variables_bcra(tipo="cer", desde="2016-04-01"):
    url = "https://www.bcra.gob.ar/PublicacionesEstadisticas/Principales_variables_datos.asp"

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

    if tipo != "reservas":
        df_cer = df_cer.apply(lambda x: x.str.replace(",", "."))
    df_cer["Fecha"] = pd.to_datetime(df_cer["Fecha"], format="%d/%m/%Y").dt.date
    df_cer[["Valor"]] = df_cer[["Valor"]].astype("float64")
    df_cer.set_index("Fecha", inplace=True)
    df_cer.rename(columns={"Valor": tipo}, inplace=True)
    if tipo == "inflacion":
        #df_cer["inflacion_acc"] = (1.0 + df_cer["inflacion"]/100).cumprod()
        ...
    return df_cer


def dolar_mep(desde="2020-04-20", hasta=None):
    from datetime import datetime

    if not hasta:
        hasta = datetime.now().date().strftime("%Y-%m-%d")
        print(hasta)
    
    # MEP ordenado de mayor a menor comienza 24 03 2020
    url = f"https://mercados.ambito.com/dolarrava/mep/historico-general/{desde}/{hasta}"
    df_dolar = pd.read_json(url)
    df_dolar = df_dolar.reindex(index=df_dolar.index[::-1]).reset_index()
    df_dolar = df_dolar.drop(columns=["index"])
    df_dolar = df_dolar.apply(lambda x: x.str.replace(",", "."))
    df_dolar = df_dolar[:-1]
    df_dolar[[1]] = df_dolar[[1]].astype("float64")


    df_dolar = df_dolar[:-1]

    df_dolar = df_dolar.rename(columns={0: "Fecha", 1: "mep"})
    df_dolar["Fecha"] = pd.to_datetime(df_dolar["Fecha"], format="%d/%m/%Y").dt.date
    df_dolar.set_index("Fecha", inplace=True)

    return df_dolar


def ccl(start="2015-01-01"):
    tickers = ["GGAL", "GGAL.BA", "AAPL.BA", "AAPL", "ARS=X", "YPF", "YPFD.BA"]
    df_close = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    df_close["cclgal"] = 10 * df_close["GGAL.BA"] / df_close["GGAL"]
    df_close["cclaapl"] = 10 * df_close["AAPL.BA"] / df_close["AAPL"]
    df_close["cclypf"] = df_close["YPFD.BA"] / df_close["YPF"]
    df_close["ccl"] = (df_close.cclgal + df_close.cclaapl + df_close.cclypf) / 3
    df_close["cclars"] = df_close.ccl / df_close["ARS=X"] - 1
    df_close.dropna(inplace=True)

    return df_close

def merval(start="2015-01-01"):
    # log_return = np.log(vfiax_monthly.open / vfiax_monthly.open.shift())

    tickers = ["ARS=X", "M.BA"]
    df_close = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    df_close.dropna(inplace=True)

    # df_close["pct_change"] = (1.0 + df_close["M.BA"].pct_change())
    # df_close["ret"] = df_close["pct_change"].cumprod()
    # df_close["date"] = df_close.index

    df_close["MUSD"] = df_close["M.BA"] / df_close["ARS=X"]

    # df_close.dropna(inplace=True)

    # df_monthly = df_close.resample('M')["M.BA"].sum().to_frame()

    # df_monthly = df_close.loc[df_close.groupby(pd.Grouper(key='date', freq='1M')).date.idxmax()]

    return df_close



if __name__ == "__main__":
    dfInflation = variables_bcra(tipo="inflacion", desde="2015-01-01")
    dfCER = variables_bcra(tipo="cer", desde="2015-01-01")
    dfMayorista = variables_bcra(tipo="mayorista", desde="2015-01-01")
    dfMerval = merval()
    dfCcl = ccl()

    print(dfCcl.tail())

    print(dfMerval.tail())

    print(dfInflation.tail(), dfInflation.shape)

    print(dfCER.tail(), dfCER.shape)

    print(dfMayorista.tail(), dfMayorista.shape)

    dfMayCer = pd.merge(dfMayorista, dfCER, left_index=True, right_index=True)
    dfMayCer = pd.merge(dfMayCer, dfMerval, left_index=True, right_index=True)
    dfMayCer = pd.merge(dfMayCer, dfCcl, left_index=True, right_index=True)
    dfMayCer['rMayCer'] = dfMayCer['mayorista'] / dfMayCer['cer']
    dfMayCer['rCclCer'] = dfMayCer['ccl'] / dfMayCer['cer']
    dfMayCer['rMerARSCer'] = dfMayCer['M.BA'] / dfMayCer['cer']
    dfMayCer['MerCcl'] = dfMayCer['M.BA'] / dfMayCer['ccl']
    dfMayCer['rMerCclCer'] = dfMayCer['MerCcl'] / dfMayCer['cer']
    dfMayCer = dfMayCer.truncate(before="2019-12-30") 
    print(dfMayCer.keys())

    figure, axis = plt.subplots(2, 1)

    axis[0].plot(dfMayCer.rMayCer)
    axis[0].axhline(y=dfMayCer.rMayCer.mean(), color = 'r')
    axis[0].set_title("Mayorista/CER")

    axis[1].plot(dfMayCer.rCclCer)
    axis[1].axhline(y=dfMayCer.rCclCer.mean(), color = 'g')
    axis[1].set_title("CCL/CER")
    plt.show()

    figure, axis = plt.subplots(2, 1)
    axis[0].plot(dfMayCer.rMerARSCer)
    axis[0].axhline(y=dfMayCer.rMerARSCer.mean(), color = 'g')
    axis[0].set_title("MervalARS/CER")

    axis[1].plot(dfMayCer.rMerCclCer)
    axis[1].axhline(y=dfMayCer.rMerCclCer.mean(), color = 'y')
    axis[1].set_title("MervalCCL/CER")
    plt.show()
