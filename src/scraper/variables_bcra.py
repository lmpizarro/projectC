import requests
import pandas as pd
from datetime import datetime


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


def test_ccl():
    import matplotlib.pyplot as plt

    # df_cer = variables_bcra('cer')
    # print(df_cer.head())
    # print(df_cer.tail())
    df_cer = variables_bcra("reservas", desde="2001-12-17")

    print(df_cer.tail())
    plt.plot(df_cer)
    plt.show()


if __name__ == "__main__":
    variables_bcra()
