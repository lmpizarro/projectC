"""obtiene todos los valores de los bonos en bonistas"""
from config import urls
import pandas as pd
from joblib import Parallel, delayed
from datetime import date, datetime


def scrap_bonistas_ticker(especie):
    url = f"{urls['bonistas_com']}/md/{especie}"
    """
        https://towardsdatascience.com/a-guide-to-scraping-html-tables-with-pandas-and-beautifulsoup-7fc24c331cf7
    """
    dfs = pd.read_html(url)

    return {
        "ticker": especie,
        "metricas1": dfs[0],
        "metricas2": dfs[1],
        "calendario": dfs[2],
    }


def ticker_by_class(bonos: dict):
    class_bonos = {}
    for desc in bonos:
        print(desc)
        if desc in ["CER", "USD", "LEDES", "LECER", "tasa fija badlar"]:
            class_bonos[desc] = list(bonos[desc]["Ticker"])
    bonos_class = {}

    for clas in class_bonos:
        for ticker in class_bonos[clas]:
            bonos_class[ticker] = clas

    return bonos_class


def scrap_bonistas_main():
    url = f"{urls['bonistas_com']}"

    dfs = pd.read_html(url)
    tickers_index = [0, 2, 4, 8, 10]
    maps = {
        0: "tasa fija badlar",
        2: "CER",
        4: "USD",
        6: "CABLE",
        8: "LEDES",
        10: "LECER",
        12: "MEP CCL ARG",
        13: "MEP CCL NY",
    }
    tickers = []
    bonos = {}
    for index, df in enumerate(dfs):
        if index < 11 and not (index % 2):
            bonos[maps[index]] = df
            if index in tickers_index:
                tickers.extend(list(df.Ticker))
        elif index > 11 and index <= 13:
            bonos[maps[index]] = df
            if index in tickers_index:
                tickers.extend(list(df.Ticker))

    return ticker_by_class(bonos)


def fecha_datetime(fecha_pq):
    fecha_pq = fecha_pq.split("-")
    pq_date = date(int(fecha_pq[0]), int(fecha_pq[1]), int(fecha_pq[2]))
    pq_date = datetime.combine(pq_date, datetime.min.time())
    return pq_date


if __name__ == "__main__":
    bonos = scrap_bonistas_main()
    tickers = bonos.keys()

    metricas = []
    """
    for ticker in tickers:
        print(ticker)
        dict_metricas = scrap_bonistas_ticker(ticker)
        metricas.append(dict_metricas)
    """
    metricas = Parallel(n_jobs=6)(delayed(scrap_bonistas_ticker)(i) for i in tickers)
    metricas_ticker = []
    for metrica in metricas:
        valores = {}
        for i in range(len(metrica["metricas1"])):
            key = metrica["metricas1"].iloc[i]["Descripción"]
            value = metrica["metricas1"].iloc[i]["Valor"]

            if key == "Variación diaria" and value[0] == "=":
                value = value[2:]

            if key == "Variación diaria":
                key = "Variación"

            if key == "Valor Técnico":
                key = "Val.Tec."

            if key == "Ticker":
                try:
                    tipo_de_bono = bonos[value.strip()]
                    if tipo_de_bono == "tasa fija badlar":
                        tipo_de_bono = "fija-badlar"
                    valores["TIPO"] = tipo_de_bono
                except:
                    pass

            valores[key] = value
        for i in range(len(metrica["metricas2"])):
            key = metrica["metricas2"].iloc[i]["Métricas"]
            value = metrica["metricas2"].iloc[i]["Valor"]
            if key == "Up TIR":
                value = value[2:]

            if key == "Riesgo - Percentil 5":
                key = "RP5"
            if key == "Riesgo - Percentil 1":
                key = "RP1"
            if key == "TIR Promedio":
                key = "TIRProm"

            if " " in key:
                key = key.replace(" ", "")

            if key == "UpTIR" and value[0] == "=":
                value = value[2:]

            valores[key] = value.strip()

        fecha_pq = metrica["calendario"].iloc[0].FECHA
        last_fecha = metrica["calendario"].iloc[-1].FECHA
        valores["PQ"] = fecha_pq

        pq_date = fecha_datetime(fecha_pq=fecha_pq)

        delta_t = (datetime.now() - pq_date).days

        valores["tPQ"] = -delta_t
        valores["FIN"] = last_fecha
        last_date = fecha_datetime(fecha_pq=last_fecha)
        delta_to_fin = (last_date - datetime.now()).days
        valores["MAT"] = round(delta_to_fin / 360, 4)
        valores["ratMD"] = round(float(valores["MD"]) / valores["MAT"], 4)
        metricas_ticker.append(valores)
    df_metricas = pd.DataFrame(metricas_ticker)

    # df_metricas.drop('RP1', axis='columns', inplace=True)
    # df_metricas.drop('RP5', axis='columns', inplace=True)
    # df_metricas.drop('Variación', axis='columns', inplace=True)

    print(df_metricas)
    df_metricas.to_csv("metricas_bonos.csv")
