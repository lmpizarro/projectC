import pandas as pd
import yfinance as yf

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

def dolares_ambito(tipo="CCL", desde="2020-04-20", hasta="2023-04-23"):
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


def dolar_may_ccl(hasta="2023-04-20"):
    df_may = dolares_ambito(tipo="MAY", hasta=hasta)
    # print(df_may.tail())
    df_ccl = dolares_ambito(tipo="CCL", hasta=hasta)
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


def ccl_gap():
    tickers = ["GGAL", "GGAL.BA", "AAPL.BA", "AAPL", "ARS=X", "M.BA"]
    df_close = yf.download(tickers, start="2015-01-01", auto_adjust=True)["Close"]
    df_close["ccl1"] = 10 * df_close["GGAL.BA"] / df_close["GGAL"]
    df_close["ccl2"] = 10 * df_close["AAPL.BA"] / df_close["AAPL"]
    df_close["ccl"] = 0.5 * (df_close.ccl1 + df_close.ccl2)
    # Merval en CCL
    df_close["m_ccl"] = df_close["M.BA"] / df_close["ccl"]
    df_close = df_close[["ccl", "ARS=X", "m_ccl"]]
    df_close["gap"] = (df_close["ccl"] - df_close["ARS=X"]) / df_close["ARS=X"]
    df_close.dropna(inplace=True)

    return df_close


def test_ccl():
    import matplotlib.pyplot as plt

    df_ccl = ccl_gap()

    print(df_ccl.head())
    print(df_ccl.tail())

    plt.plot(df_ccl.gap)
    plt.plot(df_ccl.gap.rolling(200).mean())
    plt.show()
    plt.plot(df_ccl["m_ccl"])
    plt.show()


def test_ambito():
    df = dolares_ambito(tipo="OFI")
    print(df.tail())
    df = dolares_ambito(tipo="NAC")
    print(df.tail())
    df = dolares_ambito(tipo="MEP")
    print(df.tail())


if __name__ == "__main__":
    # test_ccl()
    # test01()
    # test_ambito()
    df_mep = dolar_mep(desde="2015-04-20")

    print(df_mep.tail())
