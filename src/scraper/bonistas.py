from config import urls
import pandas as pd

def scrap_bonistas_ticker(especie):
    url = f"{urls['bonistas_com']}/md/{especie}"
    """
        https://towardsdatascience.com/a-guide-to-scraping-html-tables-with-pandas-and-beautifulsoup-7fc24c331cf7
    """
    dfs = pd.read_html(url)

    return {'ticker': especie,
            'metricas1': dfs[0],
            'metricas2': dfs[1],
            'calendario': dfs[2]}

def scrap_bonistas_main():
    url = f"{urls['bonistas_com']}"

    dfs = pd.read_html(url)
    tickers_index = [0, 2, 4, 8, 10]
    maps = {0:'tasa fija badlar',
            2:'CER',
            4:'USD',
            6: 'CABLE',
            8: 'LEDES',
            10: 'LECER',
            12: 'MEP CCL ARG',
            13: 'MEP CCL NY'}
    tickers = []
    bonos = {}
    for index, df in enumerate(dfs):
        if index < 11 and not (index % 2):
            bonos[index] = df
            if index in tickers_index:
                tickers.extend(list(df.Ticker))
        elif index > 11 and index <= 13:
            bonos[index] = df
            if index in tickers_index:
                tickers.extend(list(df.Ticker))

    for index in bonos:
        print(f'-- {index} ---------- {maps[index].upper()} --------')
        print(bonos[index])

    return tickers




from joblib import Parallel, delayed

if __name__ == '__main__':
    tickers = scrap_bonistas_main()

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
        for i in range(len(metrica['metricas1'])):
            key = metrica['metricas1'].iloc[i]['Descripción']
            value = metrica['metricas1'].iloc[i]['Valor']

            if key == "Variación diaria" and value[0] == '=':
                value = value[2:]

            if key == "Variación diaria":
                key = 'Variación'

            if key == "Valor Técnico":
                key = 'Val.Tec.'

            valores[key] = value
        for i in range(len(metrica['metricas2'])):
            key = metrica['metricas2'].iloc[i]['Métricas']
            value =  metrica['metricas2'].iloc[i]['Valor']
            if key == "Up TIR":
                value = value[2:]

            if key == 'Riesgo - Percentil 5':
                key = 'RP5'
            if key == 'Riesgo - Percentil 1':
                key = 'RP1'
            if key == 'TIR Promedio':
                key = 'TIRProm'

            if ' ' in key:
                key = key.replace(' ', '')

            if key == "UpTIR" and value[0] == '=':
                value = value[2:]


            valores[key] = value.strip()


        valores['PQ'] = metrica['calendario'].iloc[0].FECHA
        metricas_ticker.append(valores)
    df_metricas = pd.DataFrame(metricas_ticker)
    df_metricas.drop('RP1', axis='columns', inplace=True)
    df_metricas.drop('RP5', axis='columns', inplace=True)
    print(df_metricas)
    df_metricas.to_csv('metricas_bonos.csv')