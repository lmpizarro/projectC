from config import urls
import pandas as pd

def scrap_bonistas_ticker(especie):
    url = f"{urls['bonistas_com']}/md/{especie}"
    """
        https://towardsdatascience.com/a-guide-to-scraping-html-tables-with-pandas-and-beautifulsoup-7fc24c331cf7
    """
    dfs = pd.read_html(url)

    return {'metricas1': dfs[0],
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




if __name__ == '__main__':
    tickers = scrap_bonistas_main()

    for ticker in tickers:
        print(ticker)
        dict_metricas = scrap_bonistas_ticker(ticker)

        print(dict_metricas)