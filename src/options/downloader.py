import yfinance as yf
import numpy as np


stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BRK-B', 'TSLA', 'META', 'JNJ',
          'V', 'TSM', 'XOM', 'UNH', 'WMT', 'JPM', 'MA', 'PG', 'LLY', 'CVX', 'HD',
          'ASML', 'ABBV', 'SPY', 'QQQ', 'DIA', 'GLOB', 'MELI']
stocks_ = ['TSM', 'XOM', 'JPM', 'CVX', 'TM', 'PFE', 'BAC']
stocks_= ['SPY', 'QQQ', 'DIA']
stocks_ = ['YPF', 'BMA', 'TX', 'PAM', 'EDN', 'GGAL', 'LOMA']

def download_stocks(stocks: list=stocks):
    yf_data = yf.download(stocks, group_by=stocks, start='2017-01-01')
    for stock in stocks:
        pass_tuple = (stock, 'Adj Close')
        yf_data[(stock, 'returns')] = yf_data[pass_tuple].pct_change()
        yf_data[(stock, 'log_returns')] = np.log(yf_data[pass_tuple]/yf_data[pass_tuple].shift(1))
        yf_data.dropna(inplace=True)


    print(yf_data.head())

    return yf_data

