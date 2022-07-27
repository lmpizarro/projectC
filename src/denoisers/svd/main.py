#!/usr/bin/env python3
# −*− coding:utf-8 −*−

from denoise import Denoiser
import yfinance as yf 
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    symbols = ['AAPL', 'KO']
    df = yf.download(symbols, '2018-01-02')['Adj Close']

    denoiser = Denoiser()
    denoised = denoiser.denoise(df.AAPL, 50)
    df['AAPL_deno'] = denoised
    fig, ax = plt.subplots()
    ax.plot(df.AAPL)
    ax.plot(df.AAPL_deno)
    plt.show()

    df['AAPL_pct'] = df.AAPL.pct_change()
    df['AAPL_log'] = np.log(df.AAPL) - np.log(df.AAPL.shift(1))

    plt.plot(df.AAPL_pct)
    plt.plot(denoised)
    plt.show()

    print((df.AAPL - df.AAPL_deno).sum())

