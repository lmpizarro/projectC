import matplotlib.pyplot as plt
import yfinance as yf

from wavelet.denoise import deno_wvlt
from butter.filter import min_lp

if __name__ == '__main__':
    symbols = ['KO', 'PG', 'PEP']

    df = yf.download(symbols, '2015-2-1', '2022-03-07')['Adj Close']

    deno_wvlt(symbols, df)
    min_lp(symbols, df)

    for s in symbols:

        plt.plot(df[s+'_deno'])
        plt.show()
        noise = df[s]- df[s+'_deno']
        plt.plot(noise)
        plt.show()
        plt.hist(noise, bins=50)
        plt.show()

