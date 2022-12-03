import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_nss_ols


yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yields = np.array([2.68, 3.01, 3.20, 3.77, 3.96, 3.85, 3.81, 3.62, 3.56, 3.45, 3.79, 3.52]).astype(float)/100


rate_structure = {'yields': yields,
                  'maturities': yield_maturities}

def yield_curve_fit(rate_structure):
    yield_maturities = rate_structure['maturities']
    yields = rate_structure['yields']
    curve_fit, status = calibrate_nss_ols(yield_maturities,yields)

    return curve_fit


def get_opt_prices(ticker='AAPL', yield_curve=yield_curve_fit(rate_structure)):
    tckr = yf.Ticker(ticker)
    last_close = tckr.info['regularMarketPrice']
    prices = {}
    for maturity in tckr.options:
        opt = tckr.option_chain(maturity)
        calls = opt.calls
        symbol = calls.iloc[0].contractSymbol
        maturity_in_days = int((pd.Timestamp(maturity) - pd.Timestamp(datetime.now())).days)
        if maturity_in_days < 0:
            continue

        prices[maturity_in_days] = {}
        for i, c in calls.iterrows():

            rT = maturity_in_days * yield_curve(maturity_in_days/365)/365
            s_k = last_close/c.strike
            s_k = np.log(s_k) + rT
            if s_k < .5 and s_k > -.5:
                prices[maturity_in_days][s_k]=  c.lastPrice # c.impliedVolatility
    return prices


from scipy import interpolate
if __name__ == '__main__':
    prices = get_opt_prices()

    xnew = np.linspace(-0.5, 0.5, 60)
    for maturity in prices:
        skew = prices[maturity]
        stk = np.asarray([(k, skew[k]) for k in skew])
        x = stk[:, 0]
        y = stk[:, 1]
        f = interpolate.interp1d(x, y, fill_value='extrapolate', kind='linear')
        yinterp = f(xnew)
        prices[maturity] = {xn: yinterp[i] for i, xn in enumerate(xnew)}
        plt.plot(xnew, yinterp, 'x-')
        plt.show()

    df_prices = pd.DataFrame(prices)
    df_prices.fillna(0, inplace=True)
    print(df_prices.head(10))
    print(df_prices.tail(10))

    for c in df_prices.columns:
       y = df_prices[c].dropna()
       plt.plot(y.index, y)
       plt.show()
