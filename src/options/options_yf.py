import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
from matplotlib import cm


yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yields = np.array([2.68, 3.01, 3.20, 3.77, 3.96, 3.85, 3.81, 3.62, 3.56, 3.45, 3.79, 3.52]).astype(float)/100


rate_structure = {'yields': yields,
                  'maturities': yield_maturities}

def yield_curve_fit(rate_structure):
    yield_maturities = rate_structure['maturities']
    yields = rate_structure['yields']
    curve_fit, status = calibrate_nss_ols(yield_maturities,yields)

    return curve_fit

def prices_properties(prices):
    number_of_strikes = 0
    min_strike = 1E6
    max_strike = -1E6

    for expiration in prices:
        strike_min = min(prices[expiration].keys())
        if strike_min < min_strike:
            min_strike = strike_min
        strike_max = max(prices[expiration].keys())
        if strike_max > max_strike:
            max_strike = strike_max
        if len(prices[expiration]) > number_of_strikes:
            number_of_strikes = len(prices[expiration])
    number_of_expirations = len(prices)
    deltas_strike = 2 * (max_strike - min_strike)
    return number_of_expirations, number_of_strikes, min_strike, max_strike, deltas_strike


def get_opt_prices(ticker='AAPL', yield_curve=yield_curve_fit(rate_structure)):
    tckr = yf.Ticker(ticker)
    last_close = tckr.info['regularMarketPrice']
    prices_expir_moneyness = {}
    prices_expir_strike = {}
    for maturity in tckr.options:
        opt = tckr.option_chain(maturity)
        calls = opt.calls
        symbol = calls.iloc[0].contractSymbol
        maturity_in_days = int((pd.Timestamp(maturity) - pd.Timestamp(datetime.now())).days)
        if maturity_in_days < 0:
            continue

        prices_expir_moneyness[maturity_in_days] = {}
        prices_expir_strike[maturity_in_days] = {}

        for i, c in calls.iterrows():

            rT = maturity_in_days * yield_curve(maturity_in_days/365)/365
            s_k = last_close/c.strike
            s_k = np.log(s_k) + rT
            if s_k < .5 and s_k > -.5:
                prices_expir_moneyness[maturity_in_days][s_k] =  c.lastPrice # c.impliedVolatility
                prices_expir_strike[maturity_in_days][c.strike] =  c.lastPrice # c.impliedVolatility

    return prices_expir_moneyness, prices_expir_strike


def get_opt_prices_strike(ticker='AAPL'):
    tckr = yf.Ticker(ticker)
    prices_expir_strike = {}
    for maturity in tckr.options:
        opt = tckr.option_chain(maturity)
        calls = opt.calls
        symbol = calls.iloc[0].contractSymbol
        maturity_in_days = int((pd.Timestamp(maturity) - pd.Timestamp(datetime.now())).days)
        if maturity_in_days < 0:
            continue

        prices_expir_strike[maturity_in_days] = {}

        for i, c in calls.iterrows():
            prices_expir_strike[maturity_in_days][c.strike] =  c.lastPrice # c.impliedVolatility
    return prices_expir_strike


def plot_PKT_3D(volSurfacePKT):

    K = np.array(volSurfacePKT.columns, dtype="float64") # strikes
    T = volSurfacePKT.index.sort_values(ascending=True)  # maturities

    X, Y = np.meshgrid(K,T)
    Prices = np.array(volSurfacePKT, dtype="float64")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Prices, cmap=cm.hot, linewidth=0, antialiased=True)
    # ax.plot_wireframe(X, Y, Z, rstride=40, cstride=40)
    plt.show()


from scipy import interpolate

def moneyness():
    prices_moneyness, prices_strike = get_opt_prices()
    xnew = np.linspace(-0.5, 0.5, 60)
    for maturity in prices_moneyness:
        skew = prices_moneyness[maturity]
        stk = np.asarray([(k, skew[k]) for k in skew])
        x = stk[:, 0]
        y = stk[:, 1]
        f = interpolate.interp1d(x, y, fill_value='extrapolate', kind='linear')
        yinterp = f(xnew)
        prices_moneyness[maturity] = {xn: yinterp[i] for i, xn in enumerate(xnew)}
        plt.plot(xnew, yinterp, 'x-')
    plt.show()

    df_prices = pd.DataFrame(prices_moneyness)
    df_prices.fillna(0, inplace=True)
    plot_PKT_3D(df_prices)
    print(df_prices.head(10))
    print(df_prices.tail(10))
    

    for c in df_prices.columns:
       y = df_prices[c].dropna()
       plt.plot(y.index, y)
       plt.show()

if __name__ == '__main__':
    ticker = 'AAPL'
    tckr = yf.Ticker(ticker)
    last_close = tckr.info['regularMarketPrice']
    limit_inf = last_close * .5
    limit_sup = 1.5 * last_close

    prices_strike = get_opt_prices_strike()
    tuple_props = prices_properties(prices_strike)

    xnew = np.arange(tuple_props[2], tuple_props[3], .5)

    key_to_del = []
    def del_key(d, key):
        d[key/365] = d[key]
        del d[key]

    for maturity in prices_strike:
        skew = prices_strike[maturity]
        stk = np.asarray([(k, skew[k]) for k in skew])
        x = stk[:, 0]
        y = stk[:, 1]
        f = interpolate.interp1d(x, y, fill_value='extrapolate', kind='linear')
        yinterp = f(xnew)
        prices_strike[maturity] = {xn: yinterp[i] for i, xn in enumerate(xnew)}
        key_to_del.append(maturity)

    [del_key(prices_strike, key) for key in key_to_del]

    df_prices = pd.DataFrame(prices_strike)
    df_prices = df_prices.loc[(df_prices.index > 0) & (df_prices.index < limit_sup)]
    df_prices = df_prices.T
    df_prices = df_prices.loc[(df_prices.index < 1.0)]
    df_prices[df_prices < 0] = 0
    df_prices.fillna(0, inplace=True)
    print(df_prices.tail())
    plot_PKT_3D(df_prices)

    for c in df_prices.columns:
       y = df_prices[c].dropna()
       plt.plot(y.index, y, 'x-')
       plt.show()
