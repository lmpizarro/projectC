import yfinance as yf
from datetime import datetime
import pandas as pd
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def get_surface_IVKT(ticker:str):

    ticker = yf.Ticker(ticker)
    options_dates = ticker.options

    today = datetime.now().date()
    implied_volatility = pd.DataFrame()
    for d in options_dates:
        ex_date = datetime.strptime(d, "%Y-%m-%d").date()
        time_to_expire = ex_date - today

        calls: pd.DataFrame = ticker.option_chain(d).calls
        """
            ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
             'change', 'percentChange', 'volume', 'openInterest',
             'impliedVolatility', 'inTheMoney', 'contractSize', 'currency']
        """
        imp_vol_strike = calls[['strike', 'impliedVolatility']]
        imp_vol_strike = imp_vol_strike.set_index('strike')
        imp_vol_strike.rename(columns={'impliedVolatility':time_to_expire.days}, inplace=True)
        if implied_volatility.empty:
            implied_volatility = imp_vol_strike
        else:
            implied_volatility = pd.merge(implied_volatility, imp_vol_strike, on='strike')

    return implied_volatility

def draw_IVKT_2D(iv: pd.DataFrame):

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    # keys = T (maturities)
    # x = K
    Ks = iv.index
    maturities = iv.keys()

    cls_= len(maturities) // 2 + 1
    fig, axs = plt.subplots(cls_, 2)

    graph_counter = 0
    for i in range(cls_):
        for j in range(2):
            axs[i, j].plot(Ks, iv[maturities[j]])
            graph_counter += 1
            if graph_counter == len(maturities):
                break

    plt.show()

def draw_IVKT_3D(iv_surface: pd.DataFrame):

    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

    # x maturities
    # y strikes
    # z iv
    x = np.array(iv_surface.columns, dtype="float64")
    y = iv_surface.index
    X,Y = np.meshgrid(x,y)
    Z = np.array(iv_surface, dtype="float64")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.hot, linewidth=0, antialiased=True)
    # ax.plot_wireframe(X, Y, Z, rstride=40, cstride=40)
    plt.show()


def get_surface_PKT(ticker:str):

    market_prices = {}

    Ticker = yf.Ticker(ticker)
    options_dates = Ticker.options
    S0 = Ticker.info['regularMarketPrice']

    for option_date in options_dates:
        calls: pd.DataFrame = Ticker.option_chain(option_date).calls
        calls['price'] = (calls['bid'] + calls['ask']) / 2
        market_prices[option_date] = {}
        market_prices[option_date]['strike'] = list(calls['strike'])
        market_prices[option_date]['price'] = list(calls['price'])

    all_strikes = [v['strike'] for i,v in market_prices.items()]
    common_strikes = set.intersection(*map(set,all_strikes))
    print('Number of common strikes:', len(common_strikes))
    common_strikes = sorted(common_strikes)
    # print(market_prices)

    prices = []
    maturities = []
    today = datetime.now().date()
    for date, v in market_prices.items():
        exp_date = datetime.strptime(date, "%Y-%m-%d").date()
        maturity = exp_date - today
        maturities.append(maturity.days/365.25)

        price = [v['price'][i] for i,x in enumerate(v['strike']) if x in common_strikes]
        prices.append(price)

    price_arr = np.array(prices, dtype=object)

    volSurface = pd.DataFrame(price_arr, index = maturities, columns = common_strikes)
    volSurface = volSurface.iloc[(volSurface.index > 0.04)]

    return volSurface, S0

def yield_curve_fit(rate_structure):
    yield_maturities = rate_structure['maturities']
    yields = rate_structure['yields']
    curve_fit, status = calibrate_nss_ols(yield_maturities,yields)

    return curve_fit

def surface_PKT_to_long(volSurface:pd.DataFrame, rate_structure):

    # Convert our vol surface to dataframe for each option price with parameters
    volSurfaceLong = volSurface.melt(ignore_index=False).reset_index()
    volSurfaceLong.columns = ['maturity', 'strike', 'price']
    # Calculate the risk free rate for each maturity using the fitted yield curve

    volSurfaceLong['rate'] = volSurfaceLong['maturity'].apply(yield_curve_fit(rate_structure))

    return volSurfaceLong

def plotly_PKT_3D(volSurfaceLong: pd.DataFrame):
    import plotly.graph_objects as go
    from plotly.graph_objs import Surface

    fig = go.Figure(data=[go.Mesh3d(x=volSurfaceLong.maturity,
                                    y=volSurfaceLong.strike,
                                    z=volSurfaceLong.price,
                                    color='mediumblue',
                                    opacity=0.55)])

    fig.update_layout(
        title_text='Market Prices (Mesh) vs Calibrated Heston Prices (Markers)',
        scene = dict(xaxis_title='TIME (Years)',
                        yaxis_title='STRIKES (Pts)',
                        zaxis_title='INDEX OPTION PRICE (Pts)'),
        height=800,
        width=800
        )
    fig.show()

def plot_PKT_3D(volSurfacePKT):

    K = np.array(volSurfacePKT.columns, dtype="float64") # strikes
    T = volSurfacePKT.index  # maturities

    X, Y = np.meshgrid(K,T)
    Prices = np.array(volSurfacePKT, dtype="float64")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Prices, cmap=cm.hot, linewidth=0, antialiased=True)
    # ax.plot_wireframe(X, Y, Z, rstride=40, cstride=40)
    plt.show()


if __name__ == '__main__':
    yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    yields = np.array([0.15,0.27,0.50,0.93,1.52,2.13,2.32,2.34,2.37,2.32,2.65,2.52]).astype(float)/100

    rate_structure = {'yields': yields,
                      'maturities': yield_maturities}


    surface_pkt, S0 = get_surface_PKT('AAPL')
    surface_pkt_long = surface_PKT_to_long(surface_pkt, rate_structure)

    plot_PKT_3D(surface_pkt)

    import py_vollib_vectorized


    flag = 'c'
    surface_pkt_long['iv'] = \
        py_vollib_vectorized.implied_volatility.vectorized_implied_volatility(price=surface_pkt_long['price'],
        S=S0, K=surface_pkt_long['strike'],
        t=surface_pkt_long['maturity'],
        r=surface_pkt_long['rate'],
        flag=flag, q=0, return_as='numpy',
        model="black_scholes_merton")


    surface_pkt_long.drop(columns=['price', 'rate'], inplace=True)


    print(surface_pkt_long)

    # https://www.digitalocean.com/community/tutorials/pandas-melt-unmelt-pivot-function
    surface_IVKT_estim = surface_pkt_long.pivot(index='maturity', columns='strike')
    surface_IVKT_estim.fillna(inplace=True, method='ffill')
    print(surface_IVKT_estim)


    surface_IVKT_estim = surface_IVKT_estim.iv
    strikes = surface_IVKT_estim.keys()
    maturities = surface_IVKT_estim.index

    for t in maturities:
        plt.plot(strikes, surface_IVKT_estim.loc[t])
    plt.show()



