import yfinance as yf
from datetime import datetime
import pandas as pd
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import numpy as np

def get_vol_surface(ticker:str, rate_structure):

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

    # Convert our vol surface to dataframe for each option price with parameters
    volSurfaceLong = volSurface.melt(ignore_index=False).reset_index()
    volSurfaceLong.columns = ['maturity', 'strike', 'price']
    # Calculate the risk free rate for each maturity using the fitted yield curve

    yield_maturities = rate_structure['maturities']
    yields = rate_structure['yields']
    curve_fit, status = calibrate_nss_ols(yield_maturities,yields)

    volSurfaceLong['rate'] = volSurfaceLong['maturity'].apply(curve_fit)
    print(volSurfaceLong.head())

    return volSurfaceLong, S0

