# SuperFastPython.com
# example of thread local storage
import pandas as pd

url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{}/all?type=daily_treasury_yield_curve&field_tdr_date_value=2018&page&_format=csv"


from time import sleep
from threading import Thread, local


# custom target function
def task(year):
    u = url.format(year)
    # block for a moment
    df = pd.read_csv(u)
    print(f'finish {year}')
    df.to_pickle(f'yield_curve/daily_{year}.pkl')

 
def download():
    for year in range(2021, 2022):
        t = Thread(target=task, args=(year,)).start()

import numpy as np
import matplotlib.pyplot as plt
import collections

"""
https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202212

https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics

"""

"https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics"
columns = ['1 Mo', '2 Mo', '4 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
columns_m = [1, 2, 3, 4, 6, 12, 24, 36, 60, 72, 120, 240, 360]
map_y = {c: columns_m[i] for i, c in enumerate(columns)}
by_years = {}
if __name__ == "__main__":
    from pathlib import Path
    curve_path = Path('yield_curve').glob('*.pkl')
    for c in curve_path:
        year = c.stem.split('_')[1]
        df = pd.read_pickle(c)
        df = df[::-1]
        df = df.reset_index()
        
        if '30 Yr' not in df.keys():
            df['30 Yr'] = df['20 Yr']
        if '2 Mo' not in df.keys():
            df['2 Mo'] = ((df['1 Mo'] + df['3 Mo']) / 2).round(2)
        if '4 Mo' not in df.keys():
            df['4 Mo'] = ((2*df['3 Mo'] + df['6 Mo']) / 3).round(2)

        df.rename(map_y, axis=1, inplace=True)
        df[columns_m] = df[columns_m].interpolate(method='linear', axis=1)
        df[columns_m] = df[columns_m].interpolate(method='linear', axis=0)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.drop('index', axis=1, inplace=True)

        by_years[year] = df[columns_m]
        print(df.tail())


    od = collections.OrderedDict(sorted(by_years.items()))
    complete_df = pd.DataFrame()

    for k in od.keys():
        complete_df = pd.concat([complete_df, od[k]])
    

    complete_df.index

    from matplotlib import cm

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(complete_df.shape[0])
    Y = np.asarray([int(k) for k in complete_df.keys()])
    X, Y = np.meshgrid(Y, X)


    # Plot the surface.
    surf = ax.plot_surface(Y, X, complete_df, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()

    plt.plot(complete_df)
    plt.show()


    begin_y = 2008
    end_y   = begin_y + 10

    exit()

    from nelson_siegel_svensson.calibrate import calibrate_ns_ols
    for year in range(begin_y, end_y + 1):
        for e in years[str(year)].iterrows():
            terms = np.asarray(list(dict(e[1]).keys())) / 12
            rates = np.asarray(list(e[1]))
            nsv_curve, status = calibrate_ns_ols(terms, rates, tau0=1.0)  # starting value of 1.0 for the optimization of tau
            t = np.linspace(0, 30, 100)
            plt.plot(t, nsv_curve(t))
            plt.show()

nsv_curve, status = calibrate_ns_ols(terms, rates, tau0=1.0)  # starting value of 1.0 for the optimization of tau
t = np.linspace(0, 30, 100)
plt.plot(t, nsv_curve(t))
plt.show()
