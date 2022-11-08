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

columns = ['1 Mo', '2 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
if __name__ == "__main__":
    from pathlib import Path
    curve_path = Path('yield_curve').glob('*.pkl')
    for c in curve_path:
        df = pd.read_pickle(c)
        if '30 Yr' not in df.keys():
            df['30 Yr'] = df['20 Yr']
        if '2 Mo' not in df.keys():
            df['2 Mo'] = df['1 Mo'] + df['3 Mo']

        if df['2 Mo'].isnull().sum():
            print('dddd')

        print(df[columns].tail())
