from downloader import Downloader
import numpy as np
from tsfracdiff import FractionalDifferentiator
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd


tickers = ["SPY", "AAPL"]

fracDiff = FractionalDifferentiator()

dwldr = Downloader(start='2013-01-01', stocks=tickers)

dwldr.download()
dwldr.calc_log_return()

print(dwldr.yf_data_close['SPY'].tail())
print()
print()
print(dwldr.log_returns['SPY'].tail())
ticker = 'SPY'
df_calc = pd.DataFrame()
df_calc['close'] = dwldr.yf_data_close[ticker].copy()[1:]
df_calc['return'] = np.array(dwldr.log_returns[ticker].copy())
df_calc['prev_return'] = df_calc['return'].shift(1)
df_calc['prev_close'] = df_calc['close'].shift(1)
df_calc['ewm_return'] = df_calc['prev_return'].ewm(alpha=.7).mean()
df_calc['ewm_close'] = df_calc['prev_close'].ewm(alpha=.7).mean()
df_calc['frac'] = fracDiff.FitTransform(df_calc['prev_close'], parallel=True)
df_calc.dropna(inplace=True)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

features = ['prev_return', 'prev_close', 'ewm_return', 'ewm_close', 'frac']
df_train = df_calc[:1600]
df_test = df_calc[1600:]

pipe = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])

print(df_calc.tail())
print(df_calc.shape)
pipe.fit(df_train[features], df_train['close'])
sco = pipe.score(df_test[features], df_test['close'])
print(sco)
print(pipe.get_params())
print(pipe.predict(df_test[features])[-7:])
