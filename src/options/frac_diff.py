from downloader import Downloader
import numpy as np
from tsfracdiff import FractionalDifferentiator
import matplotlib.pyplot as plt
import talib as ta
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import SGDRegressor, BayesianRidge, LinearRegression, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

fracDiff = FractionalDifferentiator()
df_spy = yf.download('SPY', start='2008-01-01', auto_adjust=True)
df_spy.dropna(inplace=True)
df_spy['o'] = df_spy['Open']
df_spy[['h', 'l', 'c', 'v']] = df_spy[['High', 'Low', 'Close', 'Volume']].shift(1)
df_spy['r'] = np.log(df_spy['c']) - np.log(df_spy['c'].shift(1))
df_spy['r2'] = np.power(df_spy['r'], 2)
df_spy['r2'] = np.sqrt(df_spy['r2'].ewm(alpha=(1-0.97)).mean())
df_spy['rsi'] = ta.RSI(df_spy['c'], timeperiod=14) / ta.RSI(df_spy['c'], timeperiod=14).mean()
df_spy['MOM'] =  ta.MOM(df_spy['c'], timeperiod=5)
df_spy['U'] = df_spy['Close'] > df_spy['Close'].shift(1)

df_spy['f'] = fracDiff.FitTransform(df_spy['c'], parallel=True)
df_spy.dropna(inplace=True)
df_spy['ewm_r'] = df_spy['r'].ewm(alpha=.5).mean()
df_spy['ewm_c'] = df_spy['c'].ewm(alpha=.5).mean()
features = ['c','rsi', 'MOM', 'ewm_r', 'ewm_c', 'f', 'r', 'o']
print(df_spy[features].tail(8))

limit_t = int(.75 * df_spy.shape[0])
df_train = df_spy[:limit_t]
df_test = df_spy[limit_t:]

from sklearn import linear_model
reg = BayesianRidge()
reg = LinearRegression()
reg = HuberRegressor()
reg = linear_model.RANSACRegressor(random_state=42)
reg_ = linear_model.LassoLars(alpha=.1)
reg = linear_model.ARDRegression()
reg = linear_model.TheilSenRegressor(random_state=42, max_iter=600)


# pipe = Pipeline([('scaler', StandardScaler()), ('reg', BayesianRidge())])
pipe = make_pipeline(StandardScaler(),  reg)
pipe.fit(df_train[features], df_train['Close'])
sco = pipe.score(df_test[features], df_test['Close'])
print(sco)
df_test['predict'] = pipe.predict(df_test[features])
df_test['pu'] = pipe.predict(df_test[features])
df_test['pu'] = df_test['predict'] > df_test['predict'].shift(1)
xount = df_test['U'] == df_test['pu']
print(df_test[['U','pu', 'Close', 'predict', 'c']].tail(20))
print(xount.sum()/df_test.shape[0])
mse = mean_squared_error(pipe.predict(df_test[features]), df_test['Close'])
evs = explained_variance_score(pipe.predict(df_test[features]), df_test['Close'])
print('mse ', mse)
print('evs', evs)
exit()



tickers = ["SPY", "AAPL"]


dwldr = Downloader(start='2013-01-01', stocks=tickers)
dwldr.download()
dwldr.calc_log_return()

print(dwldr.yf_data_close['SPY'].tail())
print(dwldr.log_returns['SPY'].tail())

ticker = 'AAPL'
df_calc = pd.DataFrame()
df_calc['close'] = dwldr.yf_data_close[ticker].copy()[1:]
df_calc['return'] = np.array(dwldr.log_returns[ticker].copy())
df_calc['prev_return'] = df_calc['return'].shift(1)
df_calc['prev_close'] = df_calc['close'].shift(1)
df_calc['ewm_return'] = df_calc['prev_return'].ewm(alpha=.5).mean()
df_calc['ewm_close'] = df_calc['prev_close'].ewm(alpha=.5).mean()
df_calc['rsi_close'] = ta.RSI(df_calc['prev_close'], timeperiod=14) / ta.RSI(df_calc['prev_close'], timeperiod=14).mean()
df_calc['frac'] = fracDiff.FitTransform(df_calc['prev_close'], parallel=True)
df_calc['MOM'] =  ta.MOM(df_calc['prev_close'], timeperiod=5)
df_calc.dropna(inplace=True)

features = ['prev_return', 'prev_close', 'ewm_return', 'ewm_close', 'frac', 'rsi_close', 'MOM']
df_train = df_calc[:1600]
df_test = df_calc[1600:]

pipe = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])

print(df_calc.tail(8))
print(df_calc.shape)
pipe.fit(df_train[features], df_train['close'])
sco = pipe.score(df_test[features], df_test['close'])
print(sco)
print(pipe.get_params())
print(pipe.predict(df_test[features])[-8:])

"""
use linear regression to predict next price
https://www.kaggle.com/code/nikhilkohli/stock-prediction-using-linear-regression-starter

"""
