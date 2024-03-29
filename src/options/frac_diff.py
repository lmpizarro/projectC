from downloader import Downloader
import numpy as np
from tsfracdiff import FractionalDifferentiator
import matplotlib.pyplot as plt
import talib as ta
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    SGDRegressor,
    BayesianRidge,
    LinearRegression,
    HuberRegressor,
)
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import SVR



df_spy = yf.download("SPY", start="2008-01-01", auto_adjust=True)
df_spy.dropna(inplace=True)
df_spy["o"] = df_spy["Open"]
df_spy[["h", "l", "c", "v"]] = df_spy[["High", "Low", "Close", "Volume"]].shift(1)
df_spy["r"] = np.log(df_spy["c"]) - np.log(df_spy["c"].shift(1))
# df_spy['Close'] = np.log(df_spy['Close'])
df_spy["r2"] = np.power(df_spy["r"], 2)
df_spy["r2"] = np.sqrt(df_spy["r2"].ewm(alpha=(1 - 0.97)).mean())
df_spy["rsi"] = (
    ta.RSI(df_spy["c"], timeperiod=14) / ta.RSI(df_spy["c"], timeperiod=14).mean()
)
df_spy["MOM"] = ta.MOM(df_spy["c"], timeperiod=5)
df_spy["U"] = df_spy["Close"] > df_spy["Close"].shift(1)

fracDiff = FractionalDifferentiator()
df_spy["f"] = fracDiff.FitTransform(df_spy["c"], parallel=True)
df_spy.dropna(inplace=True)
df_spy["ewm_r"] = df_spy["r"].ewm(alpha=0.5).mean()
df_spy["ewm_c"] = df_spy["c"].ewm(alpha=0.5).mean()
features = ["c", "rsi", "MOM", "ewm_r", "ewm_c", "f", "r", "o"]
print(df_spy[features].tail(8))

limit_t = int(0.75 * df_spy.shape[0])
df_train = df_spy[:limit_t]
df_test = df_spy[limit_t:]


reg = SVR(C=1.0, epsilon=0.2)
reg = BayesianRidge()
reg = HuberRegressor()
reg = linear_model.RANSACRegressor(random_state=42)
reg_ = linear_model.LassoLars(alpha=0.1)
reg = linear_model.ARDRegression()
reg = linear_model.TheilSenRegressor(random_state=42, max_iter=600)
reg = LinearRegression()

# pipe = Pipeline([('scaler', StandardScaler()), ('reg', BayesianRidge())])
pipe = make_pipeline(StandardScaler(), reg)
pipe.fit(df_train[features], df_train["Close"])
df_test["predict"] = pipe.predict(df_test[features])
df_test["pu"] = pipe.predict(df_test[features])
df_test["pu"] = df_test["predict"] > df_test["predict"].shift(1)
xount = df_test["U"] == df_test["pu"]
print(df_test[["U", "pu", "Close", "predict", "c"]].tail(20))
print(xount.sum() / df_test.shape[0])

sco = pipe.score(df_test[features], df_test["Close"])
mse = mean_squared_error(df_test["Close"], pipe.predict(df_test[features]))
r2s = r2_score(df_test["Close"], pipe.predict(df_test[features]))
mape = mean_absolute_percentage_error(df_test["Close"], pipe.predict(df_test[features]))
print("mse ", mse)
print("r2", r2s)
print("mape", mape)
print("sco", sco)

"""
use linear regression to predict next price
https://www.kaggle.com/code/nikhilkohli/stock-prediction-using-linear-regression-starter

"""
