
import yfinance as yf
import numpy as np
import talib as ta
from tsfracdiff import FractionalDifferentiator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


ticker = 'SPY'
df_ticker = yf.download(ticker, start="2008-01-01", auto_adjust=True)
df_ticker.dropna(inplace=True)
df_ticker["o"] = df_ticker["Open"]
df_ticker[["h", "l", "c", "v"]] = df_ticker[["High", "Low", "Close", "Volume"]].shift(1)
df_ticker["r"] = np.log(df_ticker["c"]) - np.log(df_ticker["c"].shift(1))
# df_spy['Close'] = np.log(df_spy['Close'])
df_ticker["r2"] = np.power(df_ticker["r"], 2)
df_ticker["r2"] = np.sqrt(df_ticker["r2"].ewm(alpha=(1 - 0.97)).mean())
df_ticker["rsi"] = (
    ta.RSI(df_ticker["c"], timeperiod=14) / ta.RSI(df_ticker["c"], timeperiod=14).mean()
)
df_ticker["MOM"] = ta.MOM(df_ticker["c"], timeperiod=5)
df_ticker["U"] = df_ticker["Close"] > df_ticker["Close"].shift(1)

fracDiff = FractionalDifferentiator()
df_ticker["f"] = fracDiff.FitTransform(df_ticker["c"], parallel=True)
df_ticker.dropna(inplace=True)
df_ticker["ewm_r"] = df_ticker["r"].ewm(alpha=0.5).mean()
df_ticker["ewm_c"] = df_ticker["c"].ewm(alpha=0.5).mean()

limit_t = int(0.75 * df_ticker.shape[0])
df_train = df_ticker[:limit_t]
df_test = df_ticker[limit_t:]

df_ticker['cl'] = df_ticker['c'] > df_ticker['c'].shift(1)
df_ticker['CloseL'] = df_ticker['Close'] > df_ticker['Close'].shift(1)
df_ticker.dropna(inplace=True)

all_features = ["c", "cl", "MOM", 'ewm_c', "f", "r", 'o']
print(df_ticker[['Close', 'CloseL', 'c', 'cl']].tail(8))


limit_t = int(0.75 * df_ticker.shape[0])
df_train = df_ticker[:limit_t]
df_test = df_ticker[limit_t:]
df_test_out = df_test.copy()

classifier_ = MLPClassifier()
classifier = LogisticRegression()
classifier_ = KNeighborsClassifier()

classifier.fit(df_train[all_features], df_train["CloseL"])
df_test_out["predict"] = classifier.predict(df_test[all_features])
print(df_test_out[['CloseL', 'predict']].tail(20))

n_predicted = df_test_out[df_test_out['CloseL'] == df_test_out['predict']]['predict'].shape[0]

efficiency = n_predicted / df_test_out.shape[0]

print(efficiency)