"""
https://quantopian-archive.netlify.app/notebooks/notebooks/quantopian_notebook_300.html
https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter/
https://www.quantrocket.com/codeload/quant-finance-lectures/quant_finance_lectures/Lecture45-Kalman-Filters.ipynb.html
https://rdrr.io/rforge/Trading/man/DynamicBeta.html
http://jonathankinlay.com/2018/09/statistical-arbitrage-using-kalman-filter/
https://letianzj.github.io/kalman-filter-pairs-trading.html
https://lost-stats.github.io/Time_Series/Rolling_Regression.html

"""

from portfolios import download
from calcs import returns
from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import poly1d

"""
C = [1, rm - rf]
y = [ri - rf] = C @ [alfa, beta]
X = [alfa, beta]

A = [[1, 0], 
     [0, 1]], 
     
X1 = A @ X0  + wk (0,Qk)

xh10 = A @ xh00
P10 = P00 + Q0
err_1 = ri - rf  - C1 @ xh10

S1 = C1@P10@C1.T
K1 = P10 @ C1.T @ inv(S1)
xh11 = xh10 + K1 @ err_1
P11 = (I - K1@C1) @  P10

"""

def test_kalman():
    symbols = ['BIL', 'KO', 'AAPL', 'SPY', 'AVGO']

    np.random.seed(1)

    df = download(symbols)
    df: pd.DataFrame = returns(symbols, df, log__=True)

    df['SPY'] = np.random.normal(.01, .01, size=len(df))
    df['AAPL'] = 1 * df['SPY']  + np.random.normal(0, .00001, size=len(df))
    df['KO'] = .5 * df['SPY'] 


    A = np.array([[1, 0], 
                 [0.0, 1]])

    x00 = np.array([0.0, 0.0])

    p0 = 1

    P00 = np.eye(2) * p0 # state covariance
    q0 = 1e-3
    Q0 = np.eye(2) * q0 # process noise
    R0 = 1e-8 # observation noise

    print(A)

    beta = []
    for index, row in df.iterrows():
        rm_rf = row['SPY'] 
        ra_rf = row['AAPL'] # meassurement
        C1 = np.array([1, rm_rf])

        x10 = A @ x00
        P10 = A @ P00 @ A.T + Q0 # q0 process noise

        err0 = ra_rf - C1 @ x10

        S0 = C1 @ P10 @ C1.T + R0  # r0 observation noise
        K0 = P10 @ C1.T / S0

        x00 = x10 + K0 * err0
        P00 = (np.eye(2) - K0 @ C1) @ P10
        beta.append(x00[1] )

    plt.plot(np.array(beta))
    plt.show()
 


if __name__ == "__main__":

    symbols = ['BIL', 'AAPL', 'SPY']

    df = download(symbols)

    # df = returns(symbols, df, log__=True)
    
    # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=.01)

    df['sm'], _ = kf.filter(df['AAPL'])

    plt.plot(df.sm)
    plt.plot(df.AAPL)
    plt.show()

    df = returns(symbols, df)

    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[df.SPY], [np.ones(len(df.SPY))]]).T, axis=1)

    print(obs_mat)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
                      initial_state_mean=[0,0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=2,
                      transition_covariance=trans_cov)

    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(df.AAPL)


    _, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(df.AAPL.index, state_means[:,0], label='slope')
    axarr[0].legend()
    axarr[1].plot(df.AAPL.index, state_means[:,1], label='intercept')
    axarr[1].legend()
    plt.tight_layout();
    plt.show()
