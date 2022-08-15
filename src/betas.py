#!/usr/bin/env python
# -*- coding: utf-8 -*-
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from portfolios import download
from calcs import (returns, beta_by_ewma, cross_matrix)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression



def market_beta(X,Y,N=60):
    """ 
    see https://predictivehacks.com/stocks-market-beta-with-rolling-regression/
    X = The independent variable which is the Market
    Y = The dependent variable which is the Stock
    N = The length of the Window
     
    It returns the alphas and the betas of
    the rolling regression
    """
     
    # all the observations
    obs = len(X)
     
    # initiate the betas with null values
    betas = np.full(obs, np.nan)
     
    # initiate the alphas with null values
    alphas = np.full(obs, np.nan)
     
     
    for i in range((obs-N)):
        regressor = LinearRegression()
        regressor.fit(X.to_numpy()[i : i + N+1].reshape(-1,1), Y.to_numpy()[i : i + N+1])
         
        betas[i+N]  = regressor.coef_[0]
        alphas[i+N]  = regressor.intercept_
 
    return(alphas, betas)


def rolling_beta(df_rets, N=60):
    symbols = list(df_rets.keys())
    symbols.remove('MRKT')
    exog = sm.add_constant(df_rets['MRKT'])


    for s in symbols:
        endog = df_rets[s]
        rols = RollingOLS(endog, exog, window=N)

        rres = rols.fit()

        params = rres.params.copy()

        df_rets[s] = params.MRKT

    return df_rets

def rolling_beta_sk(df_rets, N=60):
    symbols = list(df_rets.keys())
    symbols.remove('MRKT')

    for s in symbols:
        alpha, beta = market_beta(df_rets['MRKT'], df_rets[s], N=N)

        df_rets[s] = beta
    
    return df_rets

def rolling_beta_fussion(df, N=60):
    symbols = list(df.keys())
    df1 = df.copy()

    df_rets = returns(symbols, df, log__=True)

    df_1 = df_rets.copy()

    df_betas2 = rolling_beta_sk(df_rets, N=N)
    df_betas1 = rolling_beta(df_1, N=N)

    df_betas1.drop(columns=['MRKT'], inplace=True)
    df_betas2.drop(columns=['MRKT'], inplace=True)

    print(df_betas1.tail())
    print(df_betas2.tail())

    symbols.append('MRKT')
    df1.rename(columns={'SPY':'MRKT'}, inplace=True)
    df_c = cross_matrix(symbols, df1)
    betas = beta_by_ewma(symbols, df_c)

    betas.drop(columns=['MRKT'], inplace=True)
    print(betas.tail())

    for s in df_betas2.keys():
        df_1[s] = .33*df_betas2[s] + .33*df_betas1[s] + .33*betas[s]

    return df_1


def test_betas():
    symbols = ['TSLA', 'KO', 'AAPL', 'SPY', 'AVGO']

    df = download(symbols)
    df1 = df.copy()
    # df['SPY'] = np.random.normal(.01, .1, size=len(df)) + np.random.normal(.01, .001, size=len(df))
    # df['BIL'] = 5 * df['SPY'] 
    # df['KO'] = .5 * df['SPY'] 

    df.rename(columns={'SPY':'MRKT'}, inplace=True)
    symbols.remove('SPY')
    symbols.append('MRKT')

    df_betas = rolling_beta_fussion(df, N=120)

    symbols.remove('MRKT')    
    plt.plot(df_betas[symbols])
    plt.show()

def test_beta_covar():
    symbols = ['TSLA', 'KO', 'AAPL', 'SPY', 'AVGO']

    
    df = download(symbols)
    df.rename(columns={'SPY':'MRKT'}, inplace=True)
    
    symbols.remove('SPY')

    symbols.append('MRKT')

    df = cross_matrix(symbols, df)
    betas = beta_by_ewma(symbols, df)

    print(betas.tail())


if __name__ == "__main__":
    test_betas()