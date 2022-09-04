#!/usr/bin/env python
# -*- coding: utf-8 -*-
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from calcs import (returns, cross_matrix)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor

"""
https://machinelearningmastery.com/robust-regression-for-machine-learning-in-python/
https://www.statsmodels.org/dev/examples/notebooks/generated/robust_models_1.html
"""

def beta_by_ewma(symbols, df_cov):
    beta = pd.DataFrame()
    symbols.remove('MRKT')
    for s in symbols:
        k_cov_mrkt = f'{s}_MRKT'
        beta[s] = df_cov[k_cov_mrkt] / df_cov['MRKT_ewma']

    return beta

def market_beta(X,Y,N=60, regressor='Linear'):
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
        if regressor=='Huber':
            regressor = HuberRegressor()
        if regressor=='RANSAC':
            regressor = RANSACRegressor()
        if regressor=='TheilSen':
            regressor=TheilSenRegressor()

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



from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import absolute
from matplotlib import pyplot
from numpy import arange

# evaluate a model
def evaluate_model(X, y, model):
	# define model evaluation method
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	# force scores to be positive
	return absolute(scores)

 # plot the dataset and the model's line of best fit
def plot_best_fit(X, y, model):
	# fut the model on all data
	model.fit(X, y)
	# plot the dataset
	pyplot.scatter(X, y)
	# plot the line of best fit
	xaxis = arange(X.min(), X.max(), 0.01)
	yaxis = model.predict(xaxis.reshape((len(xaxis), 1)))
	pyplot.plot(xaxis, yaxis, color='r')
	# show the plot
	pyplot.title(type(model).__name__)
	pyplot.show()



