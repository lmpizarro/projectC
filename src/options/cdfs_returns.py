import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nct
from scipy.stats import norm
from scipy.stats import cauchy
from fitters import *
from scipy.interpolate import splrep, BSpline


import yfinance as yf

stocks = ['AAPL',  'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BRK-B', 'TSLA', 'META', 'JNJ',
          'V', 'TSM', 'XOM', 'UNH', 'WMT', 'JPM', 'MA', 'PG', 'LLY', 'CVX', 'HD',
          'ASML', 'ABBV', 'SPY', 'QQQ', 'DIA']
stocks_ = ['TSM', 'XOM', 'JPM', 'CVX', 'TM', 'PFE', 'BAC']
stocks_= ['SPY', 'QQQ', 'DIA']
stocks_ = ['YPF', 'BMA', 'TX', 'PAM', 'EDN', 'GGAL', 'LOMA']
yf_data = yf.download(stocks, group_by=stocks, start='2012-01-01')

print(yf_data.head())

def denoise_spl(x, y, s):
    tck_s = splrep(x, y, s=s)
    return tck_s

for stock in stocks:
    pass_tuple = (stock, 'Adj Close')
    yf_data[(stock, 'returns')] = yf_data[pass_tuple].pct_change()
    yf_data[(stock, 'log_returns')] = np.log(yf_data[pass_tuple]/yf_data[pass_tuple].shift(1))
    yf_data.dropna(inplace=True)

    f_data = yf_data[stock]

    loc, scale = fit_norm(f_data)

    bins, cdf, count = create_histogram(f_data)
    spl_cdf = spline_cdf(bins, cdf)

    dYdx = -(spl_cdf(-0.005)-spl_cdf(0.005))/0.01

    print(stock, dYdx, loc, scale)


    plt.plot(bins, count, label=f'CDF {stock}')
    plt.legend()

plt.show()
exit()

def view_returns(yf_data, key='returns'):
    loc1, scale1 = norm.fit(data=yf_data[key])

    nct_fit_params = nct.fit(yf_data[key])
    cau_fit_params = cauchy.fit(yf_data[key])
    x = np.linspace(yf_data[key].min(), yf_data[key].max(), 250)
    y = nct.pdf(x, nct_fit_params[0], nct_fit_params[1], nct_fit_params[2], nct_fit_params[3])
    ynorm = norm.pdf(x, loc1, scale1)
    ycauchy = cauchy.pdf(x, cau_fit_params[0], cau_fit_params[1])
    from scipy import stats


    plt.plot(x, ycauchy)
    plt.plot(x, ynorm)
    plt.plot(x, y, label='t')
    plt.legend()
    rt = nct.rvs(nct_fit_params[0], nct_fit_params[1], nct_fit_params[2], nct_fit_params[3], size=yf_data.shape[0])
    rn = norm.rvs(loc1, scale1, size=1000)
    # count, bins, ignored = plt.hist(r, 100, density=True)
    count1, bins1, ignored1 = plt.hist(yf_data[key], 250, density=True)
    print(stats.ks_1samp(yf_data[key], stats.norm(loc1, scale1).cdf))
    print(stats.ks_1samp(yf_data[key], stats.nct(nct_fit_params[0], nct_fit_params[1], nct_fit_params[2], nct_fit_params[3]).cdf))
    print(stats.ks_1samp(yf_data[key], stats.cauchy(cau_fit_params[0], cau_fit_params[1]).cdf))
    # print(r)

    plt.show()
    # plt.scatter(rt, yf_data[key])
    # sm.qqplot(yf_data[key])
    count1, bins1, ignored1 = plt.hist(yf_data[key], 250, density=True, cumulative=True)
    # plt.plot(x, nct.cdf(x, nct_fit_params[0], nct_fit_params[1], nct_fit_params[2], nct_fit_params[3]))
    # plt.plot(x, norm.cdf(x, loc1, scale1), label='norm')
    # plt.plot(x, cauchy.cdf(x, cau_fit_params[0], cau_fit_params[1]), label='cauchy')
    plt.plot(bins1[:-1], count1)
    plt.legend()
    plt.show()
    print(nct_fit_params[2], nct_fit_params[3])
    print(loc1, scale1)

# view_returns(yf_data=yf_data, key='log_return')
# view_returns(yf_data=yf_data, key='returns')
"""
https://en.wikipedia.org/wiki/Quantile_function
https://greenteapress.com/thinkstats/
https://towardsdatascience.com/quantiles-key-to-probability-distributions-ce1786d479a9
https://www.statology.org/q-q-plot-python/
https://en.wikipedia.org/wiki/Quantile_function

"""


rng = np.random.default_rng(2022)

numbers1 = np.random.normal(0, .25, 10000)
numbers2 = np.random.normal(0, .9, 10000)
numbers = np.concatenate((numbers2, numbers1))
numbers = numbers1 * numbers2



count, bins, ignored = plt.hist(numbers, 100, density=True)
plt.show()




a = rng.beta(5, 5, 100)
print(a.mean(), a.var(), a.min(), a.max())

from scipy.stats import beta

a, b = 80, 10
#
# Generate the value between
#
x = np.linspace(beta.ppf(0.001, a, b),beta.ppf(0.999, a, b), 100)


plt.figure(figsize=(7,7))
plt.xlim(0.7, 1)
plt.plot(x, beta.pdf(x, a, b), 'r-')
plt.title('Beta Distribution', fontsize='15')
plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
plt.ylabel('Probability', fontsize='15')
plt.show()

# https://vitalflux.com/beta-distribution-explained-with-python-examples/
plt.plot(x, beta.cdf(x, a, b))
plt.show()

def getAlphaBeta(mu, sigma):
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)

    beta = alpha * (1 / mu - 1)

    return alpha, beta

s = np.random.beta(a,b, 100000)
# s = np.random.normal(0, 1, 10000)
count, bins, ignored = plt.hist(s, 50, density=True)
plt.show()

cdf = count.cumsum()
cdf = cdf / cdf[-1]

z = np.polyfit(bins[1:], cdf, deg=10)
poly = np.poly1d(z)
plt.plot(bins[1:], cdf)
plt.plot(bins[1:], poly(bins[1:]))
plt.show()
print(cdf.shape, bins.min(), bins.max())


spl = interpolate.InterpolatedUnivariateSpline(bins[1:], cdf, k=5)
xnew = bins[1:]
plt.plot(xnew, spl(xnew), label='spline')
plt.legend()
plt.show()
"""
http://pytolearn.csd.auth.gr/d1-hyptest/11/distros.html
"""

print(xnew[find_nearest(spl(xnew), 1)])

exit()

print(bins.shape[0])

print(s.mean(), s.var(), s.std(), skew(s), kurtosis(s))

print(beta.ppf(0.5, a, b))

print(getAlphaBeta(s.mean(), s.std()))


"""
exponential generalized beta distribution of the second kind
https://en.wikipedia.org/wiki/Generalized_beta_distribution
"""