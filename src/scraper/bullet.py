import numpy as np

years = 1
compound = 4
year_rate = .1

n_pays = years * compound
coupon = year_rate / compound
times = np.linspace(1/compound,years,n_pays)
coupons = np.ones(n_pays)*coupon
amortizations = np.zeros(n_pays)
amortizations[n_pays - 1] = 1
description = np.asarray([times, coupons, amortizations])

rate = .1

def constant_rate(time, rate: float = .1):

    rates = np.ones(time.shape[0])*rate
    rcs = compound * np.log(1 +  rates / compound)
    return rcs

def exp_rate(time, rate: float=.1, speed: float = -4):
    rates = np.ones(time.shape[0])*rate
    rates = np.exp(speed * time) * rates
    if speed >= 0:
        rates = (1 - np.exp(-speed * time)) * rates

    rcs = compound * np.log(1 +  rates / compound)
    return rcs

import matplotlib.pyplot as plt

def sin_rate(time, rate: float=.1, amplitude: float = .02):
    rates = np.sin(2*np.pi * time / time[-1])*amplitude + rate

    rcs = compound * np.log(1 +  rates / compound)
    return rcs


time = np.linspace(1/365, years, 365*years, endpoint=False )
rcs = sin_rate(time=time)
rcs_ = exp_rate(time=time)
npvs = np.zeros(time.shape)
for i in range(time.shape[0]):
    remaining = description[0] - time[i]
    mask = np.where( remaining <= 0, 0, 1)

    description = mask * description
    pays = description[1] + description[2]
    npv = np.exp(-remaining*rcs[i]) * pays
    npvs[i] = npv.sum()

plt.plot(npvs)
plt.show()
plt.plot(rcs)
plt.show()


