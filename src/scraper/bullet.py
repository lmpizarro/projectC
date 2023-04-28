import numpy as np

years = 3
compound = 2
rate = .1

n_pays = years * compound
coupon = rate / compound
times = np.linspace(1/compound,years,n_pays)
coupons = np.ones(n_pays)*coupon
amortizations = np.zeros(n_pays)
amortizations[n_pays - 1] = 1
description = np.asarray([times, coupons, amortizations])

r = 0.1
rc = compound * np.log(1 + r /compound)


time = np.linspace(1/365, years, 365*years, endpoint=False )
npvs = np.zeros(time.shape)
for i,t in enumerate(time):
    remaining = description[0] - t
    mask = np.where( remaining <= 0, 0, 1)

    description = mask * description
    pays = description[1] + description[2]
    npv = np.exp(-remaining*rc) * pays
    npvs[i] = npv.sum()

import matplotlib.pyplot as plt
plt.plot(npvs)
plt.show()


