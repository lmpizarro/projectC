import numpy as np
import matplotlib.pyplot as plt

DAYS_IN_A_YEAR = 365

class RateGenerators:

    def __init__(self, years: np.ndarray) -> None:
        self._time = np.linspace(1/DAYS_IN_A_YEAR, years, DAYS_IN_A_YEAR*years, endpoint=False )

    @property
    def time(self) -> np.ndarray:
        return self._time

    def constant_rate(self, rate: float = .1) -> np.ndarray:

        rates = np.ones(self.time.shape[0])*rate
        return RateGenerators.discrete_to_continuous(rates=rates)

    def exp_rate(self, rate: float=.1, speed: float = -4) -> np.ndarray:
        rates = np.ones(self.time.shape[0])*rate
        rates = np.exp(speed * self.time) * rates
        if speed >= 0:
            rates = (1 - np.exp(-speed * self.time)) * rates

        return RateGenerators.discrete_to_continuous(rates=rates)

    @staticmethod
    def discrete_to_continuous(rates: np.ndarray) -> np.ndarray:
        return compound * np.log(1 +  rates / compound)

    def sin_rate(self, rate: float=.1, amplitude: float = .02) -> np.ndarray:
        rates = np.sin(2*np.pi * self.time / self.time[-1])*amplitude + rate

        return RateGenerators.discrete_to_continuous(rates=rates)


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

tg = RateGenerators(years=years)

rcs = tg.sin_rate()
rcs_ = tg.exp_rate()
npvs = np.zeros(tg.time.shape)
for i in range(tg.time.shape[0]):
    remaining = description[0] - tg.time[i]
    mask = np.where( remaining <= 0, 0, 1)

    description = mask * description
    pays = description[1] + description[2]
    npv = np.exp(-remaining*rcs[i]) * pays
    npvs[i] = npv.sum()

plt.plot(npvs)
plt.show()
plt.plot(rcs)
plt.show()



