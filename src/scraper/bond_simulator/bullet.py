import numpy as np
import matplotlib.pyplot as plt
from sim_bonds import npv_time, price_to_yield
from sim_bonds import Ba37D

DAYS_IN_A_YEAR = 360


class RateGenerators:
    def __init__(self, years: int, compound: int = 2, endpoint=False) -> None:
        self._time = np.linspace(
            1 / DAYS_IN_A_YEAR, years, int(np.ceil(DAYS_IN_A_YEAR * years)), endpoint=endpoint
        )
        self._compound = compound

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def compound(self) -> int:
        return self._compound

    def constant_rate(self, rate: float = 0.1) -> np.ndarray:
        rates = np.ones(self.time.shape[0]) * rate
        rates = RateGenerators.discrete_to_continuous(
            rates=rates, compound=self.compound
        )
        return rates

    def exp_rate(self, rate: float = 0.1, speed: float = -4) -> np.ndarray:
        rates = np.ones(self.time.shape[0]) * rate
        rates = np.exp(speed * self.time) * rates
        if speed >= 0:
            rates = (1 - np.exp(-speed * self.time)) * rates

        return RateGenerators.discrete_to_continuous(
            rates=rates, compound=self.compound
        )

    @staticmethod
    def discrete_to_continuous(rates: np.ndarray, compound: int) -> np.ndarray:
        return compound * np.log(1 + rates / compound)

    @staticmethod
    def continuous_to_discrete(rates: np.ndarray, compound: int) -> np.ndarray:
        return compound * (np.exp(rates / compound) - 1)

    def sin_rate(self, rate: float = 0.1, amplitude: float = 0.02, cycles:float = 1) -> np.ndarray:
        rates = np.sin(2 * np.pi * self.time * cycles / self.time[-1]) * amplitude + rate

        return RateGenerators.discrete_to_continuous(
            rates=rates, compound=self.compound
        )

class Bullet:
    def __init__(self, maturity:int = 2, compound:int=2, nominal_rate:float=.1) -> None:

        self.maturity = maturity
        self.compound = compound
        self.nominal_rate = nominal_rate
        self.n_payments = maturity * compound
        self.coupon_rate = nominal_rate / compound

        times = np.linspace(1 / compound, self.maturity, self.n_payments)
        coupons = np.ones(self.n_payments) * self.coupon_rate
        amortizations = np.zeros(self.n_payments)
        amortizations[self.n_payments - 1] = 1
        self.np_description = np.asarray([times, coupons, amortizations, coupons + amortizations])



bond = Bullet()
bond = Ba37D()
description = bond.np_description
years = bond.maturity
compound = bond.compound

tg = RateGenerators(years=years, compound=compound)

today_price = 23.94
x_rate = np.linspace(0, 1, 1000)
y_price = np.zeros(1000)
for i, r in enumerate(x_rate):
    rcs = tg.constant_rate(rate=r)
    p = npv_time(description=description, time=tg.time[0:1], rates=rcs[0:1], indx=0)

    y_price[i] = p
    x_rate[i] = tg.continuous_to_discrete(rcs[0], compound=bond.compound)


today_rate = x_rate[np.abs(y_price - today_price).argmin()]
rcs = tg.constant_rate(rate=today_rate)

print(today_rate, today_price)

npvs = np.asarray(
    [
        npv_time(description=description, time=tg.time, rates=rcs, indx=i)
        for i in range(tg.time.shape[0])
    ]
)

plt.plot(npvs)
plt.show()
plt.plot(RateGenerators.continuous_to_discrete(rcs, compound=compound))
plt.show()

rcs = tg.constant_rate(rate=today_rate)
pty = price_to_yield(description=description, time=tg.time[0:1], rates=rcs[0:1], indx=0)

D = pty/today_price

MD = D /(1+ today_rate/2)

print(MD)

