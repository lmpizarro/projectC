import numpy as np
import matplotlib.pyplot as plt

DAYS_IN_A_YEAR = 365


class RateGenerators:
    def __init__(self, years: int, compound: int = 2, endpoint=False) -> None:
        self._time = np.linspace(
            1 / DAYS_IN_A_YEAR, years, int(DAYS_IN_A_YEAR * years), endpoint=endpoint
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
        return RateGenerators.discrete_to_continuous(
            rates=rates, compound=self.compound
        )

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


years = 2
compound = 4
year_rate = 0.05

n_payments = years * compound
coupon_rate = year_rate / compound

times = np.linspace(1 / compound, years, n_payments)
coupons = np.ones(n_payments) * coupon_rate
amortizations = np.zeros(n_payments)
amortizations[n_payments - 1] = 1
description = np.asarray([times, coupons, amortizations, coupons + amortizations])
from sim_bonds import Ba37D
bond = Ba37D()
description = bond.np_description
years = bond.maturity
compound = bond.compound
tg = RateGenerators(years=years, compound=compound)

rcs = tg.sin_rate(cycles=2, amplitude=0.01)
rcs_ = tg.exp_rate()
rcs = tg.constant_rate()


def npv_time(description: np.ndarray, time: np.ndarray, rates: np.ndarray, indx: int) -> float:
    remaining = description[0] - time[indx]
    description = np.where(remaining <= 0, 0, 1) * description
    npv = np.exp(-remaining * rates[indx]) * description[3]
    return npv.sum()


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
