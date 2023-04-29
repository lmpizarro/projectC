import numpy as np
import matplotlib.pyplot as plt

DAYS_IN_A_YEAR = 365


class RateGenerators:
    def __init__(self, years: np.ndarray, compound: int = 2) -> None:
        self._time = np.linspace(
            1 / DAYS_IN_A_YEAR, years, DAYS_IN_A_YEAR * years, endpoint=False
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

    def sin_rate(self, rate: float = 0.1, amplitude: float = 0.02) -> np.ndarray:
        rates = np.sin(2 * np.pi * self.time / self.time[-1]) * amplitude + rate

        return RateGenerators.discrete_to_continuous(
            rates=rates, compound=self.compound
        )


years = 3
compound = 4
year_rate = 0.1

n_pays = years * compound
coupon = year_rate / compound
times = np.linspace(1 / compound, years, n_pays)
coupons = np.ones(n_pays) * coupon
amortizations = np.zeros(n_pays)
amortizations[n_pays - 1] = 1
description = np.asarray([times, coupons, amortizations, coupons + amortizations])

rate = 0.1

tg = RateGenerators(years=years, compound=compound)

rcs = tg.sin_rate()
rcs_ = tg.exp_rate()
rcs_ = tg.constant_rate()


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
