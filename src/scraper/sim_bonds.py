import numpy as np
from datetime import date, timedelta, datetime, time
from dateutil import tz
import numpy_financial as npf

import pandas as pd

DAYS_IN_YEAR = 365


class BondSimulator:
    def __init__(
        self,
        description: pd.DataFrame,
        ref_date: date = datetime.now().date(),
        value=20,
    ) -> None:
        self.description = description
        self.ref_date = ref_date
        self.value = value
        self.max_iter: int = 0

        self.update(ref_date=ref_date)
        self.create()
        self.delta_ts(self.ref_date)


    def update(self, ref_date: date):
        self.current_bond = self.description[
            self.description.fecha >= ref_date
        ]

    def create(self):
        self.amortizations = self.current_bond["amort"].to_numpy()
        self.coupons = self.current_bond["interes"].to_numpy()
        self.pays = self.coupons + self.amortizations
        self.dates = list(self.current_bond["fecha"])

    def delta_ts(self, ref_date: date):
        self.time_to_pay = np.array(
            [(e - ref_date).days / DAYS_IN_YEAR for e in self.dates if e >= ref_date]
        )
        self.max_iter = int(self.time_to_pay[-1] * DAYS_IN_YEAR + 1)

    def increment(self, incr: int):
        ref_date = self.ref_date + timedelta(days=incr)
        return ref_date

    def process(self, incr: int, yeld: float, pays_per_year: int = 2):
        ref_date = self.increment(incr=incr)
        self.update(self.increment(incr=incr))
        self.create()
        self.delta_ts(ref_date)
        Rc = pays_per_year * np.log(1+yeld/pays_per_year)
        return (np.exp(-Rc * self.time_to_pay) * self.pays).sum()
        # return (np.power(1 + yeld/pays_per_year, -pays_per_year*self.time_to_pay) * self.pays).sum()


class Zero:
    def __init__(
        self,
        begin_date="2023-05-01",
        years_to_end=3,
        value=0.7,
        face_value=1,
    ) -> None:
        self.tir = 0
        self.value = value
        self.years_to_end = years_to_end
        self.face_value = face_value

        begin_date = date.fromisoformat(begin_date)
        self.begin_date = datetime.combine(
            begin_date, time=time(0, 0, 0), tzinfo=tz.UTC
        )
        self.delta_time_to_end = timedelta(
            seconds=years_to_end * DAYS_IN_YEAR * 24 * 3600
        )
        self.create()

    def create(self):
        self.dates = [self.begin_date, self.begin_date + self.delta_time_to_end]
        self.amortizations = [self.value, self.face_value]
        self.coupons = [0, 0]
        self.tir = (self.face_value / self.value) ** (1 / self.years_to_end) - 1


class Bullet:
    def __init__(
        self,
        emission_date="2023-04-27",
        years_to_end=3,
        compounding=2,
        nominal_yield=0.1,
        face_value=100,
        value=90,
    ) -> None:
        self.periodicity = compounding
        self.nominal_yield = nominal_yield
        self.face_value = face_value
        self.value = value

        emission_date = date.fromisoformat(emission_date)
        self.begin_date = datetime.combine(
            emission_date, time=time(0, 0, 0), tzinfo=tz.UTC
        )
        self.delta_time_pays = timedelta(days=DAYS_IN_YEAR/self.periodicity)
        self.delta_time_to_end = timedelta(days=years_to_end*DAYS_IN_YEAR)
        self.create()

    def create(self):
        i = 0
        self.dates = [self.begin_date]
        self.coupons = [0]
        self.amortizations = [-self.value]
        while self.dates[i] < self.begin_date + self.delta_time_to_end:
            self.dates.append(self.dates[i] + self.delta_time_pays)
            self.coupons.append(self.face_value * self.nominal_yield / self.periodicity)
            self.amortizations.append(0)
            i += 1
        self.amortizations[-1] = self.face_value
        self.dates = [e.date() for e in self.dates]
        self.tir = npf.rate(
            nper=len(self.dates) - 1,
            pmt=self.nominal_yield,
            pv=-self.value,
            fv=self.face_value,
            guess=0.5,
        )

        self._description = pd.DataFrame(
            {"fecha": self.dates, "interes": self.coupons, "amort": self.amortizations}
        )[1:]

    @property
    def description(self):
        return self._description


"""
https://pypi.org/project/bond-pricing/
https://numpy.org/numpy-financial/latest/

"""
bullet = Bullet(face_value=1, value=0.3)
for i, e in enumerate(bullet.dates):
    print(e, bullet.coupons[i], bullet.amortizations[i])

print(bullet.tir)

zc = Zero(years_to_end=1, value=0.8)

for i, e in enumerate(zc.dates):
    print(e, zc.coupons[i], zc.amortizations[i])

print(zc.tir)

import pandas as pd


class Ba37D:
    def __init__(
        self,
        csv_name: str = "ba37d.csv",
        ref_date: date = datetime.now().date(),
        value=20,
    ) -> None:
        self.ref_date = ref_date
        self._description = pd.read_csv(csv_name, sep=",")[
            ["fecha", "interes", "amort"]
        ]
        self._description["fecha"] = pd.to_datetime(
            self._description["fecha"], format="%d/%m/%Y"
        ).dt.date
        self._description['interes'] = self._description['interes'] / 2

        self._description['times'] = (self._description['fecha'] - self.ref_date).dt.days / DAYS_IN_YEAR
        self._description  = self._description[self._description['times'] > 0]

        
    @property
    def description(self):
        return self._description

ba37 = Ba37D()
print(ba37.description)
exit()

bond = BondSimulator(bullet.description)

prices1 = [bond.process(i, yeld=0.1) for i in range(bond.max_iter)]

prices2 = [bond.process(i, yeld=0.3321) for i in range(bond.max_iter)]
prices3 = [bond.process(i, yeld=0.3321+.01) for i in range(bond.max_iter)]

import matplotlib.pyplot as plt
plt.plot(prices1)
plt.show()
rates = np.linspace(.01, 1.01, 100)

prices3 = [bond.process(180, yeld=e) for e in rates]
prices2 = [bond.process(360, yeld=e) for e in rates]
prices1 = [bond.process(300, yeld=e) for e in rates]


plt.plot(rates, prices1)
plt.plot(rates, prices2)
plt.plot(rates, prices3)
plt.show()