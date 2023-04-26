import numpy as np
from datetime import date, timedelta, datetime, time
from dateutil import tz
import numpy_financial as npf

DAYS_IN_YEAR = 360


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
        begin_date="2023-05-01",
        years_to_end=3,
        periodicity=0.5,
        nominal_yield=0.1,
        face_value=100,
        value=90,
    ) -> None:
        self.periodicity = periodicity
        self.nominal_yield = nominal_yield
        self.face_value = face_value
        self.value = value

        begin_date = date.fromisoformat(begin_date)
        self.begin_date = datetime.combine(
            begin_date, time=time(0, 0, 0), tzinfo=tz.UTC
        )
        self.delta_time_pays = timedelta(seconds=periodicity * DAYS_IN_YEAR * 24 * 3600)
        self.delta_time_to_end = timedelta(
            seconds=years_to_end * DAYS_IN_YEAR * 24 * 3600
        )
        self.create()

    def create(self):
        i = 0
        self.dates = [self.begin_date]
        self.coupons = [0]
        self.amortizations = [-self.value]
        while self.dates[i] < self.begin_date + self.delta_time_to_end:
            self.dates.append(self.dates[i] + self.delta_time_pays)
            self.coupons.append(self.face_value * self.nominal_yield * self.periodicity)
            self.amortizations.append(0)
            i += 1
        self.amortizations[-1] = self.face_value
        self.dates = [e.date() for e in self.dates]
        self.tir = npf.rate(
            nper=len(self.dates) - 1,
            pmt=self.nominal_yield,
            pv=-self.value,
            fv=self.face_value,
            guess=.5
        )

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
