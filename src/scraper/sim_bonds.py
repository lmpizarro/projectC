import numpy as np
from datetime import date, timedelta, datetime, time
from dateutil import tz
import numpy_financial as npf
import pandas as pd


import matplotlib.pyplot as plt

import pandas as pd

DAYS_IN_YEAR = 360

def npv_time(description: np.ndarray, time: np.ndarray, rates: np.ndarray, indx: int) -> float:
    remaining = description[0] - time[indx]
    description = np.where(remaining <= 0, 0, 1) * description
    npv = np.exp(-remaining * rates[indx]) * description[3]
    return npv.sum()



"""
https://pypi.org/project/bond-pricing/
https://numpy.org/numpy-financial/latest/

"""


class Ba37D:
    def __init__(
        self,
        csv_name: str = "ae38d.csv",
        ref_date: date = datetime.now().date(),
        value=20,
    ) -> None:
        self.ref_date = ref_date
        self.compound = 1
        self._description = pd.read_csv(csv_name, sep=",")[
            ["fecha", "interes", "amort"]
        ]
        self._description["fecha"] = pd.to_datetime(
            self._description["fecha"], format="%d/%m/%Y"
        ).dt.date

        self._description["times"] = (
            self._description["fecha"] - self.ref_date
        ).dt.days / DAYS_IN_YEAR
        self._description = self._description[self._description["times"] > 0]

        self._np_description = np.asarray([
            self._description.times, self._description.interes, self._description.amort,
            self._description.interes + self._description.amort
        ])
        self.maturity = self._description.times.iloc[-1]

    @property
    def description(self):
        return self._description

    @property
    def np_description(self):
        return self._np_description

