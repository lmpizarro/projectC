from typing import List

import numpy as np

from .pay_off import PayOff, Option, OptionError


class LongStrangle(PayOff):
    def __init__(self, K1, K2, P) -> None:
        if K1 >= K2:
            raise OptionError("K2 mut be GT K1")
        self.K2 = K2
        self.K1 = K1
        self.prices = np.linspace(.5*K1, 1.5*K2, 100)

    def pay_off(self):
        a = np.where(self.prices > self.K2, self.prices - self.K2, 0)
        b = np.where(self.prices < self.K1, -self.prices + self.K1, a)
        return b

class ShortStrangle(LongStrangle):
    def __init__(self, K1, K2, P) -> None:
        super().__init__(K1, K2, P)

    def pay_off(self):
        return - super().pay_off()


