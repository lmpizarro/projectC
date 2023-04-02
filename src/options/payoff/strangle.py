from typing import List

import numpy as np

from .pay_off import PayOff, Option, OptionError


class LongStrangle(PayOff):
    def __init__(self, options: List[Option]) -> None:
        super().__init__(options)

        if self.options[0].K >= self.options[1].K:
            raise OptionError("K2 mut be GT K1")
        
        self.K2 = options[1].K
        self.K1 = options[0].K

        self.set_prices()

    def set_prices(self):
        self.prices = np.linspace(.5*self.K1, 1.5*self.K2, 100)

    def pay_off(self):
        a = np.where(self.prices > self.K2, self.prices - self.K2, 0)
        b = np.where(self.prices < self.K1, -self.prices + self.K1, a)
        return b

    def set_prices(self):
        return super().set_prices()

class ShortStrangle(LongStrangle):
    def __init__(self, options: List[Option]) -> None:
        super().__init__(options)

    def pay_off(self):
        return - super().pay_off()



