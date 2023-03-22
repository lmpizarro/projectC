from typing import List

import numpy as np

from .pay_off import PayOff, Option

class LongCall(PayOff):
    def __init__(self, options: List[Option]) -> None:
        if options[0].type != 'C':
            raise ValueError

        super().__init__(options)
        self.set_prices()

    def pay_off(self):
        po = np.where(self.prices < self.options[0].K,
                      0,
                      self.prices - self.options[0].K)
        return po

    def set_prices(self):
        return super().set_prices()

class ShortCall(PayOff):
    def __init__(self, options: List[Option]) -> None:
        super().__init__(options)

        self.lc = LongCall(options)

        self.set_prices()

    def pay_off(self):
        return - self.lc.pay_off()

    def set_prices(self):
        return super().set_prices()

class LongPut(PayOff):
    def __init__(self, options: List[Option]) -> None:
        if options[0].type != 'P':
            raise ValueError

        super().__init__(options)
        self.set_prices()

    def pay_off(self):
        po = np.where(self.prices < self.options[0].K, - self.prices + self.options[0].K, 0)
        return po

    def set_prices(self):
        return super().set_prices()

class ShortPut(PayOff):

    def __init__(self, options: List[Option]) -> None:
        super().__init__(options)

        self.set_prices()

    def pay_off(self):
        po = np.where(self.prices < self.options[0].K, - self.prices + self.options[0].K, 0)
        return - po

    def set_prices(self):
        return super().set_prices()

