from typing import List

import numpy as np

from .pay_off import PayOff, Option



class BullCallSpread(PayOff):

    def __init__(self, options: List[Option]) -> None:
        if len(options) != 2:
            raise ValueError

        super().__init__(options)


        if self.options[1].K <= self.options[0].K or \
            self.options[1].P >= self.options[0].P:
            raise OptionError

        self.set_prices()

    def set_prices(self):
        self.prices = np.linspace(0, 1.5*self.options[1].K)

    def pay_off(self):

        po_long_c = np.where(self.prices> self.options[0].K,
                            self.prices - self.options[0].K, 0)

        po_short_c = np.where(self.prices > self.options[1].K,
                            self.options[1].K - self.prices, 0)

        po = po_long_c + po_short_c - self.options[0].P + self.options[1].P

        return po


