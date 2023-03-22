from typing import List

import numpy as np

from .pay_off import PayOff, Option
from .put_call import LongCall, LongPut



class LongStraddle(PayOff):
    def __init__(self, options: List[Option]) -> None:
        super().__init__(options)

        self.put_long = LongPut(options=[options[0]])
        self.call_long = LongCall(options=[options[1]])

        self.set_prices()

    def pay_off(self):
        return self.call_long.pay_off() + self.put_long.pay_off()

    def set_prices(self):
        return super().set_prices()

class ShortStraddle(PayOff):
    def __init__(self, options: List[Option]) -> None:
        super().__init__(options)

        self.long_straddle = LongStraddle(options=options)

        self.set_prices()

    def pay_off(self):
        return - self.long_straddle.pay_off()

    def set_prices(self):
        return super().set_prices()

