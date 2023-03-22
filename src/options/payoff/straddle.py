from typing import List

import numpy as np

from .pay_off import PayOff, Option



class LongStraddle(PayOff):
    def __init__(self, K: float, P: float) -> None:
        super().__init__(K, P)
        self.call_long = LongCall(K, P)
        self.put_long = LongPut(K, P)

    def pay_off(self):
        return self.call_long.pay_off() + self.put_long.pay_off()

class ShortStraddle(PayOff):
    def __init__(self, K: float, P: float) -> None:
        super().__init__(K, P)
        self.long_straddle = LongStraddle(K, P)

    def pay_off(self):
        return - self.long_straddle.pay_off()

