import numpy as np
import matplotlib.pyplot as plt
from typing import List


from abc import ABC, abstractmethod

class OptionError(Exception):
    pass

class Option:
    def __init__(self, K: float, T: float, S: float, q: float,
                 r: float, type: str, price:float=0.0) -> None:
        self.K = K
        self.T = T
        self.S = S
        self.q = q
        self.r = r
        self.P = price


        if type not in ['P', 'C']:
            raise ValueError

        self.type = type

    def set_price(self, price: float) -> None:
        self.P = price


class PayOff(ABC):
    def __init__(self, options: List[Option]) -> None:
        self.options = options

    @abstractmethod
    def pay_off(self):
        pass

    @abstractmethod
    def set_prices(self):
        self.prices = np.linspace(.5*self.options[0].K, 1.5*self.options[0].K, 100)


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



# ls = ShortStraddle(100, 5)
# ls = ShortStrangle(50, 100, 5)
# plt.plot(lc.prices, lc.pay_off())
# plt.plot(lp.prices, lp.pay_off())
# plt.plot(plplon)
# plt.plot(plpshr)
# plt.plot(lc.prices, lc.pay_off())
# plt.plot(lp.prices, lp.pay_off())
plt.plot(sc.prices, sc.pay_off())
# po = ls.pay_off()
# cs = make_interp_spline([ls.prices[0], ls.prices[50], ls.prices[75], ls.prices[99]], [po[0], po[50], po[75], po[99]])
# plt.plot(ls.prices, cs(ls.prices))
plt.show()
