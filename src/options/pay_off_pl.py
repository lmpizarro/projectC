import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
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
    def __init__(self, option: Option) -> None:
        if option.type != 'C':
            raise ValueError

        super().__init__([option])
        self.set_prices()

    def pay_off(self):
        po = np.where(self.prices < self.options[0].K,
                      0,
                      self.prices - self.options[0].K)
        return po

    def set_prices(self):
        return super().set_prices()

class ShortCall(PayOff):
    def __init__(self, K: float, P: float) -> None:
        self.lc = LongCall(K, P)

    def pay_off(self):
        return - self.lc.pay_off()

class LongPut(PayOff):
    def __init__(self, option: Option) -> None:
        if option.type != 'P':
            raise ValueError

        super().__init__([option])
        self.set_prices()

    def pay_off(self):
        po = np.where(self.prices < self.options[0].K, - self.prices + self.options[0].K, 0)
        return po

    def set_prices(self):
        return super().set_prices()

class ShortPut(PayOff):
    def __init__(self, K: float, P: float) -> None:
        self.lp = LongPut(K, P)

    def pay_off(self):
        return - self.lp.payoff()

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


opt1 = Option(7800,.3,10,.1,.1, 'C', 79)
lc = LongCall(option=opt1)
opt2 = Option(7900,.3,10,.1,.1, 'P', 25)
lp = LongPut(option=opt2)
# ls = ShortStraddle(100, 5)
# ls = ShortStrangle(50, 100, 5)
ls = BullCallSpread(options=[opt1, opt2])
# plt.plot(lc.prices, lc.pay_off())
# plt.plot(lp.prices, lp.pay_off())
# plt.plot(plplon)
# plt.plot(plpshr)
# plt.plot(lc.prices, lc.pay_off())
# plt.plot(lp.prices, lp.pay_off())
plt.plot(ls.prices, ls.pay_off())
# po = ls.pay_off()
# cs = make_interp_spline([ls.prices[0], ls.prices[50], ls.prices[75], ls.prices[99]], [po[0], po[50], po[75], po[99]])
# plt.plot(ls.prices, cs(ls.prices))
plt.show()
