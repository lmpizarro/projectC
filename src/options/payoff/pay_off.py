import numpy as np
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

