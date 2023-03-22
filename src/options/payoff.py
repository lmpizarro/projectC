import numpy as np

import matplotlib.pyplot as plt

K = 100
P = 3

class Option:
    def __init__(self, K, P) -> None:
        self.K = K
        self.P = P

def long_strangle(op1: Option, op2: Option):

    s1 = np.linspace(0.5*op1.K, 1.5*op2.K)

    po_long_c = np.where(s1 > op2.K, s1-op2.K, 0)

    po_long_p = np.where(s1 < op1.K, op1.K-s1, 0)

    po = po_long_c + po_long_p
    return po, s1

def bull_call_spread(op1: Option, op2: Option):

    s1 = np.linspace(0.5*op1.K, 1.5*op2.K)

    po_long_c = np.where(s1 > op1.K, s1 - op1.K, 0)

    po_short_c = np.where(s1 > op2.K, (op2.K- s1), 0)

    po = (op2.P +  op1.P)*(po_long_c + po_short_c) / (op2.K - op1.K) - op1.P

    return po, s1


def long_call(op:Option):
    s1 = np.linspace(0.5*op.K, 1.5*op.K)
    po_long_c = np.where(s1 > op.K, s1-op.K, 0)

    return po_long_c, s1

def short_call(op:Option):

    po_long_c, s = long_call(op)

    return -po_long_c, s





po_long_straddle, s1 = long_strangle(Option(50, 5), Option(100,5))
po_bull_c_spread, s1 = bull_call_spread(Option(100, 2), Option(130,7))
po_long_call, sc = long_call(Option(50, 5))
po_short_call, ssc = short_call(Option(75, 5))
# plt.plot(s1, po_long_straddle)
# plt.plot(sc, po_long_call)
# plt.plot(ssc, po_short_call)
plt.plot(s1, po_bull_c_spread)

plt.show()