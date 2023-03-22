from payoff.put_call import LongCall, LongPut, ShortCall, ShortPut
from payoff.pay_off import Option

import matplotlib.pyplot as plt

opt1 = Option(7800,.3,10,.1,.1, 'C', 79)
lc = LongCall(options=[opt1])
sc = ShortCall(options=[opt1])
opt2 = Option(7900,.3,10,.1,.1, 'P', 25)
lp = LongPut(options=[opt2])
sp = ShortPut(options=[opt2])

plt.plot(lc.prices, lc.pay_off())
plt.show()

from payoff.call_spread import BullCallSpread

bcs = BullCallSpread(options=[opt1, opt2])



plt.plot(bcs.prices, bcs.pay_off())
plt.show()