import numpy as np
import matplotlib.pyplot as plt

class PayOff:

    def set_prices(self, min: float, max: float):
        self.s = np.linspace(min, max, 100)

    def long_call(self, K: float):

        return np.where(self.s < K, 0, self.s-K)

    def short_call(self, K: float):

        po = self.long_call(K)
        return  - po

    def long_put(self, K: float):

        return np.where(self.s < K, -self.s+K, 0)

    def short_put(self, K: float):

        po = self.long_put(K)
        return  - po

    def long_straddle(self, K:float):
        p = self.long_put(K)
        c = self.long_call(K)
        return p + c

    def short_straddle(self, K:float):
        return - self.long_straddle(K)

    def long_strangle(self, Kp: float, Kc:float):
        if Kc < Kp:
            raise ValueError

        p = self.long_put(Kp)
        c = self.long_call(Kc)
        return p + c

    def bull_call_spread(self, Kcl: float, Kcs:float) -> np.asarray:
        if Kcs < Kcl:
            raise ValueError

        cl = self.long_call(Kcl)
        cs = self.short_call(Kcs)
        return cl + cs

    def long_butterfly(self, Kpl: float, Km: float, Kcl: float):

        pl = self.long_call(Kpl)
        smp = self.short_call(Km)
        smc = self.short_call(Km)
        cl = self.long_call(Kcl)

        po = pl + smp + smc + cl

        return po

    def short_butterfly(self, Kpl: float, Km: float, Kcl: float):

        pl = self.short_call(Kpl)
        smp = self.long_call(Km)
        smc = self.long_call(Km)
        cl = self.short_call(Kcl)

        po = pl + smp + smc + cl

        return po

    def covered_call(self, Ksc: float):

        sc = self.short_call(Ksc)

        return sc + self.s

    def synth_long_stock(self, K: float):
        """ long synthetic future """
        lc = self.long_call(K)
        sp = self.short_put(K)

        return sp + lc

    def synth_short_stock(self, K: float):
        """ short synthetic future """
        lc = self.long_put(K)
        sp = self.short_call(K)

        return sp + lc

    def synth_call(self, K: float):

        lp = self.long_put(K)

        return self.s + lp

    def synth_put(self, K: float):

        lc = self.long_call(K)

        return - self.s + lc

    def synth_straddle(self, K):

        lc = self.long_call(K)

        return - self.s + 2 * lc


po = PayOff()
po.set_prices(0, 250)
poc = po.short_butterfly(100, 125, 150)
cover_c = po.covered_call(100)
syn_stock = po.synth_long_stock(100)
syn_stock = po.synth_straddle(100)
plt.plot(po.s, syn_stock)
plt.show()
