import numpy as np
import scipy.optimize


class Fit:
    @staticmethod
    def monoExp(x, m, t, b):
        return m * np.exp(-t * x) + b

    @staticmethod
    def optimizeExp(rs, vs, p0):
        params, cv = scipy.optimize.curve_fit(Fit.monoExp, rs, vs, p0)
        m, t, b = params

        return m, t, b
    
    @staticmethod
    def polyModel(rs, vs):
        return np.poly1d(np.polyfit(rs, vs, 8))


