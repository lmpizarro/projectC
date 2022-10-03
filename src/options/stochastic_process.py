import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba import jit
import math
import numpy as np



"""
    Euler and Milstein Discretization
    by Fabrice Douglas Rouah
"""
np.random.seed(0)

class Parameters:
    kappa: float = .4  
    theta: float = .040
    sigma: float = .000001
    r:     float = .04
    v0:    float = .04
    s0:    float = 50
    N:     int   = 252
    T:     float = 1 
    K:     float = 50


class HestonProcess:

    @staticmethod
    @njit()
    def s_milstein_bsm(st, r, sigma, dt, zs):
        st1 = st + r*st*dt + sigma*st*np.sqrt(dt)*zs + .5*sigma*sigma*dt*(zs*zs-1)
        return st1


    @staticmethod
    @njit()
    def s_milstein(st, r, vt, dt, zs):
        vtp = -vt  if vt < 0 else vt
        a = st + r*st*dt
        b = np.sqrt(vtp*dt)*st*zs
        st1 = a + b


        st1 = st*np.exp((r-.5*vt)*dt + np.sqrt(vtp*dt)*zs)
        return st1

    
    @staticmethod
    @njit()
    def v_milstein(vt, kappa, theta, sigma, zv, dt):
        vtp = -vt if vt < 0 else vt
        a = vt + kappa*(theta-vt)
        b = sigma*np.sqrt(vtp*dt)*zv
        c = .25*sigma*sigma*dt*(zv*zv-1)
        vt1 = a + b + c

        return vt1

@njit()
def npv(po, r, T):
    return np.exp(-r*T)*po

@njit()
def pay_off_call(S, K):
    return  np.maximum(S-K, 0)

@njit()
def pay_off_put(S, K):
    return  np.maximum(K-S, 0)

@njit()
def price_call(S, K, T, r):
    return np.mean(npv(pay_off_call(S,K), r, T))

@njit
def price_put(S, K, T, r):
    return np.mean(npv(pay_off_put(S,K), r, T))

def option_bsm_milstein(S, K, r, v, dt, N, M):

    last_s = np.zeros(M)
    for j in range(M):

        z = np.random.normal(0,1,N)
        s = np.zeros(N)
        s[0] = S
        for i in range(1, N):
            s[i] = HestonProcess.s_milstein_bsm(s[i-1], r, v, dt, z[i-1])
        last_s[j] = s[i]

    c=price_call(last_s, K, dt*N, r)
    p=price_put(last_s, K, dt*N, r)
    return c, p


@njit
def cnd_numba(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


def bsm_numba(S, K, r, v, T):
    """
       https://numba.pydata.org/numba-examples/examples/finance/blackscholes/results.html 
    """
    S = S
    X = K
    T = T
    R = r
    V = v
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_numba(d1)
    cndd2 = cnd_numba(d2)

    expRT = math.exp((-1. * R) * T)
    callResult = (S * cndd1 - X * expRT * cndd2)
    putResult = (X * expRT * (1.0 - cndd2) - S * (1.0 - cndd1))

    return callResult, putResult



@njit()
def bsm_mc_to_maturity(S0=50, K=50, r=0.04, sigma=0.3, T=1, N=100000):
    """
        capitulo 21 Hull
    """
    z = np.random.normal(0,1, N)
    a = r - np.power(sigma,2)*T
    w1 = z*sigma*np.sqrt(T)

    S1 = S0*np.exp(a + w1)
    S2 = S0*np.exp(a - w1)

    po1 = pay_off_call(S1, K)
    po2 = pay_off_call(S2, K)
    c = (npv(po1, r, T) + npv(po2, r, T))/2

    po1 = pay_off_put(S1, K)
    po2 = pay_off_put(S2, K)

    p = (npv(po1, r, T) + npv(po2, r, T))/2

    return np.mean(c), np.std(c), np.mean(p), np.std(p)

def simul_bsm_mc():
    params = Parameters()

    r = 0.04
    sigma = 0.2

    # cm, cs, pm, ps = bsm_mc_to_maturity(params.s0, params.K, r, sigma, params.T, N=100000)
    # print("call", cm, "put ", pm)
    c, p = bsm_numba(params.s0, params.K, r, sigma, params.T)
    print("call", c, "put ", p)
    c, p = option_bsm_milstein(params.s0, params.K, r, sigma, params.T/params.N, params.N, 10000)
    print("call", c, "put ", p)


@njit()
def gen_correlated(rho=-.9):
    z1 = np.random.normal(0, 1)
    z2 = np.random.normal(0, 1)
    zs = rho * z1 + np.sqrt(1-rho*rho)*z2
    return z1, zs

def poisson_process(lmbd: float=.01, N=1):
    """
        https://timeseriesreasoning.com/contents/poisson-process/
    """
    u = np.random.uniform(low=0, high=1, size=N)

    x = - np.log(1 - u) / lmbd
    x = x.astype(int)
    print(x)

    return x

def random_periodic_impulse(freq=8, N=504):
    u = np.random.uniform(low=0, high=1, size=N)
    b = u > (1 - freq/N)
    b = b.astype(int)
    return b


def heston_simulation():
    params = Parameters()

    N = params.N 
    dt = 1 / N
    s0 = params.s0
    v0 = params.v0 

    qs = s0 * np.ones(N)
    vs = v0 * np.ones(N)

    impulses = random_periodic_impulse(freq=8)

    v=0
    for i in range(1, N):
      

        z1, zs = gen_correlated(rho=-.9)
        v1 = HestonProcess.v_milstein(vs[i-1], 
                                       kappa=params.kappa, 
                                       theta=params.theta, 
                                       sigma=params.sigma, 
                                       zv=z1, 
                                       dt=dt)

        s1 = HestonProcess.s_milstein(qs[i-1], 
                                       r=params.r, 
                                       vt=vs[i-1],
                                       dt=dt, 
                                       zs=zs)

        qs[i] = s1
        vs[i] = v1 + 0 * np.random.normal(.00005, .1) * params.theta * impulses[i]

    return qs


def heston_mc(M):
    params = Parameters()
    S = np.zeros(M)
    for i in range(M):
        S[i]= heston_simulation()[-1]

    c = price_call(S, params.s0 ,params.r, params.T)
    p = price_put(S, params.s0 ,params.r, params.T)

    print('heston ', c, p)



if __name__ == '__main__':
    heston_mc(10000)
    simul_bsm_mc()

