import numpy as np
import matplotlib.pyplot as plt
from numba import njit


"""
    Euler and Milstein Discretization
    by Fabrice Douglas Rouah
"""
np.random.seed(0)

@njit()
def bsm_call_to_maturity(S0=50, K=50, r=0.04, sigma=0.3, T=0.5, N=100000):
    """
        capitulo 21 Hull
    """
    z = np.random.normal(0,1, N)
    a = r - np.power(sigma,2)*T
    w1 = z*sigma*np.sqrt(T)

    po1 = np.maximum(S0*np.exp(a + w1)-K, 0)
    po2 = np.maximum(S0*np.exp(a - w1)-K, 0)
    c = np.exp(-r*T)*(po2 + po1)/2

    po1 = np.maximum(K-S0*np.exp(a + w1), 0)
    po2 = np.maximum(K-S0*np.exp(a - w1), 0)
    p = np.exp(-r*T)*(po2 + po1)/2


    return np.mean(c), np.std(c), np.mean(p), np.std(p)

def simul_bsm_mc():
    cm, cs, pm, ps = bsm_call_to_maturity(N=100000)
    print("mean call", cm, "std call", cs)
    print("mean put", pm, "std put", ps)

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

class Parameters:
    kappa: float = .02
    theta: float = .15
    sigma: float = .1
    r:     float = .04
    v0:    float = .1
    s0:    float = 50
    N:     int   = 252
    T:     float = .5


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
      

        z1, zs = gen_correlated(rho=-.5)
        v1 = HestonProcess.v_milstein(vs[i-1], 
                                       kappa=params.kappa, 
                                       theta=params.theta, 
                                       sigma=params.sigma, 
                                       zv=z1, 
                                       dt=dt)

        s1 = HestonProcess.s_milstein_bsm(qs[i-1], 
                                       r=params.r, 
                                       sigma=vs[i-1],
                                       # vt=v,
                                       dt=dt, 
                                       zs=zs)


        qs[i] = s1
        vs[i] = v1 + np.random.normal(.5, .1) * params.theta * impulses[i]

    return qs


def heston_mc(M):
    params = Parameters()
    qss = []
    for i in range(M):
        qss.append(heston_simulation())

    price_to_maturity = np.zeros(M)
    for i, qs in enumerate(qss):
        price_to_maturity[i]= qs[-1]

    po = np.maximum(price_to_maturity-params.s0, 0)
    c = np.exp(-params.r*params.T)*po

    print(np.mean(c))



if __name__ == '__main__':
    heston_mc(5000)
    # simul_bsm_mc()

