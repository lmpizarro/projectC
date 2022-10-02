import numpy as np
import matplotlib.pyplot as plt
from numba import njit


"""
    Euler and Milstein Discretization
    by Fabrice Douglas Rouah
"""
np.random.seed(0)

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
    kappa: float = .05
    theta: float = .1
    sigma: float = .1
    r:     float = .04


def heston_simulation():
    params = Parameters()

    N = 252 
    dt = 1 / N
    s0 = 100
    v0 = .1

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
    qss = []
    for i in range(M):
        qss.append(heston_simulation())

    for qs in qss:
        plt.plot(qs)
    plt.show()

def bsm_call_to_maturity(S0=50, K=50, r=0.05, sigma=0.3, T=0.5, N=1000):
    z = np.random.normal(0,1, N)
    a = r - np.power(sigma,2)*T
    w1 = z*sigma*np.sqrt(T)

    ks = np.exp(-r*T)*np.maximum(S0*np.exp(a + w1)-K, 0)
    print("mean ", np.mean(ks), "sd ", np.std(ks))


if __name__ == '__main__':
    bsm_call_to_maturity(N=100000)
