import numpy as np
import numba

p0 = 0
N = 10000

rnd = np.random.normal(0,1,N)

print(np.power(rnd, 2).sum())
print(rnd.sum())


@numba.jit()
def log_normal_random_walk(N=1*252, mu=1E-5, sigma=1E-3, dt=1):
    rnd = np.random.normal(0,1,N) 
    tt = np.ones(len(rnd),dtype=np.float32)

    for i in range(1, len(tt)):
        tt[i] = tt[i-1] * (1 + mu * dt + sigma* rnd[i])

    return tt

@numba.jit()
def brownian_drift(N=1*252, mu=1E-5, sigma=1E-3, dt=1):
    rnd = np.random.normal(0,1,N) 
    tt = np.ones(len(rnd),dtype=np.float32)

    for i in range(1, len(tt)):
        tt[i] = tt[i-1]  + mu * dt + sigma* rnd[i]

    return tt


@numba.jit()
def mean_reverting(N=4*252, tt0 = 9, nu=.04, mu=1E-2, sigma=3E-2, dt=1):

    print(nu/mu)
    rnd = np.random.normal(0,1,N) 
    tt = np.ones(len(rnd),dtype=np.float32)

    tt[0] = tt0
    for i in range(1, len(tt)):
        tt[i] = tt[i-1] + (nu - mu * tt[i-1]) * dt + sigma* rnd[i]


    return tt

@numba.jit()
def mean_reverting_cir(N=4*252, tt0 = 9, nu=.04, mu=1E-2, sigma=3E-2, dt=1):

    print(nu/mu)
    rnd = np.random.normal(0,1,N) 
    tt = np.ones(len(rnd),dtype=np.float32)

    tt[0] = tt0
    for i in range(1, len(tt)):
        tt[i] = tt[i-1] + (nu - mu * tt[i-1]) * dt + sigma * np.sqrt(tt[i-1]) * rnd[i]


    return tt



tt1 = mean_reverting_cir()
tt2 = mean_reverting()

import matplotlib.pyplot as plt

plt.grid()
plt.plot(tt1)
plt.plot(tt2)
plt.show()


