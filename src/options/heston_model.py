from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
from numba import njit
"""
    https://quantpy.com.au/stochastic-volatility-models/simulating-heston-model-in-python/
"""
np.random.seed(0)

def correlated_sampler(N, M, rho):
    mu = np.array([0,0])
    cov = np.array([[1,rho],
                    [rho,1]])
    # sampling correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N,M))

    return Z

@njit()
def heston_model_sim(S0, r, v0, kappa, theta, sigma,T, N, M, Z):
    """
    Inputs:
     - S0, v0: initial parameters for asset and variance
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - T     : time of simulation
     - N     : number of time steps
     - M     : number of scenarios / simulations

    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # initialise other parameters
    dt = T/N

    # arrays for storing prices and variances
    S = np.full(shape=(N+1,M), fill_value=S0)
    v = np.full(shape=(N+1,M), fill_value=v0)

    print(S.shape)


    for i in range(1,N+1):
        S[i] = S[i-1] * np.exp( (r - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0] )
        # v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)
        v[i] = np.maximum((np.sqrt(v[i-1])+.5*sigma*np.sqrt(dt)*Z[i-1,:,1])**2 + kappa*(theta-v[i-1])*dt + .25*sigma*sigma*dt,0)

    return S, v

theta = 0.110948
kappa = 1.658242
sigma = 1.000000
rho = -0.520333
v0 = 0.043081
r = 0.0329
S0 = 400.38

T=1
N = 252                # number of time steps in simulation
M = 4            # number of simulations
"""
    - rho   : correlation between asset returns and variance
"""
Z = correlated_sampler(N, M, rho)
S,v = heston_model_sim(S0, r, v0, kappa, theta, sigma,T, N, M, Z)

print(S)

# Set strikes and complete MC option price for different strikes
K =np.arange(180, 500,10)
puts = np.array([np.exp(-r*T)*np.mean(np.maximum(k-S,0)) for k in K])
calls = np.array([np.exp(-r*T)*np.mean(np.maximum(S-k,0)) for k in K])
put_ivs = implied_vol(puts, S0, K, T, r, flag='p', q=0, return_as='numpy', on_error='ignore')
call_ivs = implied_vol(calls, S0, K, T, r, flag='c', q=0, return_as='numpy')

plt.plot(K, call_ivs, label=r'IV calls')
plt.plot(K, put_ivs, label=r'IV puts')
plt.ylabel('Implied Volatility')
plt.xlabel('Strike')
plt.title('Implied Volatility Smile from Heston Model')
plt.legend()
plt.show()

k = S0
calls = np.exp(-r*T)*np.mean(np.maximum(S-k,0))
print(calls)

# S_p,v_p = heston_model_sim(S0, v0, kappa, theta, sigma,T, N, M, Z)

"""
fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12,5))
time = np.linspace(0,T,N+1)
ax1.plot(time,S_p)
ax1.set_title('Heston Model Asset Prices')
ax1.set_xlabel('Time')
ax1.set_ylabel('Asset Prices')
ax2.plot(time,v_p)
ax2.set_title('Heston Model Variance Process')
ax2.set_xlabel('Time')
ax2.set_ylabel('Variance')
plt.show()
"""
last_S = S[:, -1] # for last column

print(last_S.mean(), last_S.std())
