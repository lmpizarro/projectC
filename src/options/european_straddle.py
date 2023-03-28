import numpy as np
from numba import jit
from scipy import stats


def opcion_europea_bs(tipo, S, K, T, r, sigma, div):
    #Defino los ds
    d1 = (np.log(S / K) + (r - div + 0.5 * sigma * sigma) * T) / sigma / np.sqrt(T)
    d2 = (np.log(S / K) + (r - div - 0.5 * sigma * sigma) * T) / sigma / np.sqrt(T)

    if (tipo == "C"):
        precio_BS = np.exp(-div*T) *S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    elif (tipo == "P"):
        precio_BS = K * np.exp(-r * T) * stats.norm.cdf(-d2) - np.exp(-div*T) * S * stats.norm.cdf(-d1)
    return precio_BS


@jit(nopython=True)
def log_normal_process(S0:float, r:float, div:float, sigma:float, T:float, pasos:int=10000):
    z = np.random.normal(0,1, pasos)
    A = (r - div - 0.5 * np.power(sigma,2))*T
    B0 =sigma*z*np.sqrt(T)
    C = S0*np.exp(A)
    prices0 = C*np.exp(B0)
    prices1 = C*np.exp(-B0)
    return prices0, prices1

S0 = 120
K = 120
r = 0.03
div = 0
sigma = .3
T = 1/12

@jit(nopython=True)
def straddle_price_mc(S0, K, r, div, sigma, T):
    prices0, prices1 = log_normal_process(S0, r, div, sigma, T)

    pay_off_C0 = np.where(prices1 < K, 0, prices1 - K)
    pay_off_C1 = np.where(prices0 < K, 0, prices0 - K)
    C  = .5 * np.exp(-r * T) * ((pay_off_C0 + pay_off_C1).mean())

    pay_off_P0 = np.where(prices1 < K, -prices1 + K, 0)
    pay_off_P1 = np.where(prices0 < K, -prices0 + K, 0)
    P  = .5 * np.exp(-r * T) * ((pay_off_P0 + pay_off_P1).mean())

    return C, P, P + C

def straddle_price_bs(S0, K, r, div, sigma, T):
    C = opcion_europea_bs("C", S0, K, T, r, sigma, div)
    P = opcion_europea_bs("P", S0, K, T, r, sigma, div)

    return C, P, P + C

def draw_T_sigma_surface_for_straddle():
    times = np.linspace(1/252, 1, 100)
    sigmas = np.linspace(0.1, 1.5, 100)
    x_2d, y_2d = np.meshgrid(times ,sigmas)

    matrix = np.zeros(100*100).reshape(100, 100)
    for j,T in enumerate(times):
        for k, sigma in enumerate(sigmas):
            C, P, _ = straddle_price_bs(S0, K, r, div, sigma, T)
            matrix[j,k] = C + P
            # print(f'price sraddle call {C:.2f} put {P:.2f} all {P+C:.2f}')

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_2d, y_2d, matrix, cmap=cm.jet)


    plt.show()

import matplotlib.pyplot as plt
def time_effect_straddle():
    for T in np.linspace(0.01, 0.99, 5):
        prices = np.linspace(80, 160, 20)
        C = opcion_europea_bs("C", prices, 120, 1 - T, 0.02, 0.3, 0)
        P = opcion_europea_bs("P", prices, 120, 1 - T, 0.02, 0.3, 0)
        c = np.where(C > P, C, 0)
        p = np.where(P > C, P, 0)
        plt.plot(prices, c+p, label=f'{T}')
        leg = plt.legend(loc='upper center')
    plt.show()
    print(C)

x  = np.linspace(0, 2*np.pi, 100)
y = 50 * np.sin(x) + S0
Pt = 20
K = 120

y1 = np.where(y > (K + Pt), 100*(y - (K  + Pt))/Pt, 0)
y2 = np.where(y < (K - Pt), 100*(-y + (K  - Pt))/Pt, 0)
y3 = y1 + y2
y3 = np.where(y3 == 0, -100, y3)
plt.plot(x, y3)
plt.plot(x, y, 'g')
plt.hlines(0, 0, x[-1], linestyles='dotted', colors='k')
plt.hlines(120, 0, x[-1], linestyles='dashed', colors='k')
plt.hlines(120+Pt, 0, x[-1], linestyles='dashdot', colors='r')
plt.hlines(120-Pt, 0, x[-1], linestyles='dashdot', colors='r')
plt.show()

