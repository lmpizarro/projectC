import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from numba import njit
from data_from_yf import get_vol_surface


class HestonParameters:

    @staticmethod
    @njit()
    def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):

        # constants
        a = kappa*theta
        b = kappa+lambd

        # common terms w.r.t phi
        rspi = rho*sigma*phi*1j

        # define d parameter given phi and b
        d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )

        # define g parameter given phi, b and d
        g = (b-rspi+d)/(b-rspi-d)

        # calculate characteristic function by components
        exp1 = np.exp(r*phi*1j*tau)
        term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
        exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)

        return exp1*term2*exp2

    @staticmethod
    @njit()
    def integrand(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r, K):
        args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
        numerator = np.exp(r*tau)*HestonParameters.heston_charfunc(phi-1j,*args) - \
                        K*HestonParameters.heston_charfunc(phi,*args)
        denominator = 1j*phi*K**(1j*phi)
        return numerator/denominator

    @staticmethod
    @njit()
    def heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
        args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r, K)

        real_integral, err = np.real( quad(HestonParameters.integrand, 0, 100, args=args) )

        return (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi


    @staticmethod
    def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
        args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

        Prices, umax, N = 0, 100, 10000
        dphi=umax/N #dphi is width

        for i in range(1,N):
            # rectangular integration
            phi = dphi * (2*i + 1)/2 # midpoint to calculate height
            numerator = np.exp(r*tau)*HestonParameters.heston_charfunc(phi-1j,*args) - \
                K * HestonParameters.heston_charfunc(phi,*args)

            denominator = 1j*phi*K**(1j*phi)

            Prices += dphi * numerator/denominator

        return np.real((S0 - K*np.exp(-r*tau))/2 + Prices/np.pi)


def SqErr(x):
    v0, kappa, theta, sigma, rho, lambd = [param for param in x]

    # Attempted to use scipy integrate quad module as constrained to single floats not arrays
    # err = np.sum([ (P_i-heston_price(S0, K_i, v0, kappa, theta, sigma, rho, lambd, tau_i, r_i))**2 /len(P) \
    #               for P_i, K_i, tau_i, r_i in zip(marketPrices, K, tau, r)])

    # Decided to use rectangular integration function in the end
    err = np.sum( (P-HestonParameters.heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r))**2 / len(P) )
    print(f'...........{err}')

    # Zero penalty term - no good guesses for parameters
    pen = 0 #np.sum( [(x_i-x0_i)**2 for x_i, x0_i in zip(x, x0)] )

    return err + pen


if __name__ == '__main__':
    yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    yields = np.array([0.15,0.27,0.50,0.93,1.52,2.13,2.32,2.34,2.37,2.32,2.65,2.52]).astype(float)/100

    rate_structure = {'yields': yields,
                      'maturities': yield_maturities}

    volSurfaceLong, S0 = get_vol_surface('TSLA', rate_structure)

    # This is the calibration function
    # heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
    # Parameters are v0, kappa, theta, sigma, rho, lambd
    # Define variables to be used in optimization
    r = volSurfaceLong['rate'].to_numpy('float')
    K = volSurfaceLong['strike'].to_numpy('float')
    tau = volSurfaceLong['maturity'].to_numpy('float')
    P = volSurfaceLong['price'].to_numpy('float')

    params_minimizer = {"v0": {"x0": 0.1, "lbub": [1e-3,0.1]},
              "kappa": {"x0": 3, "lbub": [1e-3,5]},
              "theta": {"x0": 0.05, "lbub": [1e-3,0.1]},
              "sigma": {"x0": 0.3, "lbub": [1e-2,1]},
              "rho": {"x0": -0.8, "lbub": [-1,0]},
              "lambd": {"x0": 0.03, "lbub": [-1,1]},
            }

    x0 = [param["x0"] for key, param in params_minimizer.items()]
    bnds = [param["lbub"] for key, param in params_minimizer.items()]

    print('begin')
    result = minimize(SqErr, x0, tol = 1e-3, method='SLSQP',
                      options={'maxiter': 1e4 }, bounds=bnds)

    v0, kappa, theta, sigma, rho, lambd = [param for param in result.x]
    print(v0, kappa, theta, sigma, rho, lambd)
    # 15 09 SPY
    # theta 0.07049825885996333
    # kappa 3.3900924886498793
    # sigma 0.47390978289578967
    # rho  -0.7807785506141585
    # v0    0.05814620900128352
    # lambda 0.1869936190997598
    # tsla
    # 0.1 0.0009999999999998899 0.1 0.010147513947524223 0.0 -1.0