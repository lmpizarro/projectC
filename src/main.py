from cmath import exp
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from denoisers.butter.filter import min_lp
from plot.ploter import plot_stacked
from portfolios import (min_ewma_port,
                        equal_weight_port,
                        get_cross_matrix,
                        get_cross_var_keys,
                        get_ewma_keys,
                        vars_covars,
                        returns,
                        download)



rf = 0.015
import scipy.optimize as sco

def max_SR_opt(mean_returns, cov_matrix, rf_rate, n, display = False):

    def get_ret_vol_sr(weights, mean_returns, cov_matrix, rf_rate):
        weights = np.array(weights)
        ret = np.sum(np.array(mean_returns) * weights)
        vol = np.sqrt(np.dot(weights.T,np.dot(np.array(cov_matrix),weights)))
        sr = (ret-rf)/vol
        return np.array([ret,vol,sr])

    # minimize negative Sharpe Ratio
    def neg_sharpe(weights, mean_returns, cov_matrix, rf_rate):
        return -get_ret_vol_sr(weights, mean_returns, cov_matrix, rf_rate)[2]

    # check allocation sums to 1
    def check_sum(weights):
        return np.sum(weights) - 1

    # create constraint variable
    cons = ({'type':'eq','fun':check_sum})

    # create weight boundaries
    bounds = ((0,1),)*n

    # initial guess
    init_guess = [1/n]*n

    opt_results = sco.minimize(neg_sharpe, init_guess,
                               method='SLSQP', bounds = bounds,
                               constraints = cons,
                               args = (mean_returns, cov_matrix, rf_rate),
                               options = {'disp': display})

    weights = pd.Series(np.round(opt_results.x,2), index = mean_returns.index)
    return weights, opt_results.fun


def sharpe_fun(returns, a, w, rfr):
    ret = np.dot(w.T, np.array(returns)) - rfr
    risk =  np.sqrt(np.dot(w.T, np.dot(a, w)))
    return ret / risk

def weights_func(size):
    w = np.random.uniform(size=size)
    w = w / w.sum()
    wc = 1 - w
    wc = wc / wc.sum()

    return w, wc

def max_sharpe(symbols, df, rfr=0.0001, r=False):
    np.random.seed(1)
    N_index = 0
    N_max = 12
    for index, row in df.iterrows():

        N_index += 1
        a = get_cross_matrix(symbols, row_item=row)
        returns = row[[e+'_ewm' for e in symbols]]

        if not N_index % 60:
            if r:
                w, fun = max_SR_opt(returns, a, rfr, len(symbols))
                print(np.array(w), fun, index)
            else:
                max_s = -1000000
                max_w = None
                N = 0
                M = 0
                while True:
                    M += 1
                    if M == 3000000:
                        break

                    w, wc = weights_func()

                    sharpe = sharpe_fun(returns, a , w, rfr)
                    sharpe2 = sharpe_fun(returns, a , wc, rfr)

                    if sharpe2 > sharpe:
                        sharpe = sharpe2

                    if sharpe > max_s:
                        max_s = sharpe
                        max_w = w
                        N +=1
                        if N > N_max:
                            break

                if max_w is not None:
                    print(100*np.round(max_w,2), M, N, str(index).split(' ')[0])
                else:
                    print(max_w, index)

def control_var(symbols, df):
    keys = get_cross_var_keys(symbols)
    keys.extend(get_ewma_keys(symbols))
    s = df[keys].sum(axis=1)
    f_ema = (s).ewm(span=15).mean()
    s_ema = s.ewm(span=150).mean()
    control_ = ((f_ema - s_ema)<0)
    return control_

def control_var_key(symbol, df):
    s = df[symbol+'_var']
    f_ema = s.ewm(span=15).mean()
    s_ema = s.ewm(span=150).mean()
    control_ = ((f_ema - s_ema)<0)
    return control_

symbols = ['KO', 'PEP', 'PG', 'AAPL', 'JNJ', 'AMZN', 'DE', 'CAT', 'META', 'MSFT', 'ADI']
symbols = ['MSFT', 'AVGO', 'PG', 'PEP', 'AAPL', 'KO', 'LMT', 'TSLA', 'ADI', 'MELI', 'JNJ', 'SPY', 'AMZN', 'META']

def test_equal_weight():

    symbols = ['PG', 'PEP', 'AAPL']
    np.random.seed(1)
    symbols.sort()

    df_equal = equal_weight_port(symbols)


    plt.plot(df_equal['risk'], 'k')
    plt.show()

    plt.plot(df_equal['returns'].cumsum())
    plt.show()


def test_equal_min_weight():

    symbols = ['PG', 'PEP', 'AAPL']
    np.random.seed(1)
    symbols.sort()

    df_min = min_ewma_port(symbols)
    df_equal = equal_weight_port(symbols)

    diff__ = df_equal['risk']- df_min['risk']
    print(diff__.mean())

    plt.plot(df_min['risk'])
    plt.plot(df_equal['risk'], 'k')
    plt.show()
    plt.plot(diff__)
    plt.show()

    plt.plot(df_equal['returns'].cumsum())
    plt.plot(df_min['returns'].cumsum(), 'g')
    plt.show()

def heston_model():
    import QuantLib as ql

    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates()

    calculation_date = ql.Date(6, 11, 2015)

    spot = 659.37
    ql.Settings.instance().evaluationDate = calculation_date

    dividend_yield = ql.QuoteHandle(ql.SimpleQuote(0.0))
    risk_free_rate = 0.01
    dividend_rate = 0.0
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count))

    expiration_dates = [ql.Date(6,12,2015), ql.Date(6,1,2016), ql.Date(6,2,2016),
                    ql.Date(6,3,2016), ql.Date(6,4,2016), ql.Date(6,5,2016),
                    ql.Date(6,6,2016), ql.Date(6,7,2016), ql.Date(6,8,2016),
                    ql.Date(6,9,2016), ql.Date(6,10,2016), ql.Date(6,11,2016),
                    ql.Date(6,12,2016), ql.Date(6,1,2017), ql.Date(6,2,2017),
                    ql.Date(6,3,2017), ql.Date(6,4,2017), ql.Date(6,5,2017),
                    ql.Date(6,6,2017), ql.Date(6,7,2017), ql.Date(6,8,2017),
                    ql.Date(6,9,2017), ql.Date(6,10,2017), ql.Date(6,11,2017)]
    strikes = [527.50, 560.46, 593.43, 626.40, 659.37, 692.34, 725.31, 758.28]
    data = [
        [0.37819, 0.34177, 0.30394, 0.27832, 0.26453, 0.25916, 0.25941, 0.26127],
        [0.3445, 0.31769, 0.2933, 0.27614, 0.26575, 0.25729, 0.25228, 0.25202],
        [0.37419, 0.35372, 0.33729, 0.32492, 0.31601, 0.30883, 0.30036, 0.29568],
        [0.37498, 0.35847, 0.34475, 0.33399, 0.32715, 0.31943, 0.31098, 0.30506],
        [0.35941, 0.34516, 0.33296, 0.32275, 0.31867, 0.30969, 0.30239, 0.29631],
        [0.35521, 0.34242, 0.33154, 0.3219, 0.31948, 0.31096, 0.30424, 0.2984],
        [0.35442, 0.34267, 0.33288, 0.32374, 0.32245, 0.31474, 0.30838, 0.30283],
        [0.35384, 0.34286, 0.33386, 0.32507, 0.3246, 0.31745, 0.31135, 0.306],
        [0.35338, 0.343, 0.33464, 0.32614, 0.3263, 0.31961, 0.31371, 0.30852],
        [0.35301, 0.34312, 0.33526, 0.32698, 0.32766, 0.32132, 0.31558, 0.31052],
        [0.35272, 0.34322, 0.33574, 0.32765, 0.32873, 0.32267, 0.31705, 0.31209],
        [0.35246, 0.3433, 0.33617, 0.32822, 0.32965, 0.32383, 0.31831, 0.31344],
        [0.35226, 0.34336, 0.33651, 0.32869, 0.3304, 0.32477, 0.31934, 0.31453],
        [0.35207, 0.34342, 0.33681, 0.32911, 0.33106, 0.32561, 0.32025, 0.3155],
        [0.35171, 0.34327, 0.33679, 0.32931, 0.3319, 0.32665, 0.32139, 0.31675],
        [0.35128, 0.343, 0.33658, 0.32937, 0.33276, 0.32769, 0.32255, 0.31802],
        [0.35086, 0.34274, 0.33637, 0.32943, 0.3336, 0.32872, 0.32368, 0.31927],
        [0.35049, 0.34252, 0.33618, 0.32948, 0.33432, 0.32959, 0.32465, 0.32034],
        [0.35016, 0.34231, 0.33602, 0.32953, 0.33498, 0.3304, 0.32554, 0.32132],
        [0.34986, 0.34213, 0.33587, 0.32957, 0.33556, 0.3311, 0.32631, 0.32217],
        [0.34959, 0.34196, 0.33573, 0.32961, 0.3361, 0.33176, 0.32704, 0.32296],
        [0.34934, 0.34181, 0.33561, 0.32964, 0.33658, 0.33235, 0.32769, 0.32368],
        [0.34912, 0.34167, 0.3355, 0.32967, 0.33701, 0.33288, 0.32827, 0.32432],
        [0.34891, 0.34154, 0.33539, 0.3297, 0.33742, 0.33337, 0.32881, 0.32492]]

    implied_vols = ql.Matrix(len(strikes), len(expiration_dates))

    for i in range(implied_vols.rows()):
        for j in range(implied_vols.columns()):
            implied_vols[i][j] = data[j][i]

    black_var_surface = ql.BlackVarianceSurface(
                            calculation_date, calendar,
                            expiration_dates, strikes,
                            implied_vols, day_count)
    strike = 600.0
    expiry = 1.2 # years
    volatility = black_var_surface.blackVol(expiry, strike)
    print(volatility)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    strikes_grid = np.arange(strikes[0], strikes[-1],10)
    expiry = 1.0 # years
    implied_vols = [black_var_surface.blackVol(expiry, s)
                        for s in strikes_grid] # can interpolate here
    actual_data = data[11] # cherry picked the data for given expiry

    fig, ax = plt.subplots()
    ax.plot(strikes_grid, implied_vols, label="Black Surface")
    ax.plot(strikes, actual_data, "o", label="Actual")
    ax.set_xlabel("Strikes", size=12)
    ax.set_ylabel("Vols", size=12)
    legend = ax.legend(loc="upper right")
    # plt.show()

    plot_years = np.arange(0, 2, 0.1)
    plot_strikes = np.arange(535.0, 750.0, 1.0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(plot_strikes, plot_years)
    Z = np.array([black_var_surface.blackVol(y, x)
                  for xr, yr in zip(X, Y)
                      for x, y in zip(xr,yr) ]
                 ).reshape(len(X), len(X[0]))

    surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    # dummy parameters
    v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5;

    process = ql.HestonProcess(flat_ts, dividend_ts,
                               ql.QuoteHandle(ql.SimpleQuote(spot)),
                               v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    # engine = ql.FdHestonVanillaEngine(model)

    heston_helpers = []
    black_var_surface.setInterpolation("bicubic")
    one_year_idx = 11 # 12th row in data is for 1 year expiry
    date = expiration_dates[one_year_idx]
    for j, s in enumerate(strikes):
        t = (date - calculation_date )
        p = ql.Period(t, ql.Days)
        sigma = data[one_year_idx][j]
        # sigma = black_var_surface.blackVol(t/365.25, s)
        helper = ql.HestonModelHelper(p, calendar, spot, s,
                                      ql.QuoteHandle(ql.SimpleQuote(sigma)),
                                      flat_ts,
                                      dividend_ts)
        helper.setPricingEngine(engine)
        heston_helpers.append(helper)

    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    model.calibrate(heston_helpers, lm,
                     ql.EndCriteria(500, 50, 1.0e-8,1.0e-8, 1.0e-8))
    theta, kappa, sigma, rho, v0 = model.params()

    print("theta = %f, kappa = %f, sigma = %f, rho = %f, v0 = %f" % (theta, kappa, sigma, rho, v0))

    avg = 0.0
    print("%15s %15s %15s %20s" % (
        "Strikes", "Market Value",
         "Model Value", "Relative Error (%)"))
    print("="*70)
    for i, opt in enumerate(heston_helpers):
        err = (opt.modelValue()/opt.marketValue() - 1.0)
        print("%15.2f %14.5f %15.5f %20.7f " % (
            strikes[i], opt.marketValue(),
            opt.modelValue(),
            100.0*(opt.modelValue()/opt.marketValue() - 1.0)))
        avg += abs(err)
    avg = avg*100.0/len(heston_helpers)
    print("-"*70)
    print("Average Abs Error (%%) : %5.3f" % (avg))

from scipy.optimize import minimize

if __name__ == '__main__':
    df = download(['SPY', 'DIA'])

    df = returns(['SPY'], df)
    rets = (100 * df.SPY).to_numpy()
    # rets = rets[0:30]
    mu = rets.mean()
    residuals = rets - mu

    real_var = residuals ** 2
    theta = residuals.var()
    sigma = real_var.std()
    kappa = .5
    rho = 0
    v0 = theta
    expect_var = np.zeros(len(real_var))
    expect_var[0] = theta
    print(mu, theta, kappa, sigma)

    def g_exp_var(real_var, kappa, theta, v_0):
        exp_var = np.zeros(len(real_var))
        exp_var[0] = np.max(v_0, 0)
        for i in range(1, len(exp_var)):
            exp_var[i] = np.max(real_var[i-1] + kappa * (theta - np.max(real_var[i-1],0)),0)
            if exp_var[i] < 0:
                exp_var[i] = np.abs(exp_var[i])
            elif exp_var[i] == 0:
                exp_var[i] = 0.001
        return exp_var

    def MLEheston(parameters):
        theta_, kappa_, sigma_, rho_, v0_ = parameters[0], parameters[1], \
                    parameters[2], parameters[3], parameters[4]

        """
            X returns
            mu_x mean of return
            X - mu_x: deviation of returns  D3
            sigma_x expected variance F3

            Y realised variance E3
            mu_y expected variance F3
            sigma_y volatility of variance J11

            process 1  returns  mean mu_x sigma_x sqrt(variance[process 2])
            process 2 variance of returns sigma_y

        """
        exp_var = g_exp_var(real_var=real_var, kappa=kappa_, theta=theta_, v_0=v0_)
        f1 = residuals/np.sqrt(exp_var)
        f2 = (real_var - exp_var) / sigma_
        f12 = 2*rho_ * f1 * f2
        ext = np.exp(-((f1**2 + f2**2 - f12)/(2*(1-rho_**2))))
        den = 2*np.pi*np.sqrt(exp_var)*sigma_*np.sqrt(1-rho_**2)

        if np.count_nonzero(den) != len(den):
            a = -1000
        else:
            a = (np.log(ext / den)).sum()

        return -a

    print(g_exp_var(real_var, kappa, theta, v0))

    print(MLEheston([theta, kappa, sigma, rho, v0]))

    bounds = [(0,100),(0.01,150), (0.01,100.), (-0.9,0.9), (0,100.0)]

    t = np.linspace(bounds[0][0], bounds[0][1], 10)
    k = np.linspace(bounds[1][0], bounds[1][1], 10)
    s = np.linspace(bounds[2][0], bounds[2][1], 10)
    r = np.linspace(bounds[3][0], bounds[3][1], 10)
    v = np.linspace(bounds[4][0], bounds[4][1], 10)

    """
    a_min = 1000
    p_min = None
    for i in t:
        for j in k:
            for l in s:
                for m in r:
                    for n in v:
                        p = [i,j,l,m,n]
                        a = MLEheston(p)
                        if a < a_min:
                            a_min = a
                            p_min = p
                            print(a_min)

    print(a_min, p_min)
    """

    p_min = [55.555, 0.01, 11.12, 0.9999, 11.11]
    pars = minimize(MLEheston, p_min,
                bounds=bounds, method='L-BFGS-B')
    print(pars)

    exit()
    symbols = ['MSFT', 'AVGO', 'PG', 'BIL', 'SPY']
    symbols = ['BIL', 'HON', 'CL', 'AVGO', 'PG', 'PEP', 'XOM', 'KO', 'TXN', 'MO', 'XOM',
               'PM', 'KO', 'LMT', 'INTC', 'ADI', 'HD', 'JNJ', 'WFC', 'MCD', 'T', 'GS', 'WMT',
               'SPY']

    symbols = ['AAPL', 'TSLA', 'SPY']

    df:pd.DataFrame = download(symbols, denoise=False)
    df = min_lp(symbols, df)
    df.drop(columns=symbols, inplace=True)
    df.rename(columns={f'{s}_deno':s for s in symbols}, inplace=True)



    df_rets = returns(symbols, df)

    print(df_rets.tail())
    plot_stacked(symbols, df_rets, k='', start=250)


    # df_rets = df_rets.ewm(alpha=.05).mean()
    # df_rets = butter(symbols, df_rets, 70)
    # df_rets = min_lp(symbols, df_rets)
    # plot_stacked(symbols, df_rets, k='_deno', start=250)

    print(df_rets.tail())
    exit()

    df = download(symbols=symbols, years=10)
    df_rets = returns(symbols, df)
    data_neg = {}
    data_gt = {}
    for s in symbols:
        lt_zero = df_rets[s][df_rets[s] < 0]
        gt_zero = df_rets[s][df_rets[s] > 0]
        if s not in data_neg:
            data_neg[s] = {'ticker':s}
        if s not in data_gt:
            data_gt[s] = {'ticker':s}


        data_neg[s]['mean'] = lt_zero.mean()
        data_neg[s]['dev'] = lt_zero.std()
        data_neg[s]['count'] = lt_zero.count()

        data_gt[s]['mean'] = gt_zero.mean()
        data_gt[s]['dev'] = gt_zero.std()
        data_gt[s]['count'] = gt_zero.count()

    df_neg = pd.DataFrame.from_dict(data_neg, orient='index' )
    df_neg = df_neg.set_index('ticker')
    df_gt = pd.DataFrame.from_dict(data_gt, orient='index')
    df_gt = df_gt.set_index('ticker')

    # df_gt['count'] = df_gt['count'] / df_gt['count'].loc['SPY']
    # df_gt['mean'] = df_gt['mean'] / df_gt['mean'].loc['SPY']
    # df_gt['dev'] = df_gt['dev'] / df_gt['dev'].loc['SPY']
    # df_gt['sum'] = df_gt.sum(axis=1)
    # df_gt.drop('SPY', inplace=True)

    # df_neg['count'] = df_neg['count'].loc['SPY'] / df_neg['count']
    # df_neg['mean'] = df_neg['mean'].loc['SPY'] / df_neg['mean']
    # df_neg['dev'] = df_neg['dev'].loc['SPY'] / df_neg['dev']
    # df_neg['sum'] = df_neg.sum(axis=1)
    # df_neg.drop('SPY', inplace=True)

    # df_gt['total'] = df_neg['sum'] + df_gt['sum']
    # total = df_gt.total[symbols].sum()

    # df_gt.total = df_gt.total / total
    print(df_neg)
    print(df_gt)
    # print(total)
    # print(df_gt['total'][symbols].sum())

    exit()
    df = yf.download(symbols, '2015-2-1')['Adj Close']
    df_prices = copy.deepcopy(df)



    lmbd = .94
    df_ewma = vars_covars(df, lmbd, mode='teor')

    print(df_prices.tail())
    print(df_ewma.tail())

    plot_stacked(symbols, df_ewma, '_ewma')

    print(df_ewma.keys())
    print(df.keys())



    # max_sharpe(symbols, df)
    # min_var(symbols, df)

    #plt.plot(df.KO_var)
    #plt.plot(df.PG_var> df.KO_var)
    #plt.plot(df.AAPL_var > df.KO_var, 'r')
    # plt.plot(df.AAPL_var/ df.AAPL_ewm, 'r')
    plt.plot(df.PG_csum, 'g')
    plt.show()

    print(df.KO.describe())
    control_ = control_var_key('PG', df)
    df.PG = df.PG*control_

    plt.plot(df.PG.cumsum())
    plt.show()

    # probar n portfolios aleatorios
    # https://medium.com/@Piotr_Szymanski/arithmetic-vs-log-stock-returns-in-python-7f7c3cff125


    ret_df = df[get_return_keys(symbols)]

    port_ret = ret_df.dot(equal_weights)
    dff = port_ret.ewm(alpha=1-lmbd).mean()
    dff = (dff**2).ewm(alpha=1-lmbd).mean()
    port = port_ret.cumsum()

    plt.plot(dff)

    for i in range(3):
        w, wc = weights_func(len(symbols))
        port_ret = ret_df.dot(w)
        dff = port_ret.ewm(alpha=1-lmbd).mean()
        dff = (dff**2).ewm(alpha=1-lmbd).mean()
        plt.plot(dff)

    plt.show()

    print(port.tail())