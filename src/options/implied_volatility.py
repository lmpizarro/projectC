from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_from_yf import get_iv_surface

import QuantLib as ql

def setup_helpers(engine, expiration_dates, strikes,
                  data, ref_date, spot, yield_ts,
                  dividend_ts):
    heston_helpers = []
    grid_data = []
    for i, date in enumerate(expiration_dates):
        for j, s in enumerate(strikes):
            t = (date - ref_date )
            p = ql.Period(t, ql.Days)
            vols = data[i][j]
            helper = ql.HestonModelHelper(
                p, calendar, spot, s,
                ql.QuoteHandle(ql.SimpleQuote(vols)),
                yield_ts, dividend_ts)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
            grid_data.append((date, s))
    return heston_helpers, grid_data

def cost_function_generator(model, helpers,norm=False):
    def cost_function(params):
        params_ = ql.Array(list(params))
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers]
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error
    return cost_function

def calibration_report(helpers, grid_data, detailed=False):
    avg = 0.0
    if detailed:
        print("%15s %25s %15s %15s %20s" % (
            "Strikes", "Expiry", "Market Value",
             "Model Value", "Relative Error (%)"))
        print ("="*100)
    for i, opt in enumerate(helpers):
        err = (opt.modelValue()/opt.marketValue() - 1.0)
        date,strike = grid_data[i]
        if detailed:
            print("%15.2f %25s %14.5f %15.5f %20.7f " % (
                strike, str(date), opt.marketValue(),
                opt.modelValue(),
                100.0*(opt.modelValue()/opt.marketValue() - 1.0)))
        avg += abs(err)
    avg = avg*100.0/len(helpers)
    if detailed: print("-"*100)
    summary = "Average Abs Error (%%) : %5.9f" % (avg)
    print(summary)
    return avg

def setup_model(_yield_ts, _dividend_ts, _spot,
                init_condition=(0.02,0.2,0.5,0.1,0.01)):
    theta, kappa, sigma, rho, v0 = init_condition
    process = ql.HestonProcess(_yield_ts, _dividend_ts,
                           ql.QuoteHandle(ql.SimpleQuote(_spot)),
                           v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    return model, engine

def test01():
    pkl_path ='/home/lmpizarro/devel/project/financeExperiments/projectC/src/iv.pkl'
    # iv = implied_volatility('SPY')
    # iv.to_pickle()
    iv = pd.read_pickle(pkl_path)
    len_exp_dates = len(iv.keys())
    len_strikes = len(iv)
    #  strikes along the row dimension and expiries in the column
    implied_vols = ql.Matrix(len_strikes, len_exp_dates)
    np_iv = iv.to_numpy()

    print(iv.shape, np_iv.shape, len_strikes, len_exp_dates, implied_vols.columns())

    for i in range(implied_vols.rows()):
        for j in range(implied_vols.columns()):
            implied_vols[i][j] = np_iv[i][j]

    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates()

    calculation_date = ql.Date(8, 9, 2022)
    python_calculation_date = datetime(2022, 9, 8)
    expiration_dates = [python_calculation_date + timedelta(days=i)  for i in iv.keys()]

    expiration_dates = [ql.Date(ed.day, ed.month, ed.year) for ed in expiration_dates]
    print(expiration_dates)

    strikes = list(iv.index)

    spot = 400.38
    ql.Settings.instance().evaluationDate = calculation_date

    dividend_yield = ql.QuoteHandle(ql.SimpleQuote(0.0))
    risk_free_rate = 0.0329
    dividend_rate = 0.0154
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count))


    black_var_surface = ql.BlackVarianceSurface(
        calculation_date, calendar,
        expiration_dates, strikes, implied_vols, day_count)

    strike = 415.0
    expiry = 1.2 # years
    c_bs = black_var_surface.blackVol(expiry, strike)
    print(strike, expiry, c_bs)
    # dummy parameters
    v0 = 0.02; kappa = 0.12; theta = 0.025; rho = -0.85; sigma = 0.7;

    process = ql.HestonProcess(flat_ts, dividend_ts,
                               ql.QuoteHandle(ql.SimpleQuote(spot)),
                               v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    heston_helpers = []
    black_var_surface.setInterpolation("bicubic")
    one_year_idx = 24 # 12th row in data is for 1 year expiry
    date = expiration_dates[one_year_idx]
    for j, s in enumerate(strikes):
        t = (date - calculation_date )
        p = ql.Period(t, ql.Days)
        sigma = implied_vols[one_year_idx][j]
        #sigma = black_var_surface.blackVol(t/365.25, s)
        helper = ql.HestonModelHelper(p, calendar, spot, s,
                                      ql.QuoteHandle(ql.SimpleQuote(sigma)),
                                      flat_ts,
                                      dividend_ts)
        helper.setPricingEngine(engine)
        heston_helpers.append(helper)

    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    model.calibrate(heston_helpers, lm,
                     ql.EndCriteria(500, 400, 1.0e-8,1.0e-8, 1.0e-8))
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

class MyBounds(object):
     def __init__(self, xmin=[0.,0.01,0.01,-1,0], xmax=[1,15,1,1,1.0] ):
         self.xmax = np.array(xmax)
         self.xmin = np.array(xmin)
     def __call__(self, **kwargs):
         x = kwargs["x_new"]
         tmax = bool(np.all(x <= self.xmax))
         tmin = bool(np.all(x >= self.xmin))
         return tmax and tmin


if __name__ == '__main__':
    bounds = [(0,1),(0.01,15), (0.01,1.), (-1,1), (0,1.0) ]
    summary= []
    from scipy.optimize import differential_evolution

    pkl_path ='/home/lmpizarro/devel/project/financeExperiments/projectC/src/iv.pkl'
    iv = pd.read_pickle(pkl_path)
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates()

    calculation_date = ql.Date(8, 9, 2022)
    python_calculation_date = datetime(2022, 9, 8)
    expiration_dates = [python_calculation_date + timedelta(days=i)  for i in iv.keys()]

    expiration_dates = [ql.Date(ed.day, ed.month, ed.year) for ed in expiration_dates]
    print(expiration_dates)

    strikes = list(iv.index)

    risk_free_rate = 0.0329
    dividend_rate = 0.0154
    yield_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count))


    np_iv = iv.to_numpy()
    spot = 400.38
    data = np_iv.T

    """
    model4, engine4 = setup_model(yield_ts, dividend_ts, spot)
    heston_helpers4, grid_data4 = setup_helpers(
        engine4, expiration_dates, strikes, data,
        calculation_date, spot, yield_ts, dividend_ts
    )
    initial_condition = list(model4.params())
    bounds = [(0,1),(0.01,15), (0.01,1.), (-1,1), (0,1.0) ]

    cost_function = cost_function_generator(
    model4, heston_helpers4, norm=True)

    def calculator(cost_function, bounds):
        return differential_evolution(cost_function, bounds, maxiter=1000)

    sol = calculator(cost_function, bounds)

    theta, kappa, sigma, rho, v0 = model4.params()
    print("theta = %f, kappa = %f, sigma = %f, rho = %f, v0 = %f" % \
    (theta, kappa, sigma, rho, v0))
    error = calibration_report(heston_helpers4, grid_data4, detailed=True)
    print(error)
    """

    """
        https://machinelearningmastery.com/basin-hopping-optimization-in-python/

    """
    model5, engine5 = setup_model(
    yield_ts, dividend_ts, spot,
    init_condition=(0.02,0.2,0.5,0.1,0.01))
    heston_helpers5, grid_data5 = setup_helpers(
        engine5, expiration_dates, strikes, data,
        calculation_date, spot, yield_ts, dividend_ts
    )
    initial_condition = list(model5.params())

    from scipy.optimize import basinhopping
    """
    http://gouthamanbalaraman.com/blog/valuing-european-option-heston-model-quantLib.html
    """

    mybound = MyBounds()
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds }
    cost_function = cost_function_generator(
            model5, heston_helpers5, norm=True)
    sol = basinhopping(cost_function, initial_condition, niter=10,
                       minimizer_kwargs=minimizer_kwargs,
                       stepsize=0.005,
                       accept_test=mybound,
                       interval=10)
    theta, kappa, sigma, rho, v0 = model5.params()
    print("theta = %f, kappa = %f, sigma = %f, rho = %f, v0 = %f" % \
    (theta, kappa, sigma, rho, v0))
    error = calibration_report(heston_helpers5, grid_data5, detailed=True)
    print(error)