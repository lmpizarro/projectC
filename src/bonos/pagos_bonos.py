import pandas as pd
import pickle
from datetime import datetime, timedelta, date
from scrap_bonos import DAYS_IN_A_YEAR, ONE_BPS
from soberanos import get_nominals, valor_bono_disc
import matplotlib.pyplot as plt
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
from create_bonds import create_bullet_bond, create_amortizable_bond

t = np.asarray([0.0,   0.5, 1.0,   2.0,   3.0,   4.0,   5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
y = np.asarray([0.01, 0.02, 0.03, 0.035, 0.03, 0.035, 0.040, 0.05, 0.035, 0.037, 0.038, 0.04])
nsv_curve, status = calibrate_ns_ols(t, y, tau0=1.0)  # starting value of 1.0 for the optimization of tau

def draw_cash_flow(bono):
    cash_flow = [e[1] + e[2] for e in bono['pagos']]
    plt.bar(list(range(1, len(cash_flow)+1)), cash_flow)
    plt.show()


def dia_de_pago_mas_uno(bono, days=1):
    pagos = bono['pagos']

    l_d_p = []
    for pago in pagos:
        dia_de_pago = datetime.strptime(pago[0], "%d/%m/%y").date()
        mas_uno = dia_de_pago + timedelta(days=days)
        l_d_p.append(mas_uno)
    return l_d_p

def volatility_model(t, tao, sigma=0.0005, alfa=0.1, beta=.5):
    return np.random.normal(0, sigma*(1 - alfa*t/tao + beta))

def define_term_curve(r, total_dates, s=1, term='flat'):
    dates_index = np.asarray(list(range(total_dates))) / DAYS_IN_A_YEAR
    if term == 'flat':
        print('FLAT')
        curve = r * np.ones(dates_index.shape[0])
    elif term == 'exp_inv':
        print('INV')
        curve = r * (np.exp(-s*dates_index))
    elif term == 'exp':
        curve = r * (1 - np.exp(-s*dates_index))
    else:
        curve = nsv_curve(dates_index)
    return curve


def calc_hist_price(bono, r=0.05, init_date=datetime(2022, 9, 20).date(), term='exp_inv'):
    mem_pagos = []

    for pago in bono['pagos']:
        amortizacion = pago[2]
        renta = pago[1]
        mem_pagos.append([datetime.strptime(pago[0], "%d/%m/%y").date(), renta, amortizacion])
    total_dates = (mem_pagos[-1][0]-init_date).days

    curve = define_term_curve(r, total_dates, term=term)

    plt.plot(range(total_dates), curve)
    plt.show()
    valores = []
    durations = []
    for i in range(total_dates + 1):
        new_date = init_date + timedelta(days=i)
        valor_dia = 0
        duration_dia = []
        cupon = False
        for j, mp in enumerate(mem_pagos):

            if new_date < mp[0]:
                ttm_dates = (mp[0] - new_date).days
                r_curve = curve[ttm_dates-1] + volatility_model(ttm_dates, total_dates)
                ttm_years = ttm_dates/DAYS_IN_A_YEAR
                valor = mp[1]+ mp[2]
                valor_ajustado = valor * np.exp(-r_curve*ttm_years)
                duration = valor_ajustado * ttm_years

            else:
                valor_ajustado = 0
                duration = 0

            duration_dia.append(duration)
            valor_dia += valor_ajustado
            if new_date == mp[0]:
                cobro = mp[1] + mp[2]
                cupon = False

        if valor_dia != 0:
            valores.append(valor_dia)
            durations.append(np.asarray([duration_dia]).sum()/valor_dia)
            if cupon:
                cupon = False
                incremento = 1 + cobro / valor_dia 
                print('valor_dia ', cobro, valor_dia, incremento)
                mem_pagos = [[e[0], incremento * e[1], incremento * e[2]] for e in mem_pagos]


    return valores, durations



def test_calc_prices():
    bono = create_bullet_bond(years=4)
    bono = create_amortizable_bond()

    with open('bonos.pkl', 'rb') as fp:
        bonds = pickle.load(fp)

    bono = bonds['AL30']
    # bono = bonos['GD29']
    draw_cash_flow(bono)

    # draw_cash_flow(bono)

    val_per_dia, durations_dia = calc_hist_price(bono)

    plt.grid()
    plt.plot(val_per_dia)
    plt.axhline(94)
    plt.axhline(106)
    plt.show()

    plt.plot(durations_dia)
    plt.show()

def test_others():

    bono = create_bullet_bond()
    today = date.today()
    amortizacion, renta, pair_pagos = get_nominals(bono, today)

    from fitter import Fit

    rs = np.linspace(0.0001, 1.0001, 100)
    vs = np.zeros(100)
    for i, r in enumerate(rs):
        vs[i] = valor_bono_disc(pair_pagos, r)

    p0 = (vs[0], 5, 10) # start with values near those we expect
    m, t, b = Fit.optimizeExp(rs, vs, p0)

    modelPoly = Fit.polyModel(rs, vs)

    # 3 month minute
    total_hours = 3*30*24


    rss = ONE_BPS * np.sin(4*np.pi*np.linspace(0, total_hours, total_hours)/ total_hours) + .3
    plt.plot(rss)
    plt.show()
    plt.plot(Fit.monoExp(rss, m,t,b))
    plt.show()

    plt.plot(rs, vs)
    plt.plot(rs, Fit.monoExp(rs, m, t, b), 'k')
    plt.plot(rs, modelPoly(rs), color='purple')
    plt.show()

    plt.scatter(rss, Fit.monoExp(rss, m,t,b))
    plt.show()

    r1 = .2
    Dr = ONE_BPS
    r2 =  r1 + Dr
    dp1 = modelPoly(r2) -  modelPoly(r1)
    dpdr = dp1 / Dr
    dp2 = Fit.monoExp(r2, m, t, b) -  Fit.monoExp(r1, m, t, b)
    dp3 = np.interp(r2, rs, vs) - np.interp(r1, rs, vs)
    print(dpdr, dp2/Dr, dp3/Dr)


    plt.grid()
    datapc = []
    x = []
    l_d_p = dia_de_pago_mas_uno(bono)
    for dte in l_d_p[:-1]:
        amortizacion, renta, pair_pagos = get_nominals(bono, dte)
        x.append(len(pair_pagos))
        for i, r in enumerate(rs):
            vs[i] = valor_bono_disc(pair_pagos, r)
        Dr = ONE_BPS
        r1 = .001
        r2 =  r1 + Dr
        p1 = np.interp(r1, rs, vs)
        dp3 = np.interp(r2, rs, vs) - p1
        datapc.append((dp3/p1))
        # plt.plot(np.log(rs), np.log(vs))
        plt.plot(rs, vs)
    plt.show()

    plt.plot(x, datapc)
    plt.show()


def main():
    # test_others()
    test_calc_prices()

if __name__ == '__main__':
    main()

