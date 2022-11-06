from unittest.mock import CallableMixin
import pandas as pd
from datetime import datetime, timedelta, date
from scrap_bonos import DAYS_IN_A_YEAR, ONE_BPS
from soberanos import get_nominals, valor_bono_disc
import matplotlib.pyplot as plt
import numpy as np


def create_bullet_bond(face: float=100, years: float=10, pays_per_year: int=2,
                       rate: float=.05, first_pay: str='09/01/23'):

    pagos = []
    pay_date = datetime.strptime(first_pay, "%d/%m/%y").date()
    for i in range(years-1):
        p_rate = rate / pays_per_year
        pago = p_rate * face
        for j in range(pays_per_year):
            pagos.append((pay_date.strftime("%d/%m/%y"), pago, 0))
            pay_date = pay_date + timedelta(days=180)
    pagos.append((pay_date.strftime("%d/%m/%y"), pago, 0))

    pay_date = pay_date + timedelta(days=180)
    pagos.append((pay_date.strftime("%d/%m/%y"), pago, 100))
    name = f'B{rate}-{str(pay_date.year)[2:]}'
    bono = {}
    bono['pagos'] = pagos

    return bono

def convert_bullet_to_amort(bono, periods=10):
    for i in range(-1, -(periods+1), -1):
        if i == -1:
            amrt =  -bono['pagos'][i][1] + bono['pagos'][i][2] / periods
            tuple__ = (bono['pagos'][i][0], bono['pagos'][i][1], amrt)
            bono['pagos'][i] = tuple__
            # amrt = 20 / 4

        tuple__ = (bono['pagos'][i][0], bono['pagos'][i][1], amrt)
        bono['pagos'][i] = tuple__
    return bono

def draw_cash_flow(bono):
    cash_flow = [e[1] + e[2] for e in bono['pagos']]
    plt.bar(list(range(1, len(cash_flow)+1)), cash_flow)
    plt.show()


def dias_de_pago_mas_uno(bono, days=1):
    pagos = bono['pagos']

    l_d_p = []
    for pago in pagos:
        dia_de_pago = datetime.strptime(pago[0], "%d/%m/%y").date()
        mas_uno = dia_de_pago + timedelta(days=days)
        l_d_p.append(mas_uno)
    return l_d_p

def volatility_model(t, tao, sigma=0.0005, alfa=0.1, beta=.5):
    return np.random.normal(0, sigma*(1 - alfa*t/tao + beta))

def define_term_curve(r, total_dates, term='flat'):
    dates_index = np.asarray(list(range(total_dates)))
    if term == 'flat':
        curve = r * np.ones(dates_index.shape[0])
    elif term == 'inv':
        curve = r * (np.exp(-dates_index/(2*_DAYS_IN_A_YEARN)))
    else:
        curve = r * (1 - np.exp(-dates_index/(2*DAYS_IN_A_YEAR)))
    return curve


def calc_hist_price(bono, r=0.05, init_date=datetime(2022, 10, 20).date(), term='flat'):
    mem_pagos = []

    for pago in bono['pagos']:
        amortizacion = pago[2]
        renta = pago[1]
        mem_pagos.append([datetime.strptime(pago[0], "%d/%m/%y").date(), renta, amortizacion])
    total_dates = (mem_pagos[-1][0]-init_date).days

    curve = define_term_curve(r, total_dates)

    valores = []
    durations = []
    for i in range(total_dates):
        new_date = init_date + timedelta(days=i)
        valor_dia = 0
        duration_dia = []
        for j, mp in enumerate(mem_pagos):
            if new_date < mp[0]:
                ttm_dates = (mp[0] - new_date).days
                r_curve = curve[ttm_dates-1] + volatility_model(ttm_dates, total_dates)
                ttm_years = ttm_dates/DAYS_IN_A_YEAR
                print(ttm_years)
                valor = mp[1]+ mp[2]
                valor_ajustado = valor * np.exp(-r_curve*ttm_years)
                duration = valor_ajustado * ttm_years
            else:
                valor_ajustado = 0
                duration = 0
            duration_dia.append(duration)
            valor_dia += valor_ajustado
        valores.append(valor_dia)
        durations.append(np.asarray([duration_dia]).sum()/valor_dia)
    return valores, durations

def test_calc_prices():
    bono = create_bullet_bond(years=4)

    # bono = bonos['GD29']
    draw_cash_flow(bono)

    # bono = convert_bullet_to_amort(bono)
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
    l_d_p = dias_de_pago_mas_uno(bono)
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

def calc_hist_reinv(bono, r=0.05, init_date=datetime(2022, 10, 20).date(), term='flat', cantidad=1000):
    mem_pagos = []

    face_value = 100
    for i, pago in enumerate(bono['pagos']):
        amortizacion = cantidad * pago[2]
        renta = cantidad * pago[1]
        mem_pagos.append([datetime.strptime(pago[0], "%d/%m/%y").date(), renta, amortizacion])
        a_comprar = 0
        if i < len(bono['pagos']) - 1:
            a_comprar = int((renta + amortizacion) / face_value)

        print(a_comprar, cantidad)
        cantidad = cantidad + a_comprar
    total_dates = (mem_pagos[-1][0]-init_date).days

    print(mem_pagos)

    cash_flow = [(e[1] + e[2])/100 for e in mem_pagos]
    plt.bar(list(range(1, len(cash_flow)+1)), cash_flow)
    plt.show()



def main():
    bono = create_bullet_bond(years=4)
    # calc_hist_reinv(bono)
    # test_others()
    test_calc_prices()

if __name__ == '__main__':
    main()

