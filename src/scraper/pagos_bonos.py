import pandas as pd
from datetime import datetime, timedelta, date
import pickle
from scrap_bonos import N_DAYS

bonos = {
            "GD29": {"pagos": [('09/01/23', 0.5, 0), ('09/07/23', 0.5, 0), 
                               ('09/01/24', 0.5, 0), ('09/07/24', 0.5, 0), 
                               ('09/01/25', 0.5, 10), ('09/07/25', 0.45, 10),
                               ('09/01/26', 0.4, 10), ('09/07/26', 0.35, 10),
                               ('09/01/27', 0.3, 10), ('09/07/27', 0.25, 10),
                               ('09/01/28', 0.2, 10), ('09/07/28', 0.15, 10),
                               ('09/01/29', 0.1, 10), ('09/07/29', 0.05, 10),
                               ], "pay_per_year":2},
            "GD30": {"pagos": [('09/01/23', 0.25, 0), ('09/07/23', 0.25, 0), 
                               ('09/01/24', 0.38, 0), ('09/07/24', 0.38, 4), 
                               ('09/01/25', 0.36, 8), ('09/07/25', 0.33, 8),
                               ('09/01/26', 0.3, 8), ('09/07/26', 0.27, 8),
                               ('09/01/27', 0.24, 8), ('09/07/27', 0.21, 8),
                               ('09/01/28', 0.42, 8), ('09/07/28', 0.35, 8),
                               ('09/01/29', 0.28, 8), ('09/07/29', 0.21, 8),
                               ('09/01/30', 0.14, 8), ('09/07/30', 0.07, 8),
                               ],  "pay_per_year":2},
            }


nombre_bonos = {'GD' :[35,38,41,46]}

def process_csv(nombre_bonos):
    for tipo in nombre_bonos:
        for anio in nombre_bonos[tipo]:
            df_35 = pd.read_csv(f"flujoFondos_{tipo}{anio}.csv")

            bono = {}
            pagos = []
            for row in df_35.iterrows():
                fecha = row[1]["Fecha de pago"].split('/')
                fecha = '/'.join([fecha[2], fecha[1], fecha[0][2:]])
                renta = row[1]["Renta"]
                amort = row[1]["Amortización"]
                pagos.append((fecha, renta, amort))
            bono['pagos'] = pagos
            bono["pay_per_year"] = 2
            ticker = row[1]['Ticker']

            bonos[ticker] = bono

    with open('bonos.pkl', 'wb') as fp:
        pickle.dump(bonos, fp)

def create_bullet_bond(face: float=100, years: float=10, pays_per_year: int=2, rate: float=.05, first_pay: str='09/01/23'):
   
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

def dias_de_pago_mas_uno(bono, days=1):
    pagos = bono['pagos']
    
    l_d_p = []
    for pago in pagos:
        dia_de_pago = datetime.strptime(pago[0], "%d/%m/%y").date()
        mas_uno = dia_de_pago + timedelta(days=days)
        l_d_p.append(mas_uno)
    return l_d_p

from soberanos import get_nominals, valor_bono_disc
import numpy as np

bono = create_bullet_bond()

# bono = bonos['GD29']

print(bono)

import matplotlib.pyplot as plt
def calc_hist_price(bono, r=0.051, init_date=datetime(2022, 10, 20).date(), term='norm'):
    mem_pagos = []

    for pago in bono['pagos']:
        mem_pagos.append([datetime.strptime(pago[0], "%d/%m/%y").date(), pago[1], pago[2]])
    total_dates = (mem_pagos[-1][0]-init_date).days
    dates_index = np.asarray(list(range(total_dates)))

    if term == 'flat':
        curve = r * np.ones(dates_index.shape[0]) 
    elif term == 'inv':
        curve = r * (np.exp(-dates_index/(2*N_DAYS)))
    else:
        curve = r * (1 - np.exp(-dates_index/(2*N_DAYS)))

    plt.plot(curve)
    plt.show()
    valores = []
    for i in range(total_dates):
        new_date = init_date + timedelta(days=i)
        valor_dia = 0
        for j, mp in enumerate(mem_pagos):
            if new_date < mp[0]:
                ttm_dates = (mp[0] - new_date).days
                r_curve = curve[ttm_dates-1] + np.random.normal(0, 0.0005)
                ttm_years = ttm_dates/N_DAYS
                valor = (mp[1]+ mp[2]) * np.exp(-r_curve*ttm_years)
            else:
                valor = 0
            valor_dia += valor
        valores.append(valor_dia)
    return valores

val_per_dia = calc_hist_price(bono)
plt.grid()
plt.plot(val_per_dia)
plt.axhline(94)
plt.axhline(106)
plt.show()

exit()

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
print(3*30*24*60)

plt.plot(rs, vs)
plt.plot(rs, Fit.monoExp(rs, m, t, b), 'k')
plt.plot(rs, modelPoly(rs), color='purple')
plt.show()

r1 = .2
r2 =  r1 + 0.0001 
Dr = r2 - r1
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
    r1 = .001
    r2 =  r1 + 0.0001 
    Dr = r2 - r1
    p1 = np.interp(r1, rs, vs)
    dp3 = np.interp(r2, rs, vs) - p1
    datapc.append((dp3/p1))
    # plt.plot(np.log(rs), np.log(vs))
    plt.plot(rs, vs)
plt.show()

plt.plot(x, datapc)
plt.show()
