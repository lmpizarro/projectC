from datetime import datetime, timedelta, date
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
from scrap_bonos import DAYS_IN_A_YEAR


class Bono(BaseModel):
    time_to_finish: float

def delta_time_years(date2: str, date1):
    end_date = datetime.strptime(date2, "%d/%m/%y").date()
    time_to_finish = end_date - date1
    time_to_finish = time_to_finish.total_seconds()/(3600*24*DAYS_IN_A_YEAR)

    return time_to_finish


def get_nominals(bono, today):

    total_amortizacion = 0
    total_renta = 0
    pagos = bono["pagos"]
    init_date = pagos[0][0]

    time_to_finish = delta_time_years(pagos[len(pagos)-1][0], today) 

    pairs_time_pagos = []
    for pago in pagos:
        dia_pago = datetime.strptime(pago[0], "%d/%m/%y").date()
        if dia_pago < today:
            continue
        time_to_pago = delta_time_years(pago[0], today)
        renta = pago[1]
        amortizacion = pago[2]
        total_amortizacion += amortizacion
        total_renta += renta
        r_mas_a = renta + amortizacion

        pairs_time_pagos.append((time_to_pago, r_mas_a))

    return total_amortizacion, total_renta, pairs_time_pagos

def valor_bono_cont(pair_pagos, tasa):
    valor = 0
    for e in pair_pagos:
        valor += e[1]*np.exp(-tasa*e[0])
    return valor

def valor_bono_disc(pair_pagos, tasa, pagos_p_a=2):
    valor = 0
    for e in pair_pagos:
        v = e[1]/np.power(1+tasa/pagos_p_a, pagos_p_a*e[0])
        # print(e, v)
        valor += v 
    return valor

if __name__ == '__main__':

    with open('bonos.pkl', 'rb') as fp:
        bonos = pickle.load(fp)

    for k in bonos:

       today = date.today()
       total_amortizacion, total_renta, _ = get_nominals(bonos[k], today)
       total_pago = total_amortizacion + total_renta

       print(k, round(total_amortizacion, 2), round(total_renta, 2), round(total_pago, 2))

    ticker = 'GD30'
    amortizacion, renta, pair_pagos = get_nominals(bonos[ticker], today)
    renta_pct = renta / amortizacion

    delta_r = 0.01
    print(renta_pct-delta_r, valor_bono_disc(pair_pagos, renta_pct-delta_r))
    print(renta_pct, valor_bono_disc(pair_pagos, renta_pct))
    print(renta_pct+delta_r, valor_bono_disc(pair_pagos, renta_pct+delta_r))

    rs = np.linspace(0.0001, 1.0, 100)
    vs = np.zeros(100)
    vs2 = np.zeros(100)

    import matplotlib.pyplot as plt

    for ticker in bonos:
        amortizacion, renta, pair_pagos = get_nominals(bonos[ticker], today)
        for i, r in enumerate(rs):
            vs[i] = valor_bono_disc(pair_pagos, r)
            vs2[i] = valor_bono_cont(pair_pagos, r)

        plt.plot(rs, vs)
        plt.grid()
        plt.axhline(y=21)
        plt.axvline(x=renta_pct)

    plt.show()

    plt.plot(np.diff(vs))
    plt.show()
    plt.plot(np.diff(np.diff(vs)))
    plt.show()
