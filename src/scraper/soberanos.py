from datetime import datetime, timedelta, date
from pydantic import BaseModel
import numpy as np


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

class Bono(BaseModel):
    time_to_finish: float

def delta_time_years(date2: str):
    today = date.today()
    end_date = datetime.strptime(date2, "%d/%m/%y").date()
    time_to_finish = end_date - today
    time_to_finish = time_to_finish.total_seconds()/(3600*24*365)

    return time_to_finish


def get_nominals(k):

    total_amortizacion = 0
    total_renta = 0
    pagos = bonos[k]["pagos"]
    init_date = pagos[0][0]
    time_to_finish = delta_time_years(pagos[len(pagos)-1][0]) 

    pairs_time_pagos = []
    for pago in pagos:
        dia_pago = datetime.strptime(pago[0], "%d/%m/%y").date()
        time_to_pago = delta_time_years(pago[0])
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

def valor_bono_disc(pair_pagos, tasa):
    valor = 0
    for e in pair_pagos:
        v = e[1]/np.power(1+tasa/2, 2*e[0])
        # print(e, v)
        valor += v 
    return valor


for k in bonos:
    total_amortizacion, total_renta, _ = get_nominals(k)
    total_pago = total_amortizacion + total_renta

    print(k, round(total_amortizacion, 2), round(total_renta, 2), round(total_pago, 2))

ticker = 'GD30'
amortizacion, renta, pair_pagos = get_nominals(ticker)
renta_pct = renta / amortizacion

delta_r = 0.01
print(renta_pct-delta_r, valor_bono_disc(pair_pagos, renta_pct-delta_r))
print(renta_pct, valor_bono_disc(pair_pagos, renta_pct))
print(renta_pct+delta_r, valor_bono_disc(pair_pagos, renta_pct+delta_r))

rs = np.linspace(0.0001, 1.0, 100)
vs = np.zeros(100)
vs2 = np.zeros(100)

for i, r in enumerate(rs):
    vs[i] = valor_bono_disc(pair_pagos, r)
    vs2[i] = valor_bono_cont(pair_pagos, r)

import matplotlib.pyplot as plt
plt.plot(rs, vs)
plt.grid()
plt.axhline(y=21)
plt.axvline(x=renta_pct)
plt.show()

plt.plot(np.diff(vs))
plt.show()
plt.plot(np.diff(np.diff(vs)))
plt.show()
