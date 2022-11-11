from datetime import datetime, timedelta, date
import numpy as np

def create_bullet_bond(face: float=100, years: float=10, pays_per_year: int=2,
                       rate: float=.05, first_pay: str='09/01/23'):

    pagos = []
    pay_date = datetime.strptime(first_pay, "%d/%m/%y").date()

    p_rate = rate / pays_per_year
    for i in range(years-1):
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


def create_amortizable_bond(years=5, pays_per_year=2, rate: float=.05, first_pay: str='09/01/23', n_amort: int=3, amortizacion=20):
    periods = pays_per_year * 2

    p_rate = rate / pays_per_year

    inicia_amort = years * pays_per_year - n_amort + 1
    coef_desc = np.power((1+rate/pays_per_year), -(np.arange(pays_per_year * years) + 1))

    pago_tasa = 100 * p_rate * np.ones(pays_per_year * years) 
    pago_amortizacion_zeros = np.zeros(inicia_amort - 1)

    # amortizacion = 100 / (pays_per_year * years - inicia_amort + 2)
    pago_amortizacion = amortizacion * np.ones(pays_per_year * years - inicia_amort + 1)

    print(pago_tasa.shape, pago_amortizacion_zeros.shape, pago_amortizacion.shape)
    pago_amortizacion = np.concatenate([pago_amortizacion_zeros, pago_amortizacion])
    pago_total = pago_tasa + pago_amortizacion
    print(pago_tasa, pago_total, pago_amortizacion, pago_total.sum())
  
    sum_menos_uno_amortizacion = (coef_desc * pago_amortizacion)[:-1].sum()
    sum_tasa_total = (coef_desc * pago_tasa).sum()
   
    resto_100 = 100 - (sum_menos_uno_amortizacion + sum_tasa_total)
    pago_amortizacion[-1] = (resto_100 / coef_desc[-1]).round(2)
    print((coef_desc * (pago_tasa + pago_amortizacion)).sum())

    pagos = []
    pay_date = datetime.strptime(first_pay, "%d/%m/%y").date()
    k = 0
    for i in range(years):
        for j in range(pays_per_year):
            pagos.append((pay_date.strftime("%d/%m/%y"), pago_tasa[k], pago_amortizacion[k]))
            pay_date = pay_date + timedelta(days=180)
            k += 1
 
    bono = {}
    bono['pagos'] = pagos

    return bono

def main():
    bono = create_bullet_bond(years=4)
    bono = create_amortizable_bond(years=4)

if __name__ == '__main__':
    main()

