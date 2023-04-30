from scrapers import scrap_bonos_rava
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime



def cash_flow(flujo, laminas: int = 1000):
    today = datetime.now()
    flujo.fillna(0, inplace=True)
    flujo['fecha'] = pd.to_datetime(flujo['fecha'],format= '%Y-%m-%d')
    flujo['acumulado'] = flujo.cupon.cumsum()
    flujo_inicial = -flujo.acumulado.iloc[0]
    flujo['cupon_precio'] = flujo.cupon / flujo_inicial
    flujo['acumu_precio'] = flujo.acumulado / flujo_inicial
    flujo['dias_cupon'] = (flujo.fecha - today).dt.days

    return flujo

def test():
    res = scrap_bonos_rava('gd29')

    coti_hist = pd.DataFrame(res['coti_hist'])

    print(coti_hist.head())
    flujo = pd.DataFrame(res['flujofondos']['flujofondos'])

    dolar = (res['flujofondos']['dolar'])
    tir = (res['flujofondos']['tir'])
    duration = (res['flujofondos']['duration'])

    cash_flow(flujo)
    print(res.keys())

    print(res['cotizaciones'][0])

    print(res['cuad_tecnico'])

def coti_hist(res):
    return pd.DataFrame(res['coti_hist'])

def ratio_bond_usd(year: str = 29):
    hist_gd = coti_hist(scrap_bonos_rava(f'gd{year}'))
    key_bond = f'al{year}' if year != 38 else f'ae{year}'
    hist_al = coti_hist(scrap_bonos_rava(key_bond))

    cierre_gd = hist_gd[['fecha', 'cierre', 'usd_cierre']]
    cierre_al = hist_al[['fecha', 'cierre', 'usd_cierre']]
    mrg = pd.merge(cierre_al, cierre_gd, on='fecha', suffixes=(f'_al{year}', f'_gd{year}'))
    mrg[f'usd_al{year}'] = mrg[f'cierre_al{year}']/mrg[f'usd_cierre_al{year}']
    mrg[f'usd_gd{year}'] = mrg[f'cierre_gd{year}']/mrg[f'usd_cierre_gd{year}']
    mrg[f'ratio_{year}'] = mrg[f'cierre_gd{year}']/mrg[f'cierre_al{year}']
    mrg[f'ratio_usd_{year}'] = mrg[f'usd_cierre_gd{year}']/mrg[f'usd_cierre_al{year}']

    mrg[f'ewm_ratio_{year}'] = mrg[f'ratio_{year}'].ewm(alpha=0.1).mean()
    mrg[f'z_signal_{year}'] = mrg[f'ratio_{year}'] - mrg[f'ewm_ratio_{year}']
    mrg[f'ewm_z_signal_{year}'] = mrg[f'z_signal_{year}'].ewm(alpha=0.25).mean()

    print(f"ratio {year} mean", mrg[f'ratio_{year}'].mean())
    st_dev = mrg[f'z_signal_{year}'].std()
    print(f"ratio last ", mrg.iloc[-1][f'ratio_{year}'])

    plt.title(f'g_{year}:al_{year}')
    plt.plot(mrg[f'z_signal_{year}'])
    plt.plot(mrg[f'ewm_z_signal_{year}'])

    plt.axhline(y=0.0, color='k', linestyle='-')
    plt.axhline(y=1.5*st_dev, color='y', linestyle='-')
    plt.axhline(y=st_dev, color='r', linestyle='-')
    plt.axhline(y=-st_dev, color='g', linestyle='-')
    plt.axhline(y=-1.5*st_dev, color='y', linestyle='-')
    plt.show()

def ratios_bonos_dolar():
    for y in [29, 30, 35, 38, 41]:
        ratio_bond_usd(y)

class Bono:

    def __init__(self, ticker: str = None, laminas: int = 100) -> None:
        self.ticker: str = ticker
        self.history: pd.DataFrame = None
        self.cash_flow: pd.DataFrame = None
        self.tir: float = None
        self.duration = None
        self.laminas:int = laminas
        self.precio: float = 0.0

    def dict(self):
        return self.__dict__

    def total(self):
        return self.laminas * self.precio

    def __str__(self) -> str:
        return (f'{self.ticker} tir {self.tir} dur {self.duration} precio {self.precio} lam {self.laminas} tot {self.total()}')

    def has_history(self):
        if 'usd_cierre' in self.history and \
            'cierre' in self.history and \
            not self.history.empty:

            return True
        return False

    def compound(self):
        filtered = self.cash_flow[1:-1]
        laminas = self.laminas

        composicion = []
        row = self.cash_flow.iloc[0]
        composicion.append({'fecha': row.fecha,
                        'laminas_adic': 0,
                        'pago': 0,
                        'laminas': self.laminas,
                        'acumulado': row.cupon * laminas
                        })
        for index, row in filtered.iterrows():
            laminas_adic = int(laminas * row.cupon / self.precio)
            composicion.append({'fecha': row.fecha,
                        'laminas_adic': laminas_adic,
                        'pago': laminas * row.cupon,
                        'laminas': laminas + laminas_adic,
                        'acumulado': -(laminas + laminas_adic) * self.precio})
            laminas += laminas_adic
        row = self.cash_flow.iloc[-1]
        composicion.append({'fecha': row.fecha,
                    'laminas_adic': 0,
                    'pago': row.cupon * laminas,
                    'laminas': laminas,
                    'acumulado': row.cupon * laminas
                    })

        composicion = pd.DataFrame(composicion)
        self.cash_flow[['cupon', 'renta', 'amortizacion']] = \
                    self.laminas * self.cash_flow[['cupon', 'renta', 'amortizacion']]
        self.cash_flow.acumulado = self.cash_flow.cupon.cumsum()
        mrg = pd.merge(self.cash_flow, composicion, on='fecha', suffixes=(f'_cf', f'_comp'))

        return mrg



    def invest(self, compound=False):
        if not compound:
            self.cash_flow[['cupon', 'renta', 'amortizacion']] = \
                        self.laminas * self.cash_flow[['cupon', 'renta', 'amortizacion']]
            self.cash_flow.acumulado = self.cash_flow.cupon.cumsum()
        else:
            self.compound()


def bono_fluxs(**data):

    bono = Bono(data['ticker'], data['laminas'])

    res = scrap_bonos_rava(data['ticker'])
    hist_gd = coti_hist(res)

    bono.history = hist_gd

    try:
        flujo = pd.DataFrame(res['flujofondos']['flujofondos'])
        if len(flujo) > 0:
            flujo = cash_flow(flujo)
            bono.cash_flow = flujo
            bono.precio = -1 * bono.cash_flow.iloc[0].cupon
    except:
        pass

    try:
        bono.tir = (res['flujofondos']['tir'])
    except:
        pass
    try:
        bono.duration = (res['flujofondos']['duration'])
    except:
        pass
    return bono

DRAW = False
def test_pesos():
    duales = ['TDJ23', 'TDL23', 'TDS23', 'TV23', 'TV24']
    txs =  ['T2X3', 'TX24', 'T2X4', 'TX26', 'TX28']
    en_pesos = [('CUAP', 34), ('DICP', 34), ('DIP0', 34), ('PARP', 340),]
    en_dolar = [('AL41', 31+14), ('AL29', 135+181+27), ('AE38', 38 + 22), ('AL30', 33 + 20)]
    tasa_var = ['BA37D', 'BDC24', 'BDC28', 'PBA25']
    tasa_vat = ['TO26', 'TO23']
    total = 0
    duration = 0
    for ticker in txs:
        print(ticker)
        bono = bono_fluxs(ticker=ticker[0], laminas=ticker[1])

        if DRAW:
            plt.title(bono.ticker)
            if bono.has_history():
                plt.plot(bono.history.usd_cierre)
            else:
                return
            plt.show()

        # print(bono.cash_flow)
        # bono.invest(compound=False)
        # bono.compound()
        total += bono.total()
        duration += bono.total() * bono.duration
        print(bono)
    print(total, duration / total)



# ratios_bonos_dolar()
test_pesos()

