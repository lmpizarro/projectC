import matplotlib.pyplot as plt
import pandas as pd
from rava import cash_flow, scrap_bonos_rava, coti_hist
from collections import namedtuple

Investment = namedtuple('Investment', 'ticker quantity')

DRAW = False


class Bono:
    def __init__(self, ticker: str = None, laminas: int = 100) -> None:
        self.ticker: str = ticker
        self.history: pd.DataFrame = None
        self.cash_flow: pd.DataFrame = None
        self.tir: float = None
        self.duration = None
        self.laminas: int = laminas
        self.precio: float = 0.0

    def dict(self):
        return self.__dict__

    def total(self):
        return self.laminas * self.precio

    def __str__(self) -> str:
        return f"{self.ticker} tir {self.tir} dur {self.duration} precio {self.precio} lam {self.laminas} tot {self.total()}"

    def has_history(self):
        if (
            "usd_cierre" in self.history
            and "cierre" in self.history
            and not self.history.empty
        ):
            return True
        return False

    def compound(self):
        filtered = self.cash_flow[1:-1]
        laminas = self.laminas

        composicion = []
        row = self.cash_flow.iloc[0]
        composicion.append(
            {
                "fecha": row.fecha,
                "laminas_adic": 0,
                "pago": 0,
                "laminas": self.laminas,
                "acumulado": row.cupon * laminas,
            }
        )
        for index, row in filtered.iterrows():
            laminas_adic = int(laminas * row.cupon / self.precio)
            composicion.append(
                {
                    "fecha": row.fecha,
                    "laminas_adic": laminas_adic,
                    "pago": laminas * row.cupon,
                    "laminas": laminas + laminas_adic,
                    "acumulado": -(laminas + laminas_adic) * self.precio,
                }
            )
            laminas += laminas_adic
        row = self.cash_flow.iloc[-1]
        composicion.append(
            {
                "fecha": row.fecha,
                "laminas_adic": 0,
                "pago": row.cupon * laminas,
                "laminas": laminas,
                "acumulado": row.cupon * laminas,
            }
        )

        composicion = pd.DataFrame(composicion)
        self.cash_flow[["cupon", "renta", "amortizacion"]] = (
            self.laminas * self.cash_flow[["cupon", "renta", "amortizacion"]]
        )
        self.cash_flow.acumulado = self.cash_flow.cupon.cumsum()
        mrg = pd.merge(
            self.cash_flow, composicion, on="fecha", suffixes=(f"_cf", f"_comp")
        )

        return mrg

    def invest(self, compound=False):
        if not compound:
            self.cash_flow[["cupon", "renta", "amortizacion"]] = (
                self.laminas * self.cash_flow[["cupon", "renta", "amortizacion"]]
            )
            self.cash_flow.acumulado = self.cash_flow.cupon.cumsum()
        else:
            self.compound()


def bono_fluxs(invest: Investment):
    bono = Bono(invest.ticker, invest.quantity)

    res = scrap_bonos_rava(invest.ticker)
    hist_gd = coti_hist(res)

    bono.history = hist_gd

    try:
        flujo = pd.DataFrame(res["flujofondos"]["flujofondos"])
        if len(flujo) > 0:
            flujo = cash_flow(flujo)
            bono.cash_flow = flujo
            bono.precio = -1 * bono.cash_flow.iloc[0].cupon
    except:
        pass

    try:
        bono.tir = res["flujofondos"]["tir"]
    except:
        pass
    try:
        bono.duration = res["flujofondos"]["duration"]
    except:
        pass
    return bono


DRAW = False
"""
    duales = ["TDJ23", "TDL23", "TDS23", "TV23", "TV24"]
    txs = ["T2X3", "TX24", "T2X4", "TX26", "TX28"]
    tasa_var = ["BA37D", "BDC24", "BDC28", "PBA25"]
    tasa_vat = ["TO26", "TO23"]
"""

def test_pesos():
    en_pesos = [
        Investment("CUAP", 34),
        Investment("DICP", 34),
        Investment("DIP0", 34),
        Investment("PARP", 340),
    ]
    en_dolar = [
        Investment("AL41", 45),
        Investment("AL29", 343),
        Investment("AE38", 60),
        Investment("AL30", 53),
    ]
    total = 0
    duration = 0
    for investment in en_dolar:
        print(investment)
        bono = bono_fluxs(invest=investment)

        if DRAW:
            plt.title(bono.ticker)
            if bono.has_history():
                plt.plot(bono.history.usd_cierre)
            else:
                return
            plt.show()

        print(bono.cash_flow)
        bono.invest(compound=False)
        bono.compound()
        total += bono.total()
        duration += bono.total() * bono.duration
        print(bono)
    print(total, duration / total)


if __name__ == "__main__":
    test_pesos()
