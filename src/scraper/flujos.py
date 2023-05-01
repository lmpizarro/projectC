import matplotlib.pyplot as plt
import pandas as pd
from rava import cash_flow, scrap_bonos_rava, coti_hist
from collections import namedtuple

Investment = namedtuple("Investment", "ticker quantity")

DRAW = False


class Bono:
    def __init__(self, ticker: str = None, quantity: int = 100) -> None:
        self.ticker: str = ticker
        self.quantity: int = quantity
        self.history: pd.DataFrame = None
        self.cash_flow: pd.DataFrame = None
        self.tir: float = None
        self.duration = None
        self.precio: float = 0.0

    def dict(self):
        return self.__dict__

    def total(self):
        return self.quantity * self.precio

    def __str__(self) -> str:
        return f"{self.ticker} tir {self.tir} dur {self.duration} precio {self.precio} quantity {self.quantity} tot {self.total()}"

    def summary(self) -> dict:
        return {
            "ticker": self.ticker,
            "tir": self.tir,
            "price": self.precio,
            "quantity": self.quantity,
            "total": self.total(),
            "duration": self.duration,
            "maturity": self.cash_flow.iloc[-1].years_to_coupon,
        }

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
        laminas = self.quantity

        composicion = []
        row = self.cash_flow.iloc[0]
        composicion.append(
            {
                "fecha": row.fecha,
                "laminas_adic": 0,
                "pago": 0,
                "laminas": self.quantity,
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
            self.quantity * self.cash_flow[["cupon", "renta", "amortizacion"]]
        )
        self.cash_flow.acumulado = self.cash_flow.cupon.cumsum()
        mrg = pd.merge(
            self.cash_flow, composicion, on="fecha", suffixes=(f"_cf", f"_comp")
        )

        return mrg

    def invest(self, compound=False):
        if not compound:
            self.cash_flow[["cupon", "renta", "amortizacion"]] = (
                self.quantity * self.cash_flow[["cupon", "renta", "amortizacion"]]
            )
            self.cash_flow.acumulado = self.cash_flow.cupon.cumsum()
        else:
            self.compound()


def bond_cash_flow(invest: Investment):
    res = scrap_bonos_rava(invest.ticker)
    hist_gd = coti_hist(res)

    bono = Bono(invest.ticker, invest.quantity)
    bono.history = hist_gd

    try:
        flujo = pd.DataFrame(res["flujofondos"]["flujofondos"])
        if len(flujo) > 0:
            flujo = cash_flow(flujo)
            bono.cash_flow = flujo
            bono.precio = -1 * bono.cash_flow.iloc[0].cupon
            exit()
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
        Investment("CUAP", 100),
        Investment("DICP", 100),
        Investment("DIP0", 100),
        Investment("PARP", 100),
    ]
    en_dolar = [
        Investment("al29d", 1),
        Investment("gd29d", 1),
    ]

    summary = []
    for investment in en_dolar:
        bono = bond_cash_flow(invest=investment)

        if DRAW:
            plt.title(bono.ticker)
            if bono.has_history():
                plt.plot(bono.history.usd_cierre)
            else:
                return
            plt.show()
        bono.invest(compound=False)
        bono.compound()

        print(bono.cash_flow)
        print(investment)
        summary.append(bono.summary())
    summary = pd.DataFrame(summary)
    summary["dur_to_mat"] = summary["duration"] / summary["maturity"]
    summary["Duration"] = summary["duration"] * summary["total"]
    print(summary)
    print(summary.total.sum(), summary.Duration.sum() / summary.total.sum())


if __name__ == "__main__":
    test_pesos()
