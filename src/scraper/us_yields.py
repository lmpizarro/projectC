import requests
from config import urls, url_treasury_by_year
from datetime import datetime
import numpy as np
import pandas as pd


class USBonds:
    def __init__(self, year: int = 2023) -> None:
        self.year = year
        self.yield_curve = self.treasury_yield_curve()

    def treasury_yield_curve(self):
        string_terms = [
            "1 Mo",
            "2 Mo",
            "3 Mo",
            "4 Mo",
            "6 Mo",
            "1 Yr",
            "2 Yr",
            "3 Yr",
            "5 Yr",
            "7 Yr",
            "10 Yr",
            "20 Yr",
            "30 Yr",
        ]

        resp = requests.get(
            url_treasury_by_year(year=self.year), headers={"User-Agent": "Mozilla/5.0"}
        )
        treas_df = pd.read_html(resp.text)

        filter_keys = ["Date"]
        filter_keys.extend(string_terms)
        treas_df = treas_df[0][filter_keys]

        def spliter(r: str):
            s = r.split(" ")
            if s[1] == "Mo":
                d = 30
            else:
                d = 365
            k = int(s[0])

            return d * k

        treas_df["Date"] = pd.to_datetime(treas_df["Date"], format="%m/%d/%Y").dt.date
        string_term_to_days_term = {r: spliter(r) for r in string_terms}
        days_term = string_term_to_days_term.values()
        treas_df.rename(columns=string_term_to_days_term, inplace=True)
        treas_df["mean"] = treas_df[days_term].mean(axis=1)
        treas_df.set_index("Date", inplace=True)

        return treas_df

    def today_mean(self):
        today_year = datetime.now().year
        if self.year != today_year:
            self.year = today_year
            self.yield_curve = self.treasury_yield_curve()
        return self.yield_curve["mean"].iloc[-1]

    def last_curve_points(self):
        terms = np.array(list(self.yield_curve.keys())[0:-1])
        rates = np.array(list(self.yield_curve.tail().iloc[-1])[0:-1])

        return terms, rates


if __name__ == "__main__":
    usbond = USBonds()
    df_treas = usbond.yield_curve
    print(df_treas.tail())
    print(usbond.today_mean())
    terms, rates = usbond.last_curve_points()

    daily_rates = rates / 365
    term_rates = daily_rates * terms

    print(terms)
    print(rates)

    import matplotlib.pyplot as plt

    plt.plot(df_treas["mean"])
    plt.show()

    plt.plot(terms, rates)
    plt.show()
