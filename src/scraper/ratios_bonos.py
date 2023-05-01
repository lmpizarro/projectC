from rava import scrap_bonos_rava, coti_hist
import pandas as pd
import matplotlib.pyplot as plt


def ratio_bond_usd(year: str = 29):
    hist_gd = coti_hist(scrap_bonos_rava(f"gd{year}"))
    key_bond = f"al{year}" if year != 38 else f"ae{year}"
    hist_al = coti_hist(scrap_bonos_rava(key_bond))

    cierre_gd = hist_gd[["fecha", "cierre", "usd_cierre"]]
    cierre_al = hist_al[["fecha", "cierre", "usd_cierre"]]
    mrg = pd.merge(
        cierre_al, cierre_gd, on="fecha", suffixes=(f"_al{year}", f"_gd{year}")
    )
    mrg[f"usd_al{year}"] = mrg[f"cierre_al{year}"] / mrg[f"usd_cierre_al{year}"]
    mrg[f"usd_gd{year}"] = mrg[f"cierre_gd{year}"] / mrg[f"usd_cierre_gd{year}"]
    mrg[f"ratio_{year}"] = mrg[f"cierre_gd{year}"] / mrg[f"cierre_al{year}"]
    mrg[f"ratio_usd_{year}"] = mrg[f"usd_cierre_gd{year}"] / mrg[f"usd_cierre_al{year}"]

    mrg[f"ewm_ratio_{year}"] = mrg[f"ratio_{year}"].ewm(alpha=0.1).mean()
    mrg[f"z_signal_{year}"] = mrg[f"ratio_{year}"] - mrg[f"ewm_ratio_{year}"]
    mrg[f"ewm_z_signal_{year}"] = mrg[f"z_signal_{year}"].ewm(alpha=0.25).mean()

    print(f"ratio {year} mean", mrg[f"ratio_{year}"].mean())
    st_dev = mrg[f"z_signal_{year}"].std()
    print(f"ratio last ", mrg.iloc[-1][f"ratio_{year}"])

    plt.title(f"g_{year}:al_{year}")
    plt.plot(mrg[f"z_signal_{year}"])
    plt.plot(mrg[f"ewm_z_signal_{year}"])

    plt.axhline(y=0.0, color="k", linestyle="-")
    plt.axhline(y=1.5 * st_dev, color="y", linestyle="-")
    plt.axhline(y=st_dev, color="r", linestyle="-")
    plt.axhline(y=-st_dev, color="g", linestyle="-")
    plt.axhline(y=-1.5 * st_dev, color="y", linestyle="-")
    plt.show()


def ratios_bonos_dolar():
    for y in [29, 30, 35, 38, 41]:
        ratio_bond_usd(y)


if __name__ == "__main__":
    ratios_bonos_dolar()
