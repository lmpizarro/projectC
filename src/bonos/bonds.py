import numpy as np
"""
https://www.codearmo.com/python-tutorial/Python-TVM
https://bond-pricing.readthedocs.io/en/latest/
internal rate of return bonds python package
https://towardsdatascience.com/how-to-perform-bond-valuation-with-python-bbd0cf77417
"""

def npv(df, ytm):
    pv = 0
    for index, row in df.iterrows():
        pv +=  row['TOTAL']/np.power(1+ytm, row['T'])
    return pv

def m_duration(df, ytm):
    def npv_time(df, ytm):
        pv = 0
        for index, row in df.iterrows():
            pv +=  row['T'] * row['TOTAL']/np.power(1+ytm, row['T'])
        return pv
    return npv_time(df, ytm) / npv(df, ytm)

def ytm_discrete(df, value):
    ytm0 = 0.00
    ytmf = 1.000
    ytms = np.linspace(ytm0, ytmf, 1000)
    npvs = npv(df, ytms)
    # calculate the difference array
    difference_array = np.absolute(npvs - value)

    index = difference_array.argmin()

    return ytms[index]


def ytm_continuous(value, cash_flows, deltatimes, steps=1000):
    
    rates = np.linspace(0.0001, 2., steps)
    npvs = np.zeros(rates.shape[0])

    def npv(r, cash, times):
        v = 0
        for i, c in enumerate(cash):
            v += c*np.exp(-r*times[i])
        return v

    for i, r in enumerate(rates):
        npvs[i] = npv(r, cash_flows, deltatimes)
        
    difference_array = np.absolute(npvs - value)

    index = difference_array.argmin()

    return rates[index]

