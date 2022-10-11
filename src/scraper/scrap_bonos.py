import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import numpy as np
import numpy_financial as npf
from datetime import date

"""
https://www.codearmo.com/python-tutorial/Python-TVM
https://bond-pricing.readthedocs.io/en/latest/
internal rate of return bonds python package
https://towardsdatascience.com/how-to-perform-bond-valuation-with-python-bbd0cf77417

"""
def tir(value, cash_flows, deltatimes, steps=100000):
    
    rates = np.linspace(0.0001, 2., steps)
    
    for r in rates:
        v = 0
        for i, c in enumerate(cash_flows):
            v += c*np.exp(-r*deltatimes[i])
        if np.abs(v - value) < 1e-2:

            return (v,r)
    return (None, None)

def test_tir():        
    cash_f = [.03] * 9
    cash_f.append(1.03)

    tms = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    print(tir(.96, cash_f, tms))

price = 23.90
N_DAYS = 360
ticker = 'AE38'

url = f"https://bonistas.com/md/{ticker}"

# Create object page
page = requests.get(url)


# parser-lxml = Change html to Python friendly format
# Obtain page's information
soup = BeautifulSoup(page.text, 'lxml')


div = soup.find('div', class_='col-lg-6 d-none d-lg-block d-xl-block')
table = div.find('table')

rows = table.findChildren(['th', 'tr'])

header = ["FECHA", "SALDO", "CUPÓN", "AMORTIZACIÓN", "TOTAL"]

csv = ','.join(header) + '\n'
for row in rows:
    cells = row.findChildren('td')
    line = ''
    for cell in cells:
        value = cell.string
        line += "%s," % value
    line = line[:-1]
    if line != '':
       csv += "%s\n" % line
csv += "\n"

csv_file = StringIO(csv)
df = pd.read_csv(csv_file)

today = [pd.Timestamp(date.today())] * len(df)
to_day = pd.Series(today)
df['FECHA'] = pd.to_datetime(df['FECHA'])
df['T'] = (df['FECHA'] - to_day)
df['T'] = df['T'].dt.days.astype('int16') / N_DAYS

def npv(df, ytm):
    pv = 0
    for index, row in df.iterrows():
        pv +=  row['TOTAL']/np.power(1+ytm, row['T'])
    return pv 

def calc_ytm_(df, value):
    ytm0 = 0.001
    ytmf = 2.000
    ytms = np.linspace(ytm0, ytmf, 100000)

    for j,r in enumerate(ytms):
        v = npv(df, r)
        if np.abs(value -v ) < 1e-2:
            return r

def calc_ytm(df, value):
    ytm0 = 0.00
    ytmf = 1.000
    ytms = np.linspace(ytm0, ytmf, 1000)
    npvs = npv(df, ytms)
    # calculate the difference array
    difference_array = np.absolute(npvs - value)

    index = difference_array.argmin()

    return ytms[index]

tir_ = calc_ytm(df, price)

print(tir_)



total_amor = df['AMORTIZACIÓN'].sum()
total_cupo = df['CUPÓN'].sum()

total = total_amor + total_cupo

print(total, total_amor, total_cupo)


t_total = df["T"].iloc[-1]

r_mean_t = -np.log(total_amor/total) / t_total

r_mean = -np.log(price/total) / t_total
print(r_mean_t, r_mean)


values = df.TOTAL.to_numpy()
values = np.insert(values, 0, -99.9)
irr0 = npf.irr(values)
values[0] = -price
print(irr0, npf.irr(values))

print(df.head())
print(df.tail())
print(tir(price, df['TOTAL'].to_numpy(), df['T'].to_numpy()))
