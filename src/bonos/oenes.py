import numpy as np
import pandas as pd

import datetime
from scipy import optimize 
import warnings
warnings.simplefilter("ignore")


def xnpv(rate,cashflows):
    chron_order = sorted(cashflows, key = lambda x: x[0])
    t0 = chron_order[0][0] #t0 is the date of the first cash flow

    return sum([cf/(1+rate)**((t-t0).days/365.0) for (t,cf) in chron_order])

def xirr(cashflows,guess=0.1):

    return optimize.newton(lambda r: xnpv(r,cashflows),guess)*100


def tir(cashflow, precio, plazo=1):
    flujo_total=[(datetime.datetime.today()+ datetime.timedelta(days=plazo) , -precio)]
    for i in range (len(cashflow)):
      if cashflow.iloc[i,0].to_pydatetime()>datetime.datetime.today()+ datetime.timedelta(days=plazo):
        flujo_total.append((cashflow.iloc[i,0].to_pydatetime(),cashflow.iloc[i,1]))
    
    return round(xirr(flujo_total,guess=0.1),2)

def duration(cashflow,precio,plazo=2):
  r=tir(cashflow,precio,plazo=plazo)
  denom=[]
  numer=[]
  for i in range (len(cashflow)):
    if cashflow.iloc[i,0].to_pydatetime()>datetime.datetime.today()+ datetime.timedelta(days=plazo):
      tiempo=(cashflow.iloc[i,0]-datetime.datetime.today()).days/365 #tiempo al cupon en años
      cupon=cashflow.iloc[i,1]
      denom.append(cupon/(1+r/100)**tiempo) #sum (C(1+r)^t)
      numer.append(tiempo*(cupon/(1+r/100)**tiempo))

  return round(sum(numer)/sum(denom),2)

def modified_duration(cashflow,precio,plazo=2):
  dur=duration(cashflow,precio,plazo)
  return round(dur/(1+tir(cashflow,precio,plazo)/100),2)
  

url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones%20negociables"
ON = pd.read_html(url)[0]

precios_ON=pd.DataFrame()
precios_ON['ÚltimoOperado']=ON['ÚltimoOperado'].str.replace('.','', regex=True).str.replace(',','.', regex=True).astype(float)
try:
  precios_ON['MontoOperado']=ON['MontoOperado'].str.replace('.','', regex=True).str.replace(',','.', regex=True).astype(float)
except:
  precios_ON['MontoOperado']=0
  
precios_ON['Ticker']=ON['Símbolo']
precios_ON=precios_ON[['Ticker','ÚltimoOperado','MontoOperado']]
precios_ON.set_index('Ticker',inplace=True)

Data_ON=tickers=pd.read_excel('cashflows_ON.xlsx', sheet_name='Data_ON', engine="openpyxl" )

comunes=list(set(precios_ON.index) & set(Data_ON['ticker_dolares']))

comunes.sort()

Data_ON=Data_ON.loc[Data_ON['ticker_dolares'].isin(comunes)].sort_values(by=['ticker_dolares'])

Data_ON['Precio_dolares']=list(precios_ON.loc[comunes]['ÚltimoOperado']/100)

Data_ON['Volumen']=list(precios_ON.loc[comunes]['MontoOperado'])

Data_ON['Precio_pesos']=0

# Data_ON['Volumen']=0

try:
  Data_ON['Precio_pesos']=list(precios_ON.loc[Data_ON['ticker_pesos']]['ÚltimoOperado'])
  Data_ON['Volumen']=list(precios_ON.loc[Data_ON['ticker_pesos']]['MontoOperado'])
except:
  for ticker in Data_ON['ticker_pesos']:
    if ticker in precios_ON.index:
      Data_ON.loc[Data_ON['ticker_pesos']==ticker,'Precio_pesos']=precios_ON.loc[ticker]['ÚltimoOperado']
      # Data_ON.loc[Data_ON['ticker_pesos']==ticker,'Volumen']=precios_ON.loc[ticker]['MontoOperado'] #monto operado en pesos

Data_ON.loc[Data_ON['ticker_pesos']=='CAC2O','Precio_pesos'] = 8

Data_ON['Amortizacion']=Data_ON['Amortizacion'].replace(np.nan, 'No Bullet', regex=True)

Data_ON.set_index(['ticker_pesos'], inplace=True)

tickers_ON = Data_ON.index


cashflows={}
for i in Data_ON.index:
  CF=pd.read_excel('cashflows_ON.xlsx', sheet_name=i, engine="openpyxl")
  cashflows[i]=CF

print(cashflows['IRCFO'])

for ticker in tickers_ON:
    try:
        p = Data_ON['Precio_dolares'][ticker]

    except:
        p = 'na'

    empresa = Data_ON['Empresa'][ticker].split()[0]
    if p != 'na':
        try:
            m_dur = modified_duration(cashflows[ticker],68.55,plazo=0)
        except RuntimeError as err:
            m_dur = 'na'

        try:
            tir__ = tir(cashflows[ticker], p, 2)
        except RuntimeError as err:
            tir__ = 'na'

        if tir__!= 'na' and tir__  > 0:
            print(ticker, tir__, m_dur, p, empresa)    
            

Data_ON['Emp'] = Data_ON['Empresa'].apply(lambda x: x.split()[0])
TIR={}
for j in Data_ON.index:
  try:
    tir_j=tir(cashflows[j],Data_ON.loc[j,'Precio_dolares'],plazo=1)
    TIR[j]=(modified_duration(cashflows[j],Data_ON.loc[j,'Precio_dolares'],plazo=1),tir_j)
    #(round((duration(cashflows[j],Data_ON.loc[j,'Precio_dolares'],plazo=1)/(1+tir_j/100)),2),tir_j)
  except:
    pass


TIR=pd.DataFrame.from_dict(TIR,orient='index')
TIR.columns=['MD','TIR']

Data_ON['MD']=TIR['MD']
Data_ON['TIR']=TIR['TIR']

tir_pos = Data_ON[(Data_ON.TIR > 6) ]
tir_pos['ratio'] = round(tir_pos['Precio_pesos'] / tir_pos['Precio_dolares'],2)
print(Data_ON.columns)
keys = ['TIR', 'MD', 'Precio_dolares', 'Vencimiento', 'Emp', 'Ley', 'Pago', 'lamina_minima', 'frecuencia_Pagos', 'ratio' ]
print(tir_pos[keys])


print(tir_pos[keys]['TIR'].mean())
invest_t =  0 
cf_t = 0
dates = set()
for ticker in tir_pos.index:
    # cond = (cashflows[ticker]['Fecha']>datetime.datetime.today())
    cf = cashflows[ticker][(cashflows[ticker]['Fecha'].dt.year == 2023)].Cupon.sum()
    fechas = cashflows[ticker][(cashflows[ticker]['Fecha'].dt.year == 2023)].Fecha.dt.date
    invest = tir_pos.loc[ticker]['Precio_dolares']
    if tir_pos.loc[ticker]['lamina_minima'] == 1000:
        invest = 10 * invest
        cf = 10 * cf
    invest_t += invest
    cf_t += cf
    print(ticker, tir_pos.loc[ticker]['lamina_minima'], tir_pos.loc[ticker]['Precio_dolares'], round(cf,2), invest)
    dates.update(list(fechas))


cant_dias_pago = len(sorted(dates))
print(invest_t, cf_t, cf_t / invest_t, cf_t / 12, cf_t / cant_dias_pago, cf_t / 365, 316 * cf_t / 12)

