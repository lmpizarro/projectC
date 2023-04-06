import pandas as pd
from nelson_siegel_svensson.calibrate import calibrate_ns_ols


df = pd.read_csv("metricas_bonos.csv")
print(df.keys())
df['TIR'] = df['TIR'].str.replace("%", '')
df['TIR'] = pd.to_numeric(df['TIR'], downcast='float')

df_q = df[((df.TIPO == 'CER') | (df.TIPO == 'LECER')) & ((df.TIR > 0) & (df.TIR < 100))][['TIPO', 'Ticker', 'TIR', 'MD', 'Precio', 'ratMD', 'MAT']].sort_values(by=['MD'])

import matplotlib.pyplot as plt

print(df_q)
print(df_q.shape)
print(df_q.shape)
print(df_q['MD'].shape)
print(df_q['TIR'].shape)


curve, status = calibrate_ns_ols(df_q['MD'].to_numpy(), df_q['TIR'].to_numpy()/100, tau0=1.0)  # starting value of 1.0 for the optimization of tau
assert status.success

print(curve)
plt.plot(df_q['MD'], curve(df_q['MD']), 'o-')
plt.plot(df_q['MD'], curve.forward(df_q['MD']), 'o-')
plt.show()