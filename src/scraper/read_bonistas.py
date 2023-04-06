import pandas as pd

df = pd.read_csv("metricas_bonos.csv")
print(df.keys())
df['TIR'] = df['TIR'].str.replace("%", '')
df['TIR'] = pd.to_numeric(df['TIR'], downcast='float')

df_q = df[((df.TIPO == 'CER') | (df.TIPO == 'LECER')) & ((df.TIR > 0) & (df.TIR < 100))][['TIPO', 'Ticker', 'TIR', 'MD', 'Precio', 'ratMD', 'MAT']].sort_values(by=['MD'])

import matplotlib.pyplot as plt

print(df_q)

ax = df_q.plot(kind='scatter', x='MAT', y='MD')

#label each point in scatter plot
for idx, row in df_q.iterrows():
    y = 1 if idx % 2 else -1
    ax.annotate(row['Ticker'], (row['MAT'], row['MD']), xytext=(y*5,y*10),textcoords='offset points', fontsize=6)

plt.show()