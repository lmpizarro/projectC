import pandas as pd

df = pd.read_csv("metricas_bonos.csv")
print(df.keys())
df['TIR'] = df['TIR'].str.replace("%", '')
df['TIR'] = pd.to_numeric(df['TIR'], downcast='float')

print(df[((df.TIPO == 'CER') | (df.TIPO == 'LECER')) & ((df.TIR > 0) & (df.TIR < 100))][['TIPO', 'Ticker', 'TIR', 'MD', 'Precio', 'ratMD']].sort_values(by=['MD']))