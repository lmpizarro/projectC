"""
Ticker 	Indice 	Precio 	Dif 	TIR 	MD 	Vol(M) 	Paridad 	VT 	tTIR 	UpTIR 	dQ 	pQ 	Q/P
S31M3 	Fijo 	95.10 	=0.00% 	114.7% 	0.03 	0.0 	95.1% 	100.0 	- 	- 	27 	- 	-
S28A3 	Fijo 	89.17 	=0.00% 	123.6% 	0.06 	0.0 	89.2% 	100.0 	- 	- 	55 	- 	-
S31Y3 	Fijo 	82.97 	=0.00% 	122.9% 	0.10 	0.0 	83.0% 	100.0 	- 	- 	88 	- 	-
S30J3 	Fijo 	77.75 	=0.00% 	122.3% 	0.14 	0.0 	77.8% 	100.0 	- 	- 	118 	- 	-
"""
import time
from datetime import datetime
import math

s31m3 = {"precio": -95.10, "finish": "2023-03-31"}
s28a3 = {"precio": -89.17, "finish": "2023-04-28"}
s31Y3 = {"precio": -82.97, "finish": "2023-05-31"}
s30J3 = {"precio": -77.75, "finish": "2023-06-30"}

letras = [s31m3, s28a3, s31Y3, s30J3]

def stringToDate(date: str):
    return datetime.date(datetime.strptime(date, "%Y-%m-%d"))

def Irr(letra: dict) ->float:
    return math.pow(-100 / letra["precio"], 1/ letra["T"]) - 1

def Rate(letra)->float:
    return  -(100 + letra["precio"]) / letra["precio"]

def Escentials(letra):
    begin = letra["Begin"]
    finish = stringToDate(letra["finish"])
    letra['T'] = (finish - begin).days / 365
    letra["Rate"] = Rate(letra)
    letra["Irr"] = Irr(letra)


def ForwardRate(letra1, letra2):
    r2 = 1 + letra2["Irr"] * letra2["T"]
    r1 = 1 + letra1["Irr"] * letra1["T"]
    den = letra2["T"] - letra1["T"]
    return (r2/r1 - 1) / den

def createPagos(inversionInicial: float, begin: str):
    begin = stringToDate(begin)

    pagos = {}

    pagos["T0"] = [-inversionInicial]
    Ci = inversionInicial / 4
    for i, letra in enumerate(letras):
        letra["Begin"] = begin
        Escentials(letra)
        pagos[f'T{i+1}'] = [Ci * (1 + letra["Rate"])]

    return pagos

def main():

    pagos = createPagos(100000, "2023-03-07")
    for i in range(1, len(letras)):
        keyT = f"T{i}"
        divisor = len(letras) - i
        C = sum(pagos[keyT]) / divisor
        print(keyT,  C )
        for j in range(i, len(letras)):
            k = f"T{j+1}"
            fr = ForwardRate(letras[i-1], letras[j])
            T = letras[j]["T"] - letras[j-1]["T"]
            Comp = C * math.exp(fr*T)
            pagos[k].append(Comp)
            print(k,fr, Comp, pagos[k])
        print()
    for k in pagos:
        print(k, pagos[k])
    Dp = sum(pagos["T4"]) + pagos["T0"][0]

    Dt = letras[len(letras)-1]["T"]

    irr = math.log(127000/100000) / Dt
    irr = math.pow(sum(pagos["T4"]) / 100000, 1/Dt) - 1
    print(irr)

    final = {"precio": 100*pagos["T0"][0]/sum(pagos["T4"]), "finish": "2023-06-30", "T": Dt}
    print(Irr(final))

if __name__ == "__main__":
    main()