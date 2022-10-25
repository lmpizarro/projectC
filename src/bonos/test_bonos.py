import numpy as np
from bonds import ytm_continuous

def test_ytm_continuous():        
    cash_f = np.array([.03] * 9)
    cash_f = np.append(cash_f, 1.03)

    tms = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    print(ytm_continuous(.96, cash_f, tms))

test_ytm_continuous()

