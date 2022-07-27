import scipy.signal as signal
import numpy as np

def min_lp(symbols, df, zero=0.00001):
    fs = 250
    for s in symbols:
        key = s + '_deno'
        for i in range(1, fs//2 - 1):
            sos_smooth_close = signal.butter(1, i, 'lp', fs=fs, output='sos')
            sig = signal.sosfiltfilt(sos_smooth_close, df[s])
            if np.power((df[s] - sig).mean(), 2) < zero:
                break
        df[key] = sig
    return df


