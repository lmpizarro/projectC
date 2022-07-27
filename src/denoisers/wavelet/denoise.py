import numpy as np
import pywt

# https://www.kaggle.com/code/theoviel/denoising-with-direct-wavelet-transform
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(x, wavelet='coif4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

def deno_wvlt(symbols, df):
    for s in symbols:
        key = s + '_deno'
        sig = wavelet_denoising(df[s])
        if len(df) == len(sig):
            df[key] = sig
        elif len(df) > len(sig):
            df[key] = sig[1:]
    return df


