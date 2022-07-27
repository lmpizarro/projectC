import scipy.signal as signal
from scipy.signal import hilbert
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy
import numpy as np
import yfinance as yf

'''
    ff = scipy.fft.fft((np.asarray(df.DIFCCL))[128:256])
    plt.plot(np.abs(ff)[2:64])
'''

def filter(symbols, df, key='Adj Close'):
    fs = 250

    for key in symbols:
        sos_fast = signal.butter(1, 4, 'lp', fs=fs, output='sos')
        sos_slow = signal.butter(1, 1, 'lp', fs=fs, output='sos')
        sos_bp = signal.butter(1, [8,12], 'bandpass', fs=250, output='sos')
        df['fast_ccl'] = signal.sosfiltfilt(sos_fast, df[key])
        df['slow_ccl'] = signal.sosfiltfilt(sos_slow, df[key])
        df['bp_ccl'] = signal.sosfiltfilt(sos_bp, df[key])
        df['bp_diff'] = df['bp_ccl'].diff(1)

        max_bp = df.bp_ccl.max()
        max_dbp = df.bp_diff.max()
        df.bp_diff = max_bp * df.bp_diff / max_dbp

        an_sig = hilbert(df['bp_ccl'])
        df['envelope'] = np.abs(an_sig)
        df['phase'] = np.unwrap(np.angle(an_sig))
        df['frequency'] = df['phase'].shift() / ((2.0*np.pi) * fs)

        X = np.asarray(range(0, len(df.phase))).reshape(-1, 1)
        reg = LinearRegression().fit(X, df['phase'])
        pred = reg.predict(X)
        df.phase = df.phase - pred

        df.dropna(inplace=True)

        plt.plot(df[key])
        plt.plot(df['slow_ccl'])
        plt.plot(df.bp_ccl + df['slow_ccl'], 'k')
        plt.show()

        plt.grid()
        plt.plot(df.bp_ccl)
        plt.plot(df.bp_diff, 'r')
        #plt.plot(10*np.diff(np.diff(np.asarray(df.bp_ccl))))
        plt.plot(df.envelope, 'g')
        plt.plot(-df.envelope, 'g')
        plt.show()

        # plt.grid()
        # plt.plot(8*df.envelope + df.slow_ccl)
        # plt.plot(df.slow_ccl - 8*df.envelope  )
        # plt.plot(df.slow_ccl)
        # plt.plot(df[key])

        # plt.plot(df.bp_ccl +df.slow_ccl, 'k')
        # plt.plot(df.bp2_ccl+ df.slow_ccl)
        # plt.plot(df[key])
        # plt.plot(df.slow_ccl)
        # plt.show()

        ff = scipy.fft.fft((np.asarray(df.bp_ccl)))
        plt.plot(np.abs(ff)[:250])
        plt.show()
    return df

if __name__ == '__main__':
    symbols = ['KO', 'PG', 'PEP']

    df = yf.download(symbols, '2015-2-1', '2022-03-07')['Adj Close']

    df = filter(symbols, df)