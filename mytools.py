import pandas as pd
import numpy as np

def sma(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean().to_numpy()

def ema(signal, alpha):
    """
    Exponential moving average

    signal : 1D np.array
    alpha : 0.0 < smoothing factor < 1.0
    """

    return pd.Series(signal).ewm(alpha=alpha, adjust=False).mean().to_numpy()

def rsi(price, lookback=14):
    """
    Relative strenght index

    price : 1D np.array
    lookback : number of indices in EMA smoothing window
    """
    
    # lenght of price signal
    N = len(price)
    
    # change in price
    change = price[1:] - price[:-1]
    
    # gain in price
    idx = change > 0.0
    gain = np.zeros(N)
    gain[1:][idx] = change[idx]

    # loss in price
    idx = change < 0.0
    loss = np.ones(N)
    loss[1:][idx] = -change[idx]

    # relative strength: ratio between EMA smoothed gain and loss
    alpha = 1.0 / lookback
    rs = ema(gain, alpha) / ema(loss, alpha)

    # relative strength index
    rsi = 100.0 - 100.0 / (1.0 + rs)

    return rsi
