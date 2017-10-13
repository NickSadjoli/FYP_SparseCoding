import numpy as np

def trendline_fit(x,y):
  z = np.polyfit(x, y, 3)
  p = np.poly1d(z)
  return p

def l2_norm(x): #assume x is a vector matrix
    return np.sqrt(np.sum(np.abs(x)**2))

def Recovery_Error(original_signal, test_signal):
    return (l2_norm(original_signal - test_signal)/l2_norm(original_signal))