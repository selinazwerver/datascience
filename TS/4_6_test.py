import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def calculate_fft(x, t):
    s =  np.ceil(x.size/2)
    y = (2/x.size)*fft(x)
    y = y[0:int(s)]
    ym = abs(y)
    f = np.arange(0,s)
    fspacing = 1/(t.size*(t[1] - t[0]))
    f = fspacing*f
    return f, ym

## Test code
fs = 20 # sample frequency
T = 1/fs # sample time
x = np.arange(0, 10, T)
signal = np.sin(2*2*np.pi*x) + 0.8*np.sin(4*2*np.pi*x) + 0.5*np.sin(6*2*np.pi*x)

f, ym = calculate_fft(signal, x)
plt.subplot(1,2,1)
plt.plot(x,signal) # original signal
plt.subplot(1,2,2)
plt.plot(f,ym) # fft
plt.show()