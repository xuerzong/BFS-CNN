import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

"""
parameters: 
rhp - spectral noise density unit/SQRT(Hz)
sr  - sample rate
n   - no of points
mu  - mean value, optional

returns:
n points of noise signal with spectral noise density of rho
"""
def white_noise(rho, sr, n, mu=0):
    sigma = rho * np.sqrt(sr/2)
    noise = np.random.normal(mu, sigma, n)
    return noise

if __name__ == '__main__':
    a = np.random.uniform(0.05, 0.95)
    b = np.round(a, 2)
    print(a)