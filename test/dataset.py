import numpy as np
from typing import Any
from torch.utils.data import Dataset
from src.dataset import awgn, normalization, create_bgs

import matplotlib.pyplot as plt



fMin = 10.6 # GHz
fMax = 10.9 # GHz

LEN = 151

array = np.ndarray


def create_data(name: str, n: int):

    peak_gain = 1
    bfs, sw, snr = get_param(name=name)

    bgs = np.zeros((len(bfs) * n, LEN), dtype=np.float64)

    # bfs, sw, snr have the same length
    for x in range(len(bfs)):

        _range = np.arange(x * n, (x + 1) * n)
        
        for y in range(n):
            tmp = create_bgs(
                peak_gain=peak_gain,
                sw=sw[x],
                bfs=bfs[x]
            )

            tmp = awgn(data=tmp, snr=snr[x])
            tmp = tmp / np.max(tmp)
            bgs[_range[y]] = tmp 

    bfs, sw = normalization(bfs=bfs, sw=sw)
    return bgs, bfs, sw


def get_param(name: str):
    if name == 'bfs':
        return get_bfs(), (fMax - fMin) * get_array(0.25, 17), get_array(11, 17)
    elif name == 'sw':
        return fMin + (fMax - fMin) * get_array(0.25, 9), get_sw(), get_array(11, 9)
    elif name == 'snr':
        return fMin + (fMax - fMin) * get_array(0.3, 8), (fMax - fMin) * get_array(0.25, 8), get_snr()
    else:
        return None

def get_bfs():
    return fMin + (fMax - fMin) * np.array([(x * 5 + 10) / 100 for x in range(17)])

def get_sw():
    return (fMax - fMin) * np.array([(x * 5 + 10) / 100 for x in range(9)])

def get_snr():
    return np.array([x * 2 + 5 for x in range(8)])

def get_array(value: float, num: int) -> array:
    return np.array([value for i in range(num)])