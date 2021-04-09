import numpy as np
from typing import Any
from torch.utils.data import Dataset
# form src.dataset import create_bgs, awgn, normalization



fMin = 10.6 # GHz
fMax = 10.9 # GHz

LEN = 151

array = np.ndarray

"""

"""

class BGSDynamicDataSet(Dataset):
    """
    运行时模拟生成数据, 每次都不一样
    """

    def __init__(self, size=100, n=10, name="bfs"):
        """
        :param size: 模拟的数据总量
        :param N : 一个数据由N个BGS组成
        """
        self.n = n
        self.size = size
        self.name = name

    def __getitem__(self, item):
        """
        :return: (Tensor[1, 151, N], Tensor[N])
        """
        bgs, bfs, sw = create_data(self.n)
        return bgs, bfs

    def __len__(self):
        return self.size


def create_data(name: str):
    
    bgs = np.zeros((n, LEN), dtype=float)
    peak_gain = 1
    bfs, sw, snr = get_param(name=name)
    # bfs, sw, snr have the same length
    for i in range(len(bfs));
        bgs[i] = create_bgs(
            peak_gain=_peak_gain,
            sw=sw[i],
            bfs=bfs[i]
        )

        bgs[i] = awgn(data=bgs[i], snr=_snr)
        bgs[i] = bgs[i] / np.max(bgs[i])
    
    bgs = bgs.T
    bfs, sw = normalization(bfs=bfs, sw=sw)
    return torch.tensor([bgs], dtype=torch.float), \
           torch.tensor(bfs, dtype=torch.float), \
           torch.tensor(sw, dtype=torch.float)


def get_param(name: str):
    if name == 'bfs':
        return get_bfs(), get_array(0.25, 17), get_array(11, 17)
    elif name == 'sw':
        return get_array(0.25, 9), get_sw(), get_array(11, 9)
    elif name == 'snr':
        return get_array(0.3, 8), get_array(0.25, 8), get_snr()
    else:
        return None

def get_bfs():
    return fMin + (fMax - fMin) * np.array([(x * 5 + 10) / 100 for x in range(17)])

def get_sw():
    return np.array([(x * 5 + 10) / 100 for x in range(9)])

def get_snr():
    return np.array([x * 2 + 5 for x in range(8)])

def get_array(value: float, num: int) -> array:
    return np.array([value for i in range(num)])
