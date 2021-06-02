import torch
from torch.utils.data import Dataset
from torch.tensor import Tensor
import numpy as np
from typing import Tuple

array = np.ndarray
LEN = 151

fMin = 10.75 # GHz
fMax = 10.95 # GHz

class BGSDynamicDataSet(Dataset):
    """
    运行时模拟生成数据, 每次都不一样
    """

    def __init__(self, size=100, n=10):
        """
        :param size: 模拟的数据总量
        :param N : 一个数据由N个BGS组成
        """
        self.n = n
        self.size = size

    def __getitem__(self, item):
        """
        :return: (Tensor[1, 151, N], Tensor[N])
        """
        bgs, bfs, sw = create_data(self.n)
        return bgs, bfs

    def __len__(self):
        return self.size


def create_data(
    n: int
) -> Tuple[Tensor, Tensor, Tensor]:

    bgs = np.zeros((n, LEN), dtype=np.float64)
    bfs = get_bfs(n=n)
    sw = get_sw(n=n)
    snr = get_snr(n=n)

    for i in range(n):
        peak_gain = 1
        bgs[i] = create_bgs(
            peak_gain=peak_gain,
            sw=sw[i],
            bfs=bfs[i]
        )

        bgs[i] = awgn(data=bgs[i], snr=snr[i])
        bgs[i] = bgs[i] / np.max(bgs[i])
    bgs = bgs.T
    bfs, sw = normalization(bfs=bfs, sw=sw)
    return torch.tensor([bgs], dtype=torch.float), \
           torch.tensor(bfs, dtype=torch.float), \
           torch.tensor(sw, dtype=torch.float)


def create_bgs(
    peak_gain: float,
    sw: float,
    bfs: float
) -> array:
    res = np.array([])
    for item in np.linspace(fMin, fMax, LEN):
        tmp = peak_gain / (1 + np.square((item - bfs) / (sw / 2)))
        res = np.append(res, tmp)
    return res


def get_bfs(
    n: int
) -> array:
    return fMin + (fMax - fMin) * np.random.uniform(0.05, 0.95, n)


def get_sw(
    n: int
) -> array:
    return (fMax - fMin) * np.random.uniform(0.1, 0.5, n)


def get_snr(
    n: int
) -> array:
    return np.random.uniform(5, 20, n)


def awgn(
    data: array,
    snr: float
) -> array:
    _signal = np.sum(data ** 2) / len(data)
    _noise = _signal / 10 ** (snr / 10)

    noise = np.random.randn(len(data)) * np.sqrt(_noise)
    return data + noise


def normalization(
    bfs: array,
    sw: array
) -> Tuple[array, array]:
    return (bfs - fMin) / (fMax - fMin), sw / (fMax - fMin)

if __name__ == '__main__':
    pass
