import torch
from torch.utils.data import Dataset
from torch.tensor import Tensor
import numpy as np
from scipy.optimize import leastsq
import math
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

array = np.ndarray
LEN = 151

fMin = 10.6 * math.pow(10,3)
fMax = 10.9 * math.pow(10,3)

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

    bgs = np.zeros((n, LEN), dtype=array)
    bfs = get_bfs(n=n)
    sw = get_sw(n=n)

    for i in range(n):
        peak_gain = 1
        bgs[i] = create_bgs(
            peak_gain=peak_gain,
            sw=sw[i],
            bfs=bfs[i]
        )
        # tmp[i] = awgn(tmp[i])  # 添加白噪声
    bgs = bgs.T
    return bgs
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
    return np.linspace(fMin + (fMax - fMin)*0.05, fMin + (fMax - fMin)*0.95, n)
    return np.random.uniform(fMin + (fMax - fMin)*0.05, fMin + (fMax - fMin)*0.95, n)


def get_sw(
    n: int
) -> array:
    return np.random.uniform((fMax - fMin) * 0.1, (fMax - fMin) * 0.5, n)


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


def lorentz(p: array, x: array) -> float:
    return p[0] / ((x - p[1]) ** 2 + p[2])


def error_func():
    return None


def lorentz_fit(x: array, y: array) -> Tuple[float, array]:
    p3 = ((np.max(x) - np.min(x)) / 10) ** 2
    p2 = (np.max(x) + np.min(x)) / 2
    p1 = np.max(y) * p3

    c = np.min(y)

    p0 = np.array([p1, p2, p3, c], dtype=float)

    solp, ier = leastsq(
        func=error_func,
        x0=p0,
        args=(x, y),
        maxfev=200000
    )

    return lorentz(solp, x), solp


def get_snr(
    data: array,            # 被测数据
    data_lorentz: array,    # 经过洛伦兹拟合过的被测数据
    solp: array             # 拟合后的参数
) -> float:
    _max = lorentz(solp, solp[1])
    _variance = np.var(data - data_lorentz)
    _snr = _max ** 2 / _variance
    snr = 10 * np.log10(_snr)
    return snr


if __name__ == '__main__':
    pass

