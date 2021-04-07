import torch
from torch.utils.data import Dataset
from torch.tensor import Tensor
import numpy as np
from typing import Tuple

array = np.ndarray
LEN = 151

bfsMin = 10.85 # GHz  截取最小频率
bfsMax = 10.95 # GHz  截取最大频率
bfsRange = bfsMax - bfsMin

swMin = 0.02 # GHz
swMax = 0.06 # GHz
swRange = swMax - swMin

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


def create_data(n: int) -> Tuple[Tensor, Tensor, Tensor]:
    bfs = np.random.uniform(bfsMin + bfsRange*0.05, bfsMax - bfsRange*0.05, n)
    sw = np.random.uniform(swMin + swRange * 0.01, swMax - swRange * 0.05, n)
    tmp = np.zeros((n, LEN), dtype=array)

    bfs, sw = normalization(bfs=bfs, sw=sw)
    for i in range(n):
        peak_gain = 1
        tmp[i] = create_bgs(
            data=tmp[i],
            peak_gain=peak_gain,
            sw=sw[i],
            bfs=bfs[i]
        )  # 生成理想BGS
        # tmp[i] = awgn(tmp[i])  # 添加白噪声
    bgs = tmp.T
    return tmp
    bgs = bgs / np.max(bgs)
    bfs, sw = normalization(bfs, sw)
    return torch.tensor([bgs], dtype=torch.float), \
           torch.tensor(bfs, dtype=torch.float), \
           torch.tensor(sw, dtype=torch.float)


def create_bgs(
        data: array,
        peak_gain: float,
        sw: float,
        bfs: float
) -> array:
    y = np.array([])
    for item in np.linspace(0, 1, len(data)):
        tmp = peak_gain / (1 + np.square((item - bfs) / (sw / 2)))
        y = np.append(y, tmp)
    return y


def awgn(data: array) -> array:
    snr = np.random.uniform(5, 20)
    _signal = np.sum(data ** 2) / len(data)
    _noise = _signal / 10 ** (snr / 10)

    noise = np.random.randn(len(data)) * np.sqrt(_noise)
    return data + noise


def normalization(bfs: array, sw: array) -> Tuple[array, array]:
    return (bfs - bfsMin) / bfsRange, sw / bfsRange


if __name__ == '__main__':
    pass