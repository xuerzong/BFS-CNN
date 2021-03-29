import torch
from torch.utils.data import Dataset
from torch.tensor import Tensor
import numpy as np
from typing import Tuple

array = np.ndarray
LEN = 151


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
    bfs = np.random.uniform(0.05, 0.95, n) * LEN
    tmp = np.zeros((n, LEN), dtype=array)
    sw = np.array([])
    for i in range(n):
        line_width = np.random.uniform(0.1, 0.5) * LEN
        sw = np.append(sw, line_width)
        peak_gain = 10
        tmp[i] = create_bgs(
            data=tmp[i],
            peak_gain=peak_gain,
            sw=line_width,
            bfs=bfs[i]
        )  # 生成理想BGS
        tmp[i] = awgn(tmp[i])  # 添加白噪声
    bgs = tmp.T
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
    for item in range(len(data)):
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
    return bfs / (LEN - 1), sw / (LEN - 1)


if __name__ == '__main__':
    a = create_data(3)
    print(a)
