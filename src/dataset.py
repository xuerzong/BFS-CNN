import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

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
        根据item返回BGS模拟数据
        :return: Tenor[1, 151, W]
        """
        return create_data(self.n)

    def __len__(self):
        return self.size


class BGSFixedDataSet(Dataset):
    """
    TODO 将模拟的数据保存到文件中，使用同一批数据
    """

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


def create_data(n: int):
    """
    模拟生成数据
    :param n: n条轨道的BGS组成一个数据
    :return: (Tensor[1, 151, N], Tensor[N])
            分别对应BGS 和 BFS
    """
    # 思路：随机模拟BFS信号，生成对应的BGS
    bfs = np.random.uniform(0.05, 0.95, n) * LEN
    # bgs = np.zeros((LEN, n), dtype=float)
    tmp = np.zeros((n, LEN), dtype=array)
    for i in range(n):
        line_width = np.random.uniform(0.1, 0.5) * LEN
        peak_gain = 10
        _bfs = bfs[i]
        tmp[i] = create_bgs(data=tmp[i], peak_gain=peak_gain, line_width=line_width, bfs=_bfs)
        tmp[i] = awgn(tmp[i])
        tmp[i] = frequency_normalization(tmp[i])
    bgs = tmp.T
    # return bgs
    return torch.tensor([bgs], dtype=torch.float), torch.tensor(bfs, dtype=torch.float)


def create_bgs(data: array, peak_gain: float, line_width: float, bfs: float) -> array:
    y = np.array([])
    for item in range(len(data)):
        tmp = peak_gain / (1 + np.square((item - bfs) / (line_width / 2)))
        y = np.append(y, tmp)
    return y


def awgn(data: array) -> array:
    """
    加入高斯白噪声 Additive White Gaussian Noise
    :param data: 原始信号
    :return: 加入噪声后的信号
    """
    snr = np.random.uniform(5, 20)
    _signal = np.sum(data ** 2) / len(data)
    _noise = _signal / 10 ** (snr / 10)

    noise = np.random.randn(len(data)) * np.sqrt(_noise)
    return data + noise


def frequency_normalization(data: array) -> array:
    """
    BFS 归一化处理
    :param data: 原始数据
    :return: 处理后的数据
    """
    _min = np.min(data)
    _max = np.max(data)

    return (data - _min) / (_max - _min)


if __name__ == '__main__':
    a = create_data(200)
    # a = a.T
    print(len(a))
    for i in range(80):
        plt.plot(np.arange(len(a[i])), a[i])
    plt.ylim(-0.1, 1.1)
    plt.show()
    # dataset = BGSDynamicDataSet()
    # loader = DataLoader(dataset=dataset,
    #                     shuffle=False,
    #                     batch_size=1,
    #                     num_workers=2
    #                     )
    # for i, data in enumerate(loader):
    #     print(data)
    #     break
        # if i % 20 == 0:
        #     print(data)
        #     print('{}/{}'.format(i * 3, len(dataset)))
