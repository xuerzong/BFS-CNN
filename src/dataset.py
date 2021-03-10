import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


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


def create_data(n):
    """
    模拟生成数据
    :param n: n条轨道的BGS组成一个数据
    :return: (Tensor[1, 151, N], Tensor[N])
            分别对应BGS 和 BFS
    """
    # 思路：随机模拟BFS信号，生成对应的BGS
    # 1. 随机生成一组BFS， 形状[N]
    bfs = np.random.uniform(0.05, 0.95, n)
    # 2. TODO 每一个BFS生成对应的BGS, 形状[151, N]
    bgs = np.zeros((151, n), dtype=float)
    # 3. TODO 添加白噪声
    # 4。TODO 根据频率进行归一化
    return torch.tensor([bgs], dtype=torch.float), torch.tensor(bfs, dtype=torch.float)


if __name__ == '__main__':
    dataset = BGSDynamicDataSet()
    loader = DataLoader(dataset=dataset,
                        shuffle=False,
                        batch_size=1,
                        num_workers=2
                        )
    for i, data in enumerate(loader):
        print(data)
        break
        # if i % 20 == 0:
        #     print(data)
        #     print('{}/{}'.format(i * 3, len(dataset)))
