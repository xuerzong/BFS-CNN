# TODO 洛伦兹拟合 通过使用洛伦兹拟合，判断BFS-CNN的结果是否实现

import numpy as np
import torch
from torch.tensor import Tensor
from torch.utils.data import DataLoader, Dataset
import os
from typing import List, Any, Tuple
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from torch import nn
from test.dataset import create_data

fMin = 10.6 # GHz
fMax = 10.9 # GHz
LEN = 151

SIZE = 1

array = np.ndarray


class BGSTestDataset(Dataset):
    def __init__(self, bgs: array, size: int):
        self.bgs = bgs
        self.size = size

    def __getitem__(self, item):
        return torch.tensor([self.bgs], dtype=torch.float)

    def __len__(self):
        return self.size


def lorentz(
    p: array,
    x: array
) -> array:
    return p[0] / ((x - p[1]) ** 2 + p[2])


def error_func(p: array, x: array, z: Any) -> Any:
    return z - lorentz(p, x)

# 洛伦兹拟合
def lorentz_fit(x: array, y: array) -> Tuple[array, array]:
    p3 = ((np.max(x) - np.min(x)) / 10) ** 2
    p2 = (np.max(x) + np.min(x)) / 2
    p1 = np.max(y) * p3

    c = np.min(y)

    p0 = np.array([p1, p2, p3, c], dtype=np.float64)

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



def bfs_cnn(
    bgs: array
) -> array:
    cnn_model: nn.Module
    res = np.array([])

    if os.path.exists('model.pkl'):
        cnn_model = torch.load('model.pkl', map_location="cpu")
    else:
        print('There is not a file named "model.pkl"')
        return
    
    dataset = BGSTestDataset(bgs=bgs, size=1)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    device = "cpu"

    cnn_model.to(device)
    cnn_model.eval()

    for i, x in enumerate(test_loader):
        x.to(device)
        output = cnn_model(x)
        output = output.view(output.size(0), -1)
        res = np.append(res, output.detach().numpy())

    return res

def lcf(
    data: array
) -> array:
    res = 0
    x = np.arange(LEN)

    tmp, solp = lorentz_fit(x, data)

    for i in range(len(tmp)):
        if tmp[i] == np.max(tmp):
            res =  i / LEN
            break

    return res


def get_sd_rmse(data: array, dataH: array) -> Tuple[array, array]:
    _rmse = np.array([])
    _sd = np.array([])
    for i in range(len(data)):
        _rmse = np.append(_rmse, RMSE(data[i], dataH[i]))
        _sd = np.append(_sd, SD(data[i]))
    
    return _sd, _rmse


def RMSE(data, dataH):
    return np.sqrt(np.square(data - dataH).mean())


def SD(data):
    return np.std(data, ddof=0)

if __name__ == '__main__':

    test_arr = ['sw', 'bfs', 'snr']

    n = 224

    for item in test_arr:
        bgss, bfs, lj = create_data(item, n)

        bgss_cnn = np.array([bgss[i:i+n] for i in range(0, len(bgss), n)], dtype=np.float64)
        cnn_res = np.zeros((len(bgss_cnn), n), dtype=np.float64)
        lcf_res = np.zeros((len(bgss_cnn), n), dtype=np.float64)
    
        for i in range(len(bgss_cnn)):

            for j in range(len(bgss_cnn[i])):
                lcf_res[i][j] = lcf(bgss_cnn[i][j])

            cnn_res[i] = bfs_cnn(bgs=bgss_cnn[i].T)
    
        _bfs = np.array([[bfs[i]] * n for i in range(len(bfs))])

        a1, b1 = get_sd_rmse(cnn_res, _bfs)
        a2, b2 = get_sd_rmse(lcf_res, _bfs)

        x = np.arange(len(a1))

        x_label = ''

        if item == 'bfs':
            x = [i * 5 + 10 for i in x]
            x_label = 'Normalized BFS(%)'
        elif item == 'sw':
            x = [i * 5 + 10 for i in x]
            x_label = 'Normalized SW(%)'
        elif item == 'snr':
            x = [i * 2 + 5 for i in x]
            x_label = 'SNR(db)'


        plt.figure()
        plt.xlabel(x_label)
        plt.ylabel('SD(%)')
        plt.plot(x, a1*100, label="cnn", marker='o')
        plt.plot(x, a2*100, label="lcf", marker='o')
        plt.legend()
        plt.savefig(f'assets/sd_{item}')
        
        plt.figure()
        plt.xlabel(x_label)
        plt.ylabel('RMSE(%)')
        plt.plot(x, b1*100, label="cnn", marker='o')
        plt.plot(x, b2*100, label="lcf", marker='o')
        plt.legend()
        plt.savefig(f'assets/rmse_{item}')

