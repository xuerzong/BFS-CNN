# TODO 洛伦兹拟合 通过使用洛伦兹拟合，判断BFS-CNN的结果是否实现

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from typing import List, Any, Tuple
from matplotlib import pyplot as plt

from test.dataset import BGSTestDataset

array = np.ndarray


def lorentz(
    p: array,
    x: array
) -> float:
    return p[0] / ((x - p[1]) ** 2 + p[2])


def error_func(p: array, x: array, z: Any) -> Any:
    return z - lorentz(p, x)

# 洛伦兹拟合
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



def bfs_cnn(
    dataset: DataLoader
) -> array:

    res = np.zeros(151, dtype=array)

    print(os.path.exists('dataset.py'))
    return

    if os.path.exists('model.pkl'):
        cnn_model = torch.load('model.pkl')
    else:
        print('There is not a file named "model.pkl"')
        return None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cnn_model.to(device)
    cnn_model.eval()

    for i, x in enumerate(dataset):
        x, target = x
        x, target = x.to(device), target.to(device)

        output = cnn_model(x)
        output = output.view(output.size(0), -1)
        
        res[i] = output.numpy()

    return res

def lcf(dataset: DataLoader):
    pass


def get_sd_rmse(data: array, dataH: array) -> Tuple[array, array]:
    _rmse = np.array([])
    _sd = np.array([])
    for item in data:
        _rmse = np.append(_rmse, RMSE(item, dataH))
        _sd = np.append(_sd, SD(item))
    
    return _sd, _rmse


def analyze_data():
    bgss, bfs = BGSTestDataset()

    rmse = np.zeros(2, dtype=array)
    sd = np.zeros(2, dtype=array)

    for bgs in bgss:
        data_loader = DataLoader(dataset=bgs)

        cnn_res = bfs_cnn(data_loader)
        lcf_res = lcf(data_loader)

        cnn_rmse, cnn_sd = get_sd_rmse(cnn_res, bfs)
        lcf_rmse, lcf_sd = get_sd_rmse(lcf_res, bfs)


        rmse[0] = np.append(rmse[0], cnn_rmse.mean())
        rmse[1] = np.append(rmse[1], lcf_rmse.mean())

        sd[0] = np.append(sd[0], cnn_sd.mean())
        sd[1] = np.append(sd[1], lcf_sd.mean())

    return rmse, sd



def RMSE(data, dataH):
    return np.sqrt(np.square(data - dataH).mean())


def SD(data):
    return np.std(data, ddof=0)

if __name__ == '__main__':
    rmse, sd = analyze_data()

    print(rmse, sd)
