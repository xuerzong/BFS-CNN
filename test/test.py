import numpy as np
from typing import List, Any
from matplotlib import pyplot as plt


array = np.ndarray


def create_data(data: float, peak_gain: float, line_width: float, bfs: float) -> float:
    """
    :param data: 参数x
    :param peak_gain: 峰值
    :param line_width: 半高全宽
    :param bfs: BFS
    :return: 参数y
    """

    _bfs = np.random.uniform(0.05, 0.95)
    _sw = np.random.uniform(0.1, 0.5)
    _snr = np.random.uniform(5, 20)

    return peak_gain / (1 + np.square((data - bfs) / (line_width / 2)))


def awgn(data: array, snr: float) -> array:
    """
    加入高斯白噪声 Additive White Gaussian Noise
    :param data: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    """
    _signal = np.sum(data**2)/len(data)
    _noise = _signal/10**(snr/10)

    noise = np.random.randn(len(data)) * np.sqrt(_noise)
    return data + noise


def normalization(data: array) -> array:
    """
    BFS 归一化处理
    :param data: 原始数据
    :return: 处理后的数据
    """
    _min = np.min(data)
    _max = np.max(data)

    print(_max, _min)

    return (data - _min) / (_max - _min)


if __name__ == '__main__':
    a = [1, -1, 2, -7, 8]
    x = np.array(a)

    print(x)
    # print(a)
    # print(add_noise(b, 20))
    # xn = prepare_data(b)

    # print(np.sum(abs(np.array(a))))
    LEN = 151
    _x = np.arange(0, LEN)
    print(_x)
    y = np.array([])
    for i in _x:
        # bfs           5%～95%
        # line_width    10%～50%
        # peak_gain     5dB～20dB 信号最大增益和噪声功率之间的比值
        y = np.append(y, create_data(data=i, peak_gain=10, line_width=0.13*LEN, bfs=0.5*LEN))
    yn = awgn(y, 20)
    yn = normalization(yn)
    print(yn)
    plt.plot(_x/150, yn, label="sigmoid") # _x/150: 归一化
    plt.ylim(-0.1, 1.1)
    plt.show()
