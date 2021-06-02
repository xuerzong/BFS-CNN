from src.models import BFSCNN
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import leastsq, curve_fit
from typing import Any, Tuple
import pickle
import torch
from torch.utils.data import DataLoader
from test.dataset import BGSTestDataset



array = np.ndarray
LEN = 201

maxB = 10950
minB = 10750


def bfs_cnn(
    cnn_model: BFSCNN,
    bgs: array
) -> array:
    
    dataset = BGSTestDataset(bgs=bgs, size=1)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    res = np.array([])

    device = "cuda"
    cnn_model.eval()
    x = test_loader[0].to(device)
    output = cnn_model(x)
    output = output.view(output.size(0), -1)
    res = output.cpu().data.numpy()


    return res


def lorentz(
    p: array,
    x: array
) -> array:
    return p[0] / (1 + (x - p[1])**2 / (p[2] / 2))


def error_func(
    p: array,
    x: array,
    z: Any
) -> Any:
    return z - lorentz(p, x)

# 洛伦兹拟合


def lorentz_fit(
    x: array,
    y: array
) -> Tuple[array, array]:
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


def lcf(
    data: array
) -> float:
    """洛伦兹拟合
    :param data: 需要拟合的曲线，即一条布里渊频谱
    :returns: 拟合得出的布里渊频移
    """
    x = np.arange(len(data))
    tmp, solp = lorentz_fit(x, data)
    return solp[1]


def shift2temperature(
    shift: float
) -> float:
    return (shift - 10814.6526) / 0.97487


def shifts2temperatures(
    shifts: array
) -> array:
    return (shifts - 10814.6526) / 0.97487


def get_diff_temp_data(
    frequency: int,
    model: BFSCNN
):
    """获取不同温度的数据，生成.pkl文件
    :param frequency: 采样频率
    :returns None
    """

    def get_path(
        temperature: int,
        frequency: int
    ) -> str:
        return f'test/data/Testing Data/{temperature}DegC/{frequency}MHz/'

    temps = [30, 40, 50, 60]

    res = np.array([{} for x in temps])

    for index, item in enumerate([get_path(x, frequency) for x in temps]):
        lcf_res, cnn_res = get_shifts(path=item, model=model)
        res[index] = {
            'lcf_res': lcf_res,
            'cnn_res': cnn_res
        }

    with open('diff_temp_data.pkl', 'wb') as f:
        pickle.dump(res, f)


def get_diff_trace_data(
    frequency: int,
    model: BFSCNN
):
    """获取45摄氏度下不同trace的数据，生成.pkl文件
    :param frequency: 采样频率
    :returns: None
    """

    def get_path(
        trace: int,
        frequency: int
    ) -> str:
        return f'test/data/AVG45DegC/AVG45D{trace}/{frequency}MHz/'

    temps = [250, 500, 1000, 2000]

    res = np.array([{} for x in temps])

    for index, item in enumerate([get_path(x, frequency) for x in temps]):
        lcf_res, cnn_res = get_shifts(path=item, model=model)
        res[index] = {
            'lcf_res': lcf_res,
            'cnn_res': cnn_res
        }

    with open('diff_trace_data.pkl', 'wb') as f:
        pickle.dump(res, f)



def plot_res(
    data: array,
    type: str
) -> None:
    """画图
    :param data: 数据
    :param type: 类型 'temperature': 不同温度 | 'trace': 45摄氏度下不同trace
    :returns: None
    """

    def plot_shift(
        filename: str,
        start: str = None,
        end: str = None
    ):
        plt.figure()
        for item in data:
            lcf_res, cnn_res = item['lcf_res'], item['cnn_res']
            plt.plot(np.arange(len(cnn_res[start:end])),
                     cnn_res[start:end], label=f'cnn')
            plt.plot(np.arange(len(lcf_res[start:end])),
                     lcf_res[start:end], label=f'lcf')
        plt.legend()
        plt.xlabel('Distance(m)')
        plt.ylabel('BFS(MHz)')
        plt.savefig(filename)

    # all

    plot_shift(
        filename=f'{type}_all.png',
        start=30000
    )

    plot_shift(
        filename=f'{type}_200.png',
        start=-200
    )

    plot_shift(
        filename=f'{type}_sr.png',
        start=-75,
        end=-35
    )


    plt.figure()
    for item in data:
        lcf_res, cnn_res = item['lcf_res'], item['cnn_res']
        plt.plot(np.arange(len(cnn_res[-200:])),
                 shifts2temperatures(cnn_res[-200:]), label=f'cnn')
        plt.plot(np.arange(len(lcf_res[-200:])),
                 shifts2temperatures(lcf_res[-200:]), label=f'lcf')
    plt.legend()
    plt.xlabel('Distance(m)')
    plt.ylabel('Temperature(℃)')
    plt.savefig(f'{type}_200_temperature.png')

    cnn_rmse = [] 
    cnn_sd = []
    lcf_rmse = []
    lcf_sd = []

    for index, item in enumerate(data):
        lcf_res, cnn_res = item['lcf_res'], item['cnn_res']
        lcf_temp = shifts2temperatures(shifts=lcf_res[-50:])
        cnn_temp = shifts2temperatures(shifts=cnn_res[-50:])

        now_temperatures = [30, 40, 50,
                            60] if type == 'temperature' else [45, 45, 45, 45]


        a1, b1 = RMSE(cnn_temp, np.array(
            [now_temperatures[index] for x in cnn_temp])), SD(cnn_temp)
        
        cnn_rmse.append(a1)
        cnn_sd.append(b1)

        a2, b2 = RMSE(lcf_temp, np.array(
            [now_temperatures[index] for x in lcf_temp])), SD(lcf_temp)
        lcf_rmse.append(a2)
        lcf_sd.append(b2)
    x, xlabel = (
        [30, 40, 50, 60],
        'Temperature(℃)'
    ) if type == 'temperature' else (
        [250, 500, 1000, 2000],
        'Trace'
    )

    plt.figure()
    plt.plot(x, cnn_rmse, label='cnn', marker='o')
    plt.plot(x, lcf_rmse, label='lcf', marker='x')
    plt.xlabel(xlabel)
    plt.ylabel('RMSE(℃)')
    plt.legend()
    plt.savefig(f'{type}_rmse.png')

    plt.figure()
    plt.plot(x, cnn_sd, label='cnn', marker='o')
    plt.plot(x, lcf_sd, label='lcf', marker='x')
    plt.xlabel(xlabel)
    plt.ylabel('SD(℃)')
    plt.legend()
    plt.savefig(f'{type}_sd.png')


    cnn_sr = []
    lcf_sr = []

    for item in data:
        lcf_res, cnn_res = item['lcf_res'], item['cnn_res']
        start = -64
        end = -61
        cnn_sr.append(get_SR(cnn_res[-120:-70], cnn_res[start:end], cnn_res[-60:-10]))
        lcf_sr.append(get_SR(lcf_res[-120:-70], lcf_res[start:end], lcf_res[-60:-10]))

    print('##### cnn #####') 
    print(cnn_sr)

    print('##### lcf #####')
    print(lcf_sr)



def get_shifts(
    path: str,
    model: BFSCNN
) -> Tuple[array, array]:
    """获取布里渊频移结果
    :param path: 真实值路径
    :returns: 洛伦兹拟合结果，卷积神经网络拟合结果
    """

    # 获取真实数据.dat文件
    dir_list = os.listdir(path)
    dir_data = []

    for item in dir_list:
        if item.split('.')[1] == 'dat':
            dir_data.append(item)

    dir_data.sort()

    # 获取每个.dat文件中的值
    real_data = np.zeros((len(dir_data), 125001), dtype=array)

    for i in range(len(dir_data)):
        data = np.fromfile(f'{path}{dir_data[i]}', dtype='f8')
        real_data[i] = data

    # 处理数据
    for i in range(1, len(real_data)):
        real_data[i] = real_data[i] - np.min(real_data[i])

    # 提取布里渊频移信息
    lcf_res = np.array([])
    cnn_res = np.array([])

    # 提取部分数据进行处理，避免数据量太大导致处理时间较长
    count = 0
    cnn_count = 0
    cnn_input = np.array([])

    for item in real_data.T:

        if count > 55340 and count <= 88940:
            data_item = np.array(item, dtype=np.float64)

            #  对data_item进行切片，使得长度为151，以适应BFSCNN的输入
            data_item = data_item[25:-25]

            lcf_tmp = lcf(data=data_item)
            cnn_tmp = np.array([])

            if cnn_count == 223:
                cnn_tmp = bfs_cnn(cnn_model=model, bgs=cnn_input.T)
                cnn_count = 0
                cnn_input = np.array([])
            else:
                cnn_input = np.append(cnn_input, data_item)
                cnn_count += 1

            lcf_res = np.append(lcf_res, lcf_tmp)
            cnn_res = np.append(cnn_res, cnn_tmp)

        count += 1

    return lcf_res, cnn_res


def get_SR(
    start: array,
    middle: array,
    end: array
) -> float:

    """求取空间分辨率
    :param start: 发生频移前取值段
    :param middle: 上升沿取值段
    :param end: 发生频移后取值段
    :returns: 空间分辨率
    """

    def f(x, A, B):
        return A * x + B

    before_shift = np.average(start)
    after_shift = np.average(end)

    # 拟合直线
    A, B = curve_fit(f, np.arange(len(middle)), middle)[0]

    start_loc = (before_shift - B) / A
    end_loc = (after_shift - B) / A

    # 空间分辨率取10%~90%
    SR = (end_loc - start_loc) * 0.8

    return SR


def get_sd_rmse(
    data: array,
    dataH: array
) -> Tuple[array, array]:
    _rmse = np.array([])
    _sd = np.array([])
    for i in range(len(data)):
        _rmse = np.append(_rmse, RMSE(data[i], dataH[i]))
        _sd = np.append(_sd, SD(data[i]))

    return _sd, _rmse


def RMSE(data, dataH):
    return np.sqrt(np.square(data - dataH).mean())


def SD(data):
    return np.std(data, ddof=1)


if __name__ == "__main__":

    # 模型测试集
    cnn_model: BFSCNN
    if os.path.exists('cnn_model.pkl'):
        cnn_model = torch.load('cnn_model.pkl')
    else:
        print('There is no file named cnn_model.pkl')

    res_data = None

    if not os.path.exists('diff_temp_res.pkl'):
        # 获取1MHz的数据
        get_diff_temp_data(frequency=1, model=cnn_model)

    with open('diff_temp_res.pkl', 'rb') as f:
        res_data = pickle.load(f)
    plot_res(data=res_data, type='temperature')

    res_data = None

    if not os.path.exists('diff_trace_res.pkl'):
        # 获取1MHz的数据
        get_diff_trace_data(frequency=1, model=cnn_model)

    with open('diff_trace_res.pkl', 'rb') as f:
        res_data = pickle.load(f)
    plot_res(data=res_data, type='trace')