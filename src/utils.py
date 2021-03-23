import csv
import os
import numpy as np
from typing import List

array = np.ndarray


def create_csv(filename: str, csv_head: List[str]) -> None:
    """
    :param filename: 文件名
    :param csv_head: csv头部信息
    :return: None
    """
    if not os.path.exists('data'):
        os.mkdir('data')
    path = f'data/{filename}.csv'
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)


def write_csv(filename: str, data: array, flag: bool = True) -> None:
    """
    :param filename: 文件名
    :param data: 需要写入的数据
    :param flag: True: 多行写入(默认) False: 单行写入
    """
    path = f'data/{filename}.csv'
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        if flag:
            csv_write.writerows(data)
        else:
            csv_write.writerow(data)